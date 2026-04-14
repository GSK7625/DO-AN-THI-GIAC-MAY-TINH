import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
from torchvision import transforms
from PIL import Image

# Import model CSRNet 
from model import CSRNet

def evaluate_csrnet(model_path, data_root, save_dir, limit=5):
    """
    Đánh giá mô hình CSRNet trên tập test và trực quan hoá kết quả (Ground Truth vs Prediction)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Khởi tạo mô hình
    model = CSRNet(load_weights=False).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded CSRNet weights from: {model_path}")
    else:
        print(f"WARNING: Weights not found at {model_path}. Running with random weights.")
    
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    import glob
    test_img_paths = sorted(glob.glob(os.path.join(data_root, "part_A", "test_data", "images", "*.jpg")))
    
    mae, mse = 0.0, 0.0
    count = 0
    
    # Lấy một số ảnh mẫu để trực quan hoá
    sample_paths = test_img_paths[:limit]
    
    fig, axes = plt.subplots(len(sample_paths), 3, figsize=(15, 5*len(sample_paths)))
    
    print("Starting evaluation and visualization...")
    for i, img_path in enumerate(sample_paths):
        # Đọc ảnh gốc
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # Đọc Ground Truth
        h5_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth')
        with h5py.File(h5_path, 'r') as hf:
            gt_map = np.asarray(hf['density'])
        gt_count = np.sum(gt_map)
        
        # Dự đoán
        with torch.no_grad():
            output = model(img_tensor)
            pred_map = output.squeeze().cpu().numpy()
            
        pred_count = np.sum(pred_map)
        
        # Tính toán sai số
        diff = gt_count - pred_count
        mae += abs(diff)
        mse += diff**2
        count += 1
        
        # Trực quan hoá
        ax_img = axes[i, 0] if limit > 1 else axes[0]
        ax_gt = axes[i, 1] if limit > 1 else axes[1]
        ax_pred = axes[i, 2] if limit > 1 else axes[2]
        
        ax_img.imshow(img_pil)
        ax_img.set_title(f"Ảnh Gốc\n{os.path.basename(img_path)}")
        ax_img.axis('off')
        
        ax_gt.imshow(gt_map, cmap=cm.jet)
        ax_gt.set_title(f"Ground Truth\nTổng số người: {gt_count:.1f}")
        ax_gt.axis('off')
        
        ax_pred.imshow(pred_map, cmap=cm.jet)
        ax_pred.set_title(f"Dự Đoán (CSRNet)\nTổng số người: {pred_count:.1f}")
        ax_pred.axis('off')
        
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "csrnet_prediction_samples.png")
    plt.savefig(save_path)
    
    print("-" * 50)
    print(f"Saved visualization to: {save_path}")
    print(f"MAE (over {limit} samples): {mae/count:.2f}")
    print(f"RMSE (over {limit} samples): {np.sqrt(mse/count):.2f}")
    print("-" * 50)

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "ShanghaiTech")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "csrnet", "last.pth")
    SAVE_DIR = os.path.join(PROJECT_ROOT, "results", "csrnet")
    
    evaluate_csrnet(MODEL_PATH, DATA_ROOT, SAVE_DIR, limit=5)
