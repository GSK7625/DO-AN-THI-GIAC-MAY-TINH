import os
import glob
import scipy.io as sio
import cv2
from tqdm import tqdm
import yaml

def convert_mat_to_yolo(part_path, box_size=15):
    """
    Chuyển đổi file .mat (chứa tọa độ centre) sang định dạng bounding box YOLO (.txt)
    box_size: Kích thước chiều rộng/chiều cao giả định cho đầu người (pixels)
    """
    image_dir = os.path.join(part_path, "images")
    gt_dir = os.path.join(part_path, "ground-truth")
    labels_dir = os.path.join(part_path, "labels")
    
    os.makedirs(labels_dir, exist_ok=True)
    
    img_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    print(f"Starting conversion for {len(img_paths)} images in {part_path}...")
    
    converted_count = 0
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        
        base_name = os.path.basename(img_path)
        # Các file .mat có dạng GT_IMG_xxx.mat
        mat_name = base_name.replace(".jpg", ".mat").replace("IMG_", "GT_IMG_")
        mat_path = os.path.join(gt_dir, mat_name)
        txt_path = os.path.join(labels_dir, base_name.replace(".jpg", ".txt"))
        
        if not os.path.exists(mat_path):
            continue
            
        mat = sio.loadmat(mat_path)
        points = mat["image_info"][0, 0][0, 0][0]
        
        with open(txt_path, "w") as f:
            for point in points:
                x, y = point[0], point[1]
                
                # YOLO format: id_class x_center_norm y_center_norm width_norm height_norm
                x_center = float(x) / w
                y_center = float(y) / h
                box_w = float(box_size) / w
                box_h = float(box_size) / h
                
                # Cắt giới hạn tọa độ [0, 1] để không bị lỗi out-of-bounds
                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                box_w = min(box_w, 1.0)
                box_h = min(box_h, 1.0)
                
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
        converted_count += 1

def create_yaml(data_root):
    """
    Tạo cấu hình data.yaml cho YOLO
    """
    abs_data_root = os.path.abspath(data_root)
    
    yaml_content = {
        'path': abs_data_root,
        'train': 'part_A/train_data/images',
        'val': 'part_A/test_data/images', # Dùng tập test để làm validation cho baseline
        'nc': 1,
        'names': ['head']
    }
    
    yaml_path = os.path.join(data_root, 'shanghaitech_partA.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Created YOLO config at {yaml_path}")

if __name__ == "__main__":
    # Script này chạy từ thư mục gốc của project
    DATA_ROOT = os.path.join(os.getcwd(), "data", "ShanghaiTech")
    
    if not os.path.exists(DATA_ROOT):
        print(f"Directory not found: {DATA_ROOT}")
    else:
        # Xử lý cho Part A
        for split in ["train", "test"]:
            path = os.path.join(DATA_ROOT, "part_A", f"{split}_data")
            if os.path.exists(path):
                convert_mat_to_yolo(path, box_size=15)
                
        # Tạo cấu hình
        create_yaml(DATA_ROOT)
