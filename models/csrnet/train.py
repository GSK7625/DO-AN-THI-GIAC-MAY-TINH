import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import CSRNet
from dataset import CrowdDataset
from tqdm import tqdm

def train(trial=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị (device): {device}")
    
    model = CSRNet(load_weights=True).to(device)
    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # Cấu hình đường dẫn
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "ShanghaiTech")
    
    # Dataset & DataLoader
    train_dataset = CrowdDataset(DATA_ROOT, part='A', split='train')
    
    if trial:
        train_dataset.img_paths = train_dataset.img_paths[:5]
        
    print(f"Tổng số ảnh trong tập huấn luyện: {len(train_dataset)}")    
    
    # CSRNet thường huấn luyện với batch_size=1 vì kích thước các ảnh trong ShanghaiTech khác nhau
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    epochs = 1 if trial else 10
    
    save_dir = os.path.join(PROJECT_ROOT, "results", "csrnet")
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        print(f"\n[ Epoch {epoch+1}/{epochs} ]")
        
        # Sử dụng tqdm để hiển thị thanh tiến độ
        for img, target in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            img = img.to(device)
            target = target.to(device)
            
            output = model(img)
            
            # Kích thước đầu ra của CSRNet là 1/8 của đầu vào (do 3 lớp MaxPool)
            # Trọng số target (density map) cần được thu nhỏ để khớp với đầu ra
            # Nhân với 64 (8x8) để bảo toàn tổng số người sau khi thu nhỏ kích thước không gian
            target_resized = F.interpolate(
                target, 
                size=(output.shape[2], output.shape[3]), 
                mode='bilinear', 
                align_corners=False
            ) * 64.0
            
            loss = criterion(output, target_resized)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"=> Training Loss trung bình: {avg_loss:.4f}")
        
        # Lưu checkpoint tại mỗi epoch
        torch.save(model.state_dict(), os.path.join(save_dir, "last.pth"))
            
    print("\nHuấn luyện CSRNet hoàn tất!")

if __name__ == '__main__':
    # Huấn luyện mô hình CSRNet đầy đủ (10 epochs)
    train(trial=False)
