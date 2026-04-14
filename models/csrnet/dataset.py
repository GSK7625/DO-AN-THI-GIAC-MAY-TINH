import os
import glob
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class CrowdDataset(Dataset):
    def __init__(self, root_dir, part='A', split='train', transform=None):
        """
        root_dir: Thư mục chứa dataset (ví dụ: 'data/ShanghaiTech')
        part: 'A' hoặc 'B'
        split: 'train' hoặc 'test'
        """
        self.root_dir = root_dir
        self.part = part
        self.split = split
        self.transform = transform
        
        self.img_paths = sorted(glob.glob(os.path.join(root_dir, f"part_{part}", f"{split}_data", "images", "*.jpg")))
        
        if self.transform is None:
            # Áp dụng chuẩn hóa theo ImageNet (vì dùng VGG16 pretrained)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        h5_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth')
        
        # Load ảnh
        img = Image.open(img_path).convert('RGB')
        
        # Load density map
        with h5py.File(h5_path, 'r') as hf:
            target = np.asarray(hf['density'])
            
        if self.transform:
            img = self.transform(img)
            
        target = torch.from_numpy(target).unsqueeze(0)  # Shape: (1, H, W)
        
        return img, target

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "ShanghaiTech")
    
    print(f"Data root: {DATA_ROOT}")
    dataset = CrowdDataset(DATA_ROOT, part='A', split='train')
    
    if len(dataset) > 0:
        img, target = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Density map shape: {target.shape}")
        print(f"Total count (ground truth): {target.sum().item():.2f}")
        print("Data loaded successfully!")
    else:
        print("No images found. Please check paths.")
