import os
from ultralytics import YOLO

def train_yolov8(trial=False):
    print("Đang khởi tạo YOLOv8...")
    model = YOLO("yolov8n.pt")
    
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    YAML_PATH = os.path.join(PROJECT_ROOT, "data", "ShanghaiTech", "shanghaitech_partA.yaml")
    
    if not os.path.exists(YAML_PATH):
        print(f"Không tìm thấy cấu hình tại: {YAML_PATH}. Hãy chạy data/label_converter.py để tạo cấu hình.")
        return
        
    epochs = 1 if trial else 50
    batch_size = 2 if trial else 8
    
    print(f"Bắt đầu huấn luyện (chạy thử={trial})...")
    
    save_dir = os.path.join(PROJECT_ROOT, "results", "yolov8")
    os.makedirs(save_dir, exist_ok=True)
    
    results = model.train(
        data=YAML_PATH,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        project=save_dir,
        name="baseline",
        exist_ok=True
    )
    
    print("Huấn luyện YOLOv8 hoàn tất!")

if __name__ == "__main__":
    train_yolov8(trial=False)
