# Đồ Án Thị Giác Máy Tính & Mô Phỏng Giao Thông Thông Minh

Hệ thống tích hợp giám sát giao thông bằng thị giác máy tính (Computer Vision) kết hợp điều khiển tối ưu hóa đèn tín hiệu thông minh sử dụng Học Tăng Cường (Reinforcement Learning) trên nền tảng SUMO.

## 📌 Tổng Quan Dự Án
Dự án được thực hiện bởi sinh viên **Đỗ Trung Kiên** nhằm mục tiêu xây dựng một giải pháp giao thông thông minh toàn diện gồm hai thành phần cốt lõi:
1. **Hệ thống Thị giác máy tính (Computer Vision)**: Tự động nhận diện, theo dõi đa đối tượng (Multi-Object Tracking) và hiệu chỉnh góc nhìn camera để tính toán lưu lượng, quỹ đạo và vận tốc thực tế của các phương tiện giao thông từ video ghi hình.
2. **Hệ thống Mô phỏng Giao thông (SUMO Simulation)**: Thử nghiệm kịch bản lưu thông thực tế và đánh giá hiệu năng các bộ điều khiển đèn giao thông thích ứng (Fixed-Time vs Q-Learning vs Deep Q-Learning) dựa trên hàng đợi và độ trễ trung bình.

---

## 🚀 Tính Năng Chính

### 1. Phân Tích Giao Thông Bằng Thị Giác Máy Tính (src/)
- **Nhận diện & Theo dõi**: Phát hiện đa dạng phương tiện (xe con, xe máy, xe buýt, xe tải, người đi bộ) sử dụng mô hình **YOLOv8** và thư viện **Supervision**.
- **Hiệu chỉnh Góc nhìn (Perspective Calibration)**: Chuyển đổi hệ tọa độ ảnh camera sang tọa độ thực tế dạng mét ($m$) giúp đo lường chính xác tốc độ ($km/h$) và quỹ đạo di chuyển.
- **Giao diện Trực quan (GUI)**: Giao diện hiện đại viết bằng **CustomTkinter** hỗ trợ hiển thị video trực tiếp, vẽ vùng kiểm soát (Region of Interest - ROI), hiển thị thông số tốc độ phương tiện và xuất dữ liệu báo cáo.
- **Xuất dữ liệu**: Xuất các file báo cáo định dạng CSV về lưu lượng dòng xe, quỹ đạo chi tiết và vận tốc để phục vụ phân tích sâu.

### 2. Mô Phỏng & Tối Ưu Hóa Đèn Giao Thông (sumo_simulation/)
- **Tích hợp Bản đồ Thực tế**: Sử dụng bản đồ nút giao từ OpenStreetMap (OSM) cấu hình mạng lưới giao thông SUMO.
- **So sánh 3 Bộ điều khiển Đèn tín hiệu (Traffic Light Controllers)**:
  - **Fixed-Time (Cố định)**: Bộ chu kỳ đèn cố định truyền thống làm mốc so sánh.
  - **Q-Learning RL (Học máy)**: Tự động điều khiển đèn dựa trên học bảng Q-Table để tối thiểu hóa hàng đợi của nút giao thông.
  - **Deep Q-Learning (DQL / DQN)**: Bộ điều khiển đèn thông minh ứng dụng mạng nơ-ron sâu DQN, cho phép xử lý kịch bản giao thông phức tạp và lưu lượng cực lớn.
- **Đánh giá Trực quan**: Tự động thu thập dữ liệu hàng đợi, độ trễ và phần thưởng lũy kế (cumulative reward), sau đó biểu diễn qua các đồ thị so sánh hiệu suất và xuất báo cáo markdown (`heavy_comparison_report.md`).

---

## 📂 Cấu Trúc Thư Mục

```text
├── src/                                  # Mã nguồn module Thị giác máy tính
│   ├── SimJamCVAnalytics.py              # Giao diện GUI chính (CustomTkinter)
│   ├── calib_and_track_ui.py             # Logic giao diện tracking và hiệu chỉnh camera
│   ├── detection_module.py               # Module nhận diện YOLO và tracking
│   ├── analytics_module.py               # Module xử lý quỹ đạo, tốc độ phương tiện
│   ├── cv_intersection_performance_analysis.py  # Phân tích hiệu năng nút giao thông
│   └── dependency_manager.py             # Quản lý tự động tải các gói thư viện
│
├── sumo_simulation/                      # Mã nguồn module Mô phỏng giao thông
│   ├── traci_osm_ft.py                   # Mô phỏng điều khiển Fixed-Time
│   ├── traci_osm_ql.py                   # Thuật toán điều khiển Q-Learning
│   ├── traci_osm_dql.py                  # Thuật toán điều khiển Deep Q-Learning (DQN)
│   ├── run_simulation_analysis.py        # Kịch bản chính chạy so sánh các thuật toán
│   ├── compare_heavy_traffic.py          # Kịch bản mô phỏng mật độ giao thông lớn
│   └── *.xml / *.sumocfg / *.csv / *.png # Bản đồ, cấu hình SUMO và đồ thị kết quả
│
├── weights/                              # Trọng số của mô hình học sâu
│   └── best.pt                           # Trọng số YOLOv8 đã được huấn luyện
│
├── requirements.txt                      # Danh sách các thư viện Python bắt buộc
└── README.md                             # Tài liệu hướng dẫn dự án
```

---

## 🛠️ Yêu Cầu Cài Đặt

### 1. Môi trường Python
* **Python phiên bản 3.12 trở lên**.
* Cài đặt các thư viện Python liên quan:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Cài đặt SUMO (Simulation of Urban MObility)
* Tải và cài đặt phần mềm SUMO từ trang chủ [Eclipse SUMO](https://eclipse.dev/sumo/).
* Đảm bảo biến môi trường **`SUMO_HOME`** đã được cấu hình trỏ tới đường dẫn thư mục cài đặt SUMO trên hệ thống của bạn (Ví dụ trên Windows: `C:\Program Files (x86)\Eclipse\Sumo`).

---

## 🎮 Hướng Dẫn Sử Dụng

### 1. Khởi chạy Giao diện Phân tích Thị giác Máy tính
Khởi chạy giao diện người dùng để thực hiện nhận diện vật thể và đo đạc tốc độ phương tiện trên video:
```bash
python src/SimJamCVAnalytics.py
```
* **Lưu ý**: Lần đầu chạy chương trình sẽ tự động kiểm tra và cài đặt các thư viện thiếu qua giao diện Bootstrap cài đặt nhanh.

### 2. Khởi chạy Mô phỏng và Đánh giá Đèn Giao thông (SUMO)
Chạy kịch bản mô phỏng giao thông để so sánh hiệu năng giữa các bộ điều khiển Fixed-Time, Q-Learning và Deep Q-Learning:
```bash
python sumo_simulation/run_simulation_analysis.py
```
Sau khi chạy xong, chương trình sẽ tự động:
1. Xuất dữ liệu so sánh chi tiết ra các file CSV.
2. Vẽ và lưu các biểu đồ so sánh (`sumo_cv_delay_comparison.png`, `sumo_cv_speed_comparison.png`, v.v.).
3. Tạo báo cáo tóm tắt hiệu suất trực quan ngay trong thư mục mô phỏng.

---

## 📜 Giấy Phép & Bản Quyền
Dự án được phân phối dưới dạng mã nguồn mở phục vụ mục đích nghiên cứu học tập:
* Mô hình YOLOv8 tuân theo giấy phép **AGPL-3.0** từ Ultralytics.
* Giao diện UI sử dụng **CustomTkinter** giấy phép MIT.
