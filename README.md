# 🚦 SimJam: Smart Traffic Control & Simulation System (SUMO)

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![SUMO Version](https://img.shields.io/badge/SUMO-1.20.0%2B-green.svg)](https://eclipse.dev/sumo/)
[![Computer Vision Ready](https://img.shields.io/badge/Computer--Vision-YOLOv8--Ready-orange.svg)](https://github.com/ultralytics/ultralytics)

Hệ thống tích hợp mô phỏng, phân tích và tối ưu hóa điều khiển đèn tín hiệu giao thông thông minh tại nút giao thông thực tế bằng **SUMO (Simulation of Urban MObility)** thông qua giao diện **TraCI (Traffic Control Interface)**. Dự án hỗ trợ so sánh hiệu năng các thuật toán điều khiển thích ứng nâng cao (Fixed-Time, Actuated Control, Max-Pressure) dưới các kịch bản lưu lượng khác nhau, đồng thời sẵn sàng tích hợp module Thị giác máy tính (Computer Vision) để trích xuất lưu lượng xe trực tiếp.

---

## 📌 Các Tính Năng Nổi Bật

* **Mô phỏng Đồ họa Tương tác (Interactive GUI)**:
  - Cho phép chạy trực quan trên `sumo-gui` với khả năng tùy chỉnh động các tham số: kịch bản lưu lượng (Thấp, Trung bình, Cao), thuật toán điều khiển và tốc độ mô phỏng (delay) ngay từ giao diện dòng lệnh (CLI).
* **So sánh và Đánh giá Tự động (3x3 Grid Benchmark)**:
  - Tự động hóa chạy thử nghiệm chéo 3 thuật toán dưới 3 mức lưu lượng khác nhau (tổng cộng 9 kịch bản thử nghiệm).
  - Tự động tổng hợp dữ liệu, vẽ biểu đồ so sánh trực quan và xuất báo cáo markdown (`outputs/scenario_comparison_report.md`) cùng tệp CSV chi tiết.
* **Các Thuật toán Điều khiển Đèn tín hiệu**:
  - **Fixed-Time (FT)**: Điều khiển chu kỳ cố định mặc định của SUMO (dùng làm mốc đối chứng).
  - **Actuated Control (AC)**: Điều khiển cảm biến thích ứng, tự động điều chỉnh khoảng thời gian đèn xanh dựa trên sự hiện diện của xe tại các làn đường thông qua vòng dò cảm biến (Lane Area Detectors).
  - **Max-Pressure (MP)**: Thuật toán điều khiển động phân bổ pha xanh tối ưu dựa trên chênh lệch áp suất hàng đợi giữa các nhánh vào và nhánh ra của nút giao.
* **Sẵn sàng tích hợp Thị giác Máy tính (Computer Vision Ready)**:
  - Môi trường đã được định cấu hình sẵn sàng cho các mô hình nhận diện đối tượng YOLOv8 và thư viện xử lý luồng Supervision nhằm phát hiện, theo dõi và tính toán lưu lượng giao thông thực tế từ camera.

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
├── sumo_simulation/                     # Module Mô phỏng giao thông SUMO
│   ├── configs/                         # Cấu hình mạng lưới & kịch bản mô phỏng
│   │   ├── osm_cut.net.xml              # Bản đồ mạng lưới nút giao (cắt từ OpenStreetMap)
│   │   ├── osm_cut.netecfg              # Cấu hình biên tập mạng lưới Netedit
│   │   ├── osm_cut_rl.add.xml           # Định nghĩa cảm biến đo hàng đợi (Lane Area Detectors - E2)
│   │   ├── osm_cut_rl.rou.xml           # Luồng phương tiện (Route/Flow) ngẫu nhiên phục vụ thử nghiệm
│   │   └── osm_cut_rl.sumocfg           # Tệp cấu hình tổng thể của kịch bản mô phỏng
│   │
│   ├── outputs/                         # Kết quả đầu ra (Biểu đồ so sánh, file CSV, báo cáo phân tích)
│   │
│   ├── run_comparison_scenarios.py      # Script tự động hóa chạy lưới 9 thử nghiệm (3 flows x 3 algos)
│   └── watch_simulation.py              # Script giao diện CLI tương tác xem mô phỏng trực quan trên GUI
│
├── requirements.txt                     # Danh sách thư viện Python phụ thuộc
└── README.md                            # Hướng dẫn sử dụng dự án (Tài liệu này)
```

---

## 🛠️ Yêu Cầu Hệ Thống & Cài Đặt

### 1. Cài đặt Python & Thư viện
Dự án yêu cầu **Python 3.12 trở lên**. Hãy khởi tạo môi trường ảo và cài đặt các thư viện phụ thuộc:

```bash
# Khởi tạo môi trường ảo (Khuyên dùng)
python -m venv venv
source venv/bin/activate  # Trên Linux/macOS
.\venv\Scripts\activate   # Trên Windows

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

*Các thư viện chính được cài đặt bao gồm: `customtkinter`, `opencv-python`, `numpy`, `pandas`, `ultralytics`, `supervision`, `matplotlib`.*

### 2. Cài đặt SUMO (Simulation of Urban MObility)
1. Tải và cài đặt phiên bản mới nhất tại [Eclipse SUMO Download](https://eclipse.dev/sumo/download/).
2. Thiết lập biến môi trường **`SUMO_HOME`**:
   - **Windows**: Trỏ đường dẫn đến thư mục cài đặt SUMO (ví dụ: `C:\Program Files (x86)\Eclipse\Sumo`).
   - **Linux**: Trỏ đường dẫn đến thư mục cài đặt (ví dụ: `/usr/share/sumo` hoặc `/usr/local/share/sumo`).
3. Đảm bảo thêm thư mục `bin` của SUMO vào biến môi trường `PATH` để hệ thống có thể nhận diện lệnh `sumo` và `sumo-gui` từ Terminal.

---

## 🎮 Hướng Dẫn Sử Dụng

### 1. Xem Mô Phỏng Tương Tác (GUI Mode)
Để khởi chạy mô phỏng trực quan trên giao diện đồ họa, chọn kịch bản lưu lượng và thuật toán điều khiển theo nhu cầu:

```bash
python sumo_simulation/watch_simulation.py
```

* **Quy trình hoạt động**:
  1. Giao diện CLI sẽ yêu cầu bạn lựa chọn mức lưu lượng xe mong muốn (Thấp / Trung bình / Cao).
  2. Chọn thuật toán điều khiển muốn áp dụng (Fixed-Time / Actuated / Max-Pressure).
  3. Nhập độ trễ mô phỏng (delay) tính bằng mili-giây (mặc định là `50 ms`).
  4. Cửa sổ `sumo-gui` sẽ xuất hiện, nhấn biểu tượng **Play** (hoặc phím `Ctrl + A`) để bắt đầu xem mô phỏng.

---

### 2. Chạy Đánh Giá & So Sánh Tự Động (Grid Benchmark)
Nếu bạn muốn chạy thử nghiệm hàng loạt để đánh giá hiệu năng chéo của cả 3 thuật toán dưới mọi điều kiện tải khác nhau:

```bash
python sumo_simulation/run_comparison_scenarios.py
```

* **Kết quả đầu ra**: Sau khi hoàn thành, hệ thống sẽ lưu các tệp phân tích vào thư mục `sumo_simulation/outputs/`:
  - **`scenario_comparison_results.csv`**: Bảng dữ liệu chi tiết chứa các chỉ số hiệu năng của từng bước thử nghiệm.
  - **`scenario_comparison_report.md`**: Báo cáo markdown tổng hợp kết quả chi tiết kèm phân tích kỹ thuật.
  - **Các biểu đồ trực quan hóa** dạng ảnh PNG:
    * `scenario_comparison_avg_queue.png` (Độ dài hàng đợi trung bình)
    * `scenario_comparison_avg_wait.png` (Thời gian chờ trung bình của xe)
    * `scenario_comparison_throughput.png` (Tổng số xe thoát khỏi nút giao)
    * `scenario_comparison_total_delay.png` (Tổng thời gian trễ tích lũy)

---

## 📊 Phân Tích Kỹ Thuật & Chỉ Số Đánh Giá

Hệ thống sử dụng các vòng dò cảm biến **E2 (Lane Area Detectors)** đặt trước nút giao để liên tục theo dõi các chỉ số quan trọng thông qua API TraCI:

1. **Avg Queue Length (Độ dài hàng đợi trung bình - xe)**: Tổng số phương tiện đang dừng/chạy dưới tốc độ tối thiểu ($< 0.1 m/s$) trên làn đường đi vào.
2. **Avg Waiting Time (Thời gian chờ trung bình - s)**: Tổng thời gian dừng của phương tiện tích lũy tính từ lúc đi vào khu vực cảm biến.
3. **Throughput (Lưu lượng thông qua - xe)**: Tổng số lượng xe hoàn thành hành trình đi qua nút giao trong suốt khoảng thời gian mô phỏng.
4. **Total Delay (Tổng thời gian trễ - s)**: Tổng chênh lệch thời gian giữa hành trình thực tế của xe và hành trình lý tưởng (chạy với tốc độ tối đa không bị cản trở).

### Nhận Xét Chung Về Hiệu Năng Thuật Toán:
* **Fixed-Time (FT)**: Thích hợp với luồng xe ổn định, không biến động. Dễ gây lãng phí thời gian xanh nếu một hướng đường không có phương tiện lưu thông.
* **Actuated Control (AC)**: Tốt cho lưu lượng thấp đến trung bình. Tự động kéo dài hoặc kết thúc pha xanh tùy vào sự xuất hiện của xe, giúp giảm thiểu độ trễ đáng kể so với FT.
* **Max-Pressure (MP)**: Đạt hiệu quả cao nhất trong kịch bản lưu lượng cao/quá tải. Bằng cách tính toán chênh lệch số lượng xe giữa các làn vào và làn ra, thuật toán tối đa hóa lưu lượng thông xe, ngăn ngừa ùn tắc cục bộ kéo dài.