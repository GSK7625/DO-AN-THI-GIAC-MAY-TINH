# SimJam: Smart Traffic Control & Simulation System (SUMO)

Hệ thống mô phỏng, phân tích và tối ưu hóa điều khiển đèn tín hiệu giao thông tại nút giao bằng phần mềm SUMO (Simulation of Urban MObility) thông qua giao diện TraCI (Traffic Control Interface). Dự án hỗ trợ so sánh hiệu năng các thuật toán điều khiển dưới các kịch bản lưu lượng khác nhau, tích hợp dữ liệu thu được từ module thị giác máy tính (Computer Vision) để tái hiện luồng giao thông thực tế.

## Các thuật toán điều khiển

- **Fixed-Time (FT)**: Điều khiển theo chu kỳ thời gian cố định.
- **Actuated Control (AC)**: Điều khiển cảm biến thích ứng, tự động điều chỉnh thời gian đèn xanh dựa trên sự hiện diện của xe thông qua vòng dò cảm biến (Lane Area Detectors - E2).
- **Max-Pressure (MP)**: Thuật toán điều khiển động tối ưu pha dựa trên chênh lệch áp suất hàng đợi (chênh lệch số lượng xe giữa các nhánh vào và nhánh ra).

## Cấu trúc thư mục chính

- `sumo_simulation/`: Thư mục chính chứa mã nguồn mô phỏng.
  - `configs/`: Chứa bản đồ mạng lưới (.net.xml) và cấu hình mô phỏng (.sumocfg, .rou.xml, .add.xml).
  - `core/`: Thư viện xử lý mô phỏng (`simulator.py`), thuật toán điều khiển (`controllers.py`) và xuất báo cáo (`reporting.py`).
  - `outputs/`: Nơi lưu trữ biểu đồ so sánh dạng PNG và các file dữ liệu CSV sau khi chạy đánh giá.
  - `generate_real_routes.py`: Sinh cấu hình luồng xe thực tế cho SUMO từ dữ liệu camera.
  - `watch_simulation.py`: Script CLI tương tác để chạy và theo dõi trực quan mô phỏng (GUI).
  - `run_comparison_scenarios.py`: Chạy thử nghiệm hàng loạt trên các kịch bản lưu lượng nhân tạo (Thấp, Trung bình, Cao).
  - `run_real_comparison.py`: Chạy thử nghiệm và so sánh thuật toán trên kịch bản lưu lượng thực tế.
- `input/`: Chứa dữ liệu tracking thực tế đầu vào (ví dụ: `tmc.csv`, `vehicle_tracks_xy.csv`).
- `trainyolosm.ipynb`: Notebook huấn luyện mô hình YOLOv8 phục vụ nhận diện phương tiện.
- `requirements.txt`: Danh sách các thư viện Python phụ thuộc.

## Cài đặt

### 1. Môi trường Python
Yêu cầu Python 3.12 trở lên. Khởi tạo môi trường ảo và cài đặt thư viện:

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
.\venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Cấu hình SUMO
1. Cài đặt phiên bản Eclipse SUMO.
2. Cấu hình biến môi trường `SUMO_HOME` trỏ tới thư mục cài đặt SUMO.
3. Thêm thư mục `bin` của thư mục cài đặt SUMO vào biến môi trường `PATH` để có thể thực thi lệnh `sumo` và `sumo-gui`.

## Hướng dẫn sử dụng

### 1. Xem mô phỏng trực quan (Interactive GUI)
Chạy script tương tác dưới dạng CLI để lựa chọn kịch bản và thuật toán điều khiển:

```bash
python sumo_simulation/watch_simulation.py
```

### 2. Đánh giá tự động trên các kịch bản lưu lượng nhân tạo
Chạy toàn bộ 9 kịch bản kiểm thử (3 mức lưu lượng x 3 thuật toán) và tự động xuất biểu đồ so sánh:

```bash
python sumo_simulation/run_comparison_scenarios.py
```

### 3. Đánh giá tự động trên lưu lượng thực tế
So sánh hiệu năng của 3 thuật toán trên dữ liệu lưu lượng thực tế trích xuất từ camera:

```bash
python sumo_simulation/run_real_comparison.py
```

### 4. Tạo luồng xe từ dữ liệu camera
Tạo file cấu hình luồng giao thông cho SUMO dựa trên dữ liệu tracking đầu vào trong thư mục `input`:

```bash
python sumo_simulation/generate_real_routes.py
```

## Chỉ số đánh giá hiệu năng

- **Avg Queue (Hàng đợi trung bình - xe)**: Số lượng phương tiện dừng hoặc đi chậm trên các làn đường đi vào nút giao.
- **Avg Wait (Thời gian chờ trung bình - giây)**: Tổng thời gian dừng chờ của phương tiện tại nút giao.
- **Throughput (Lưu lượng thông xe - xe)**: Tổng số xe hoàn thành hành trình qua nút giao thành công.
- **Total Delay (Tổng thời gian trễ - giây)**: Sự chênh lệch giữa thời gian di chuyển thực tế và thời gian di chuyển lý tưởng.