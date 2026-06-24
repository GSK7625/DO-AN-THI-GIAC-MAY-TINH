# Báo cáo so sánh kịch bản điều khiển giao thông

Báo cáo đánh giá **Fixed-Time (FT)**, **Actuated Control (AC)** và **Max-Pressure (MP)**
trên ba kịch bản lưu lượng:
1. **Lưu lượng thấp** (Scale = 0.5)
2. **Lưu lượng trung bình** (Scale = 1.0)
3. **Lưu lượng cao** (Scale = 1.5)

> **Phương pháp thống kê**: Mỗi thuật toán / kịch bản được chạy 5 lần với seed ngẫu nhiên khác nhau.
> Kết quả hiển thị dưới dạng **giá trị trung bình ± độ lệch chuẩn**.
> Seeds đã dùng: `41748, 54877, 68584, 61269, 99032`

---

## 1. Kết quả chi tiết theo từng Kịch bản (mean ± std)

### Kịch bản 1: Lưu lượng thấp (Scale = 0.5)
| Thuật toán | Avg Queue Length (xe) | Avg Waiting Time (s) | Throughput (xe) | Total Delay (s) | Avg Delay/Vehicle (s) | LOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FT | 11.81 ± 0.06 | 31.73 ± 0.14 s | 263 ± 0 | 11130.5 ± 60.1 s | 38.65 ± 0.21 s | **D** |
| AC | 4.15 ± 0.13 | 8.16 ± 0.38 s | 270 ± 1 | 3460.6 ± 141.4 s | 12.02 ± 0.49 s | **B** |
| MP | 3.93 ± 0.15 | 7.45 ± 0.42 s | 272 ± 1 | 3237.0 ± 152.8 s | 11.24 ± 0.53 s | **B** |

### Kịch bản 2: Lưu lượng trung bình (Scale = 1.0)
| Thuật toán | Avg Queue Length (xe) | Avg Waiting Time (s) | Throughput (xe) | Total Delay (s) | Avg Delay/Vehicle (s) | LOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FT | 22.60 ± 0.11 | 31.31 ± 0.16 s | 520 ± 1 | 22607.4 ± 134.0 s | 40.08 ± 0.24 s | **D** |
| AC | 9.32 ± 0.08 | 9.16 ± 0.14 s | 536 ± 1 | 8033.5 ± 88.9 s | 14.24 ± 0.16 s | **B** |
| MP | 8.20 ± 0.29 | 7.41 ± 0.36 s | 538 ± 1 | 6894.4 ± 311.9 s | 12.22 ± 0.55 s | **B** |

### Kịch bản 3: Lưu lượng cao (Scale = 1.5)
| Thuật toán | Avg Queue Length (xe) | Avg Waiting Time (s) | Throughput (xe) | Total Delay (s) | Avg Delay/Vehicle (s) | LOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FT | 31.57 ± 0.13 | 33.32 ± 0.14 s | 775 ± 1 | 37055.1 ± 200.0 s | 44.11 ± 0.24 s | **D** |
| AC | 19.00 ± 0.22 | 13.75 ± 0.22 s | 798 ± 1 | 17441.5 ± 250.9 s | 20.76 ± 0.30 s | **C** |
| MP | 15.03 ± 0.68 | 9.05 ± 0.55 s | 799 ± 2 | 13233.2 ± 688.4 s | 15.75 ± 0.82 s | **B** |

---

## 2. Biểu đồ trực quan hóa hiệu năng

### 2.1 Độ dài hàng đợi trung bình
![Average Queue Length](scenario_comparison_avg_queue.png)

### 2.2 Thời gian chờ trung bình
![Average Waiting Time](scenario_comparison_avg_wait.png)

### 2.3 Tổng xe thông qua
![Throughput](scenario_comparison_throughput.png)

### 2.4 Tổng thời gian trễ
![Total Delay](scenario_comparison_total_delay.png)

---

## 3. Đánh giá và Phân tích kỹ thuật

1. **Độ ổn định thống kê**: Mỗi cặp (kịch bản, thuật toán) được đánh giá qua 5 lần chạy độc lập
   với seed ngẫu nhiên khác nhau mỗi phiên, đảm bảo kết quả khách quan.

2. **Lưu lượng thấp (Scale 0.5)**: AC và MP không lãng phí pha xanh cho hướng trống,
   giảm đáng kể hàng đợi và thời gian chờ so với FT.

3. **Lưu lượng trung bình (Scale 1.0)**: AC tối ưu theo hiện diện thực tế của xe.
   MP bắt đầu thể hiện ưu thế phân bổ đều áp lực hàng đợi.

4. **Lưu lượng cao (Scale 1.5)**: MP vượt trội nhờ trực tiếp giải tỏa hướng có
   hàng đợi lớn nhất, ngăn tắc nghẽ cục bộ kéo dài.
