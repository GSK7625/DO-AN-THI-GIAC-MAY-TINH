# Báo cáo so sánh kịch bản điều khiển giao thông
    
Báo cáo này đánh giá hiệu năng của ba thuật toán điều khiển đèn tín hiệu giao thông: **Fixed-Time (FT - Chu kỳ cố định)**, **Actuated Control (AC - Cảm biến lưu lượng)** và **Max-Pressure (MP - Tối đa hóa áp lực)** dưới ba kịch bản lưu lượng khác nhau:
1. **Lưu lượng thấp** (Scale = 0.5)
2. **Lưu lượng trung bình** (Scale = 1.0)
3. **Lưu lượng cao** (Scale = 1.5)

---

## 1. Kết quả chi tiết theo từng Kịch bản

### Kịch bản 1: Lưu lượng thấp (Scale = 0.5)
| Thuật toán | Avg Queue Length (xe) | Avg Waiting Time (s) | Throughput (xe) | Total Delay (s) | Avg Delay/Vehicle (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| FT | 11.81 | 31.72 s | 262 | 11126.6 s | 38.63 s |
| AC | 3.66 | 6.77 s | 271 | 2947.4 s | 10.23 s |
| MP | 3.82 | 7.20 s | 271 | 3115.9 s | 10.82 s |

### Kịch bản 2: Lưu lượng trung bình (Scale = 1.0)
| Thuật toán | Avg Queue Length (xe) | Avg Waiting Time (s) | Throughput (xe) | Total Delay (s) | Avg Delay/Vehicle (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| FT | 22.72 | 31.55 s | 521 | 22829.0 s | 40.48 s |
| AC | 8.92 | 8.49 s | 535 | 7622.1 s | 13.51 s |
| MP | 8.03 | 7.30 s | 539 | 6700.2 s | 11.88 s |

### Kịch bản 3: Lưu lượng cao (Scale = 1.5)
| Thuật toán | Avg Queue Length (xe) | Avg Waiting Time (s) | Throughput (xe) | Total Delay (s) | Avg Delay/Vehicle (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| FT | 31.45 | 33.19 s | 776 | 36859.6 s | 43.88 s |
| AC | 16.49 | 11.11 s | 798 | 14748.3 s | 17.56 s |
| MP | 13.45 | 7.69 s | 802 | 11599.5 s | 13.81 s |

---

## 2. Biểu đồ trực quan hóa hiệu năng

### 2.1 Độ dài hàng đợi trung bình (Average Queue Length)
![Average Queue Length](scenario_comparison_avg_queue.png)

### 2.2 Thời gian chờ trung bình (Average Waiting Time)
![Average Waiting Time](scenario_comparison_avg_wait.png)

### 2.3 Tổng xe thông qua (Throughput)
![Throughput](scenario_comparison_throughput.png)

### 2.4 Tổng thời gian chậm/trễ (Total Delay)
![Total Delay](scenario_comparison_total_delay.png)

---

## 3. Đánh giá và Phân tích kỹ thuật

1. **Kịch bản Lưu lượng thấp (Scale 0.5)**:
   - Các thuật toán thông minh (**AC**, **MP**) phản ứng linh hoạt giúp giảm đáng kể hàng đợi và thời gian chờ so với **FT** truyền thống do không lãng phí thời gian xanh cho các hướng không có xe.

2. **Kịch bản Lưu lượng trung bình (Scale 1.0)**:
   - **Actuated Control (AC)** hoạt động rất tốt nhờ tối ưu thời gian xanh theo sự hiện diện thực tế của phương tiện.
   - **Max-Pressure (MP)** bắt đầu thể hiện ưu thế phân bổ đều áp lực hàng đợi giữa các nhánh.

3. **Kịch bản Lưu lượng cao (Scale 1.5)**:
   - Khi lưu lượng tăng cao, **Max-Pressure (MP)** vượt trội hơn hẳn **AC** và **FT** vì nó trực tiếp giải tỏa các nhánh có độ dài hàng đợi lớn nhất, ngăn chặn tình trạng tắc nghẽn cục bộ kéo dài và đạt Throughput cao nhất cùng với Total Delay thấp nhất.
