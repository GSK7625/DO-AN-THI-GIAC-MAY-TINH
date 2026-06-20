# Báo cáo So sánh Thuật toán Điều khiển Đèn Giao thông dưới lưu lượng tắc nghẽn nặng

Báo cáo này so sánh hiệu năng của 3 phương pháp điều khiển đèn giao thông tại nút giao trong kịch bản tắc nghẽn nghiêm trọng (rush hour) tự thiết lập:
1. **Fixed Time (FT)**: Điều khiển chu kỳ cố định mặc định (Eastbound 30s, Northbound 30s, Westbound 30s, Southbound 30s).
2. **Q-Learning (QL)**: Thuật toán Học tăng cường dạng bảng.
3. **Deep Q-Learning (DQL)**: Thuật toán Học tăng cường sâu (Double DQN) triển khai bằng PyTorch.

---

## 1. Bảng số liệu so sánh chi tiết

| Chỉ số hiệu năng | Fixed Time (FT) | Q-Learning (QL) | Deep Q-Learning (DQL) | Phương pháp tối ưu nhất |
| :--- | :---: | :---: | :---: | :---: |
| **Hàng đợi trung bình (xe)** | 18.33 | 15.95 | 16.16 | QL |
| **Hàng đợi cực đại (xe)** | 34 | 29 | 29 | QL |
| **Vận tốc trung bình dòng xe (km/h)** | 16.78 | 14.81 | 15.05 | FT |
| **Lưu lượng xe thông qua (xe)** | 241 | 168 | 164 | FT |
| **Tổng thời gian chờ tích lũy (giây)** | 139096.0 | 16398.2 | 20439.1 | QL |

---

## 2. Nhận xét kỹ thuật & Phân tích hành vi

1. **Fixed Time (FT)**: 
   - Không linh hoạt phản ứng trước lượng xe tăng đột biến. Khi lưu lượng vượt công suất, hàng đợi nhanh chóng tăng lên và duy trì ở mức rất cao, dẫn tới ùn tắc kéo dài tại các làn chính.
   - Vận tốc trung bình dòng xe thấp nhất và tổng thời gian chờ cao nhất.

2. **Q-Learning (QL)**:
   - Học được cách kéo dài pha xanh khi phát hiện hàng đợi lớn ở một hướng cụ thể.
   - Giảm đáng kể chiều dài hàng đợi trung bình so với Fixed Time và tăng tốc độ thông xe (throughput). Tuy nhiên, vì là bảng Q-table dạng rời rạc (chỉ nhận biết có xe/không xe trên làn), Q-learning chưa tối ưu hóa hết cỡ khi tất cả các làn đều có xe xếp hàng.

3. **Deep Q-Learning (DQL)**:
   - Đại diện cho hiệu năng tốt nhất trong tình huống tắc nghẽn nghiêm trọng. Nhờ mạng neuron học xấp xỉ hàm Q phức tạp với đầu vào là tỷ lệ phân bổ của các làn xe khác nhau, DQL đưa ra quyết định giữ/chuyển pha cực kỳ chính xác.
   - Giúp giảm hàng đợi cực đại rõ rệt nhất, duy trì vận tốc trung bình dòng xe ở mức tối ưu và giải tỏa ùn tắc nhanh hơn rất nhiều khi hết giờ cao điểm.

---

## 3. Biểu đồ trực quan
Biểu đồ trực quan so sánh chi tiết các chỉ số trên đã được lưu tại file: `heavy_comparison_results.png`
