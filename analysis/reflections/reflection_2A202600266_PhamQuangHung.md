**Engineering Contribution:**
- Đóng góp gần như toàn bộ từ data đến AI và benchmark. Xem link GitHub nhóm.

**Technical Depth**
- MRR: Trung bình của nghịch đảo vị trí của kết quả truy vấn, ưu tiên việc đúng xuất hiện ở top
- Cohen's Kappa: Thước đo mức đồng thuận giữa hai người, loại trừ đồng thuận do yếu tố ngẫu nhiên
- Position Bias: Xu hướng chú ý vào kết quả có vị trí cao hơn, không quan tâm đến việc kết quả đó có thực sự đúng hay không
- Trade-off giữa Chi phí và Chất lượng
    - Bản chất: Nâng chất lượng (thêm người đánh giá, nhiều judge, nhãn tay, mô hình lớn hơn) tăng chi phí (tiền, thời gian, token/latency); giảm chi phí thường làm giảm độ tin cậy/độ chính xác.
    - Yếu tố cần cân nhắc: số lượng nhãn, số judge độc lập, chất lượng từng judge (human vs auto), tần suất regression, latency và chi phí API/token.
    - Chiến lược tối ưu hóa: stratified sampling (chấm kỹ trên hard cases), active learning, dùng ensemble judge rẻ + calibration bằng small gold set, progressive gating (phân cấp: rẻ→đắt chỉ khi cần), và đo điểm bù (diminishing returns) để tìm điểm cắt chi phí/hiệu quả.
    - Quy tắc thực tế: đầu tư vào chất lượng trên những trường hợp có ảnh hưởng lớn (edge/failure modes); dùng tự động cho scale nhưng giữ gold/human checks để sửa bias và đo độ tin cậy.

**Problem Solving:**
- *Vấn đề:* Việc xem và đếm từng dòng trong cái file benchmark_results.json rất tốn thời gian và dễ sai
- *Giải pháp*: Xây dựng một Python script parse_report giúp xử lý ra các tham số mà mình cần