# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 50
- **Tỉ lệ Pass/Fail:** 5 / 45 (10.0% pass)
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.572
    - Relevancy: 0.544
- **Điểm LLM-Judge trung bình:** 2.583 / 5.0

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Other | 21 | Mix of partial/inferred answers; mismatch between expected phrasing and retrieved context |
| Incomplete | 12 | Thiếu thông tin trong corpus hoặc retriever không trả về đoạn chứa thông tin cần thiết |
| Incorrect | 7 | Trích dẫn sai, hiểu nhầm yêu cầu hoặc nhầm lẫn trong đối chiếu với ground-truth |
| Tone Mismatch | 5 | Văn phong chưa phù hợp với style chuyên nghiệp; cần chuẩn hoá system prompt |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: Điều kiện để làm remote theo chính sách là gì?
1. **Symptom:** Agent trả lời "Không đủ dữ liệu..." thay vì nêu điều kiện cụ thể.
2. **Why 1:** Tài liệu cung cấp không bao gồm điều kiện remote (hoặc không được retriever tìm thấy).
3. **Why 2:** Retriever hiện tại dùng heuristic token-overlap; không phát hiện được câu/đoạn tương ứng.
4. **Why 3:** Không có bước reranking hoặc semantic search để ưu tiên đoạn chứa điều kiện.
5. **Root Cause:** Coverage gap trong corpus kết hợp với retrieval yếu (heuristic token-overlap).

### Case #2: Chính sách làm thêm giờ yêu cầu gì để hợp lệ?
1. **Symptom:** Agent trả lời thiếu chi tiết hoặc nói thiếu dữ liệu trong khi ground-truth yêu cầu phê duyệt bằng văn bản.
2. **Why 1:** Thông tin yêu cầu phê duyệt tồn tại nhưng không được truy xuất.
3. **Why 2:** Chunking/segmenting có thể tách rời câu quan trọng khỏi context.
4. **Why 3:** Reranking/semantic match chưa được sử dụng để lấy câu cụ thể.
5. **Root Cause:** Chunking + retrieval/reranking design không đủ chính xác.

### Case #3: Nếu tôi phát hiện chiếc laptop mới của mình bị hỏng, tôi làm gì?
1. **Symptom:** Agent không cung cấp hướng dẫn xử lý rõ ràng.
2. **Why 1:** Tài liệu hướng dẫn xử lý phần cứng không có trong corpus hoặc nằm trong tài liệu khác.
3. **Why 2:** Retriever không tìm thấy tài liệu liên quan (hoặc trích xuất tóm tắt thiếu chi tiết).
4. **Why 3:** Prompt không yêu cầu agent tra cứu explicit troubleshooting steps.
5. **Root Cause:** Data coverage gap và prompt/retrieval pipeline thiếu bước xác thực nguồn.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] Thay đổi Chunking strategy từ Fixed-size sang Semantic Chunking.
- [ ] Thêm semantic retrieval (embeddings) + reranking để cải thiện recall.
- [ ] Cập nhật System Prompt để nhấn mạnh: "Chỉ trả lời nếu có trong context; nếu không có, báo rõ 'không có trong tài liệu' và gợi ý bước tiếp theo".
- [ ] Mở rộng corpus: bổ sung các tài liệu vận hành (onboarding, hardware procedures, overtime policy).
- [ ] Thêm unit/integration tests cho retriever + reranker + aggregator.
- [ ] Log/trace retrieval candidates in reports để dễ debug và phân tích lỗi truy xuất.
