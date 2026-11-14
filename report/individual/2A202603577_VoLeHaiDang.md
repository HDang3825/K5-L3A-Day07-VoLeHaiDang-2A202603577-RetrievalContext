# Báo cáo Day 7 — Agentic RAG

**Họ và tên:** Võ Lê Hải Đăng  
**MSSV:** 2A202603577  
**Lớp:** K5 - L3A  
**Kho lưu trữ:** `c:\Hdang\VinUni\K5-L3A-Day07-VoLeHaiDang-2A202603577-RetrievalContext`

---

## 1. Dữ liệu và benchmark

- **Nguồn tài liệu & Bản quyền:** Báo cáo thường niên 2025 của Công ty Cổ phần Sữa Việt Nam (Vinamilk) (`VNM_AR_2025`), công bố thông tin đại chúng định kỳ trên HOSE & website Quan hệ cổ đông, sử dụng phi thương mại cho mục đích học tập/nghiên cứu.
- **Số trang & Số chunk:** 110 trang hữu ích (trang 2 đến 111), chia thành **762 chunks** (Chiến lược Strategy B kế thừa trực tiếp từ Day 6).
- **Vector store & Embedding model:**
  - Vector DB: **ChromaDB** (`chroma.sqlite3`, collection `strategy_b`).
  - Embedding Model: **google/embeddinggemma-2** (chiều vector: 768), chế độ suy luận local offline.
- **Agent Model:** **deepseek/deepseek-chat** kết nối qua OpenRouter API.
- **Phân chia Benchmark:**
  - **Tập Dev (17 câu):** 15 câu có đáp án (gold evidence rõ ràng kèm quote, số trang, chunk id) + 2 câu unanswerable (`dev-unans-01`, `dev-unans-02`).
  - **Tập Test (7 câu):** 5 câu có đáp án + 2 câu unanswerable (`test-unans-01`, `test-unans-02`).
  - Tất cả câu unanswerable đều có chủ đề liên quan mật thiết tới ngành sữa/Vinamilk nhưng hoàn toàn không xuất hiện số liệu trong BCTN 2025.
- **Tiêu chí chấm điểm câu trả lời:**
  - **0:** Sai thông tin hoặc thiếu bằng chứng / Từ chối sai câu có đáp án.
  - **1:** Đúng một phần nội dung.
  - **2:** Đúng và đủ nội dung theo gold evidence.
  - **Abstention đúng:** Agent nhận biết và chủ động từ chối các câu hỏi unanswerable thay vì sinh hallucination.

---

## 2. Retrieval và thử nghiệm

Thử nghiệm 3 phương pháp Retrieval khác nhau trên cùng tập dữ liệu 762 chunks:
1. **BM25 (Lexical):** Xây dựng bộ chỉ mục tần suất từ tiếng Việt, tính điểm Robertson-Spärck Jones BM25 ($k_1=1.5, b=0.75$).
2. **Dense Retrieval:** ChromaDB kết hợp vector embedding của `google/embeddinggemma-2`.
3. **Hybrid Retrieval (RRF):** Kết hợp Dense và BM25 thông qua Reciprocal Rank Fusion ($k=60$).

### Bảng kết quả so sánh Retrieval & Context:

| Cấu hình | Phương pháp, top-k, budget | Giả thuyết / Thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian TB (ms) |
|---|---|---|---|---|---|---|---|
| **Baseline** | BM25, top_k=3, budget=1200 | Lexical tìm từ khóa chính xác | dev | 9/15 (60.0%) | 12/15 (80.0%) | 9/15 (60.0%) | 13.0 ms |
| **Phương pháp 2** | Dense, top_k=3, budget=1200 | Embedding bắt ngữ nghĩa tổng thể | dev | 6/15 (40.0%) | 8/15 (53.3%) | 7/15 (46.7%) | 229.5 ms |
| **Phương pháp 3 (Cải thiện 1)** | Hybrid RRF, top_k=3, budget=1200 | Kết hợp điểm mạnh của cả Lexical & Dense | dev | **10/15 (66.7%)** | **12/15 (80.0%)** | **10/15 (66.7%)** | 171.2 ms |
| **Cải thiện 2** | Hybrid RRF, top_k=3, budget=1800, max_search=3 | Cho phép tìm kiếm nhiều lượt (multi-turn) | dev | 10/15 (66.7%) | 12/15 (80.0%) | 10/15 (66.7%) | 215.0 ms |
| **Cấu hình chốt (Hybrid RRF)** | Hybrid RRF, top_k=3, budget=1200 | Áp dụng cấu hình tối ưu lên tập test | test | **4/5 (80.0%)** | **5/5 (100.0%)** | **4/5 (80.0%)** | 175.4 ms |
| **Baseline (Test)** | BM25, top_k=3, budget=1200 | So sánh đối chứng trên test | test | 4/5 (80.0%) | 5/5 (100.0%) | 4/5 (80.0%) | 14.1 ms |

### Phân tích lỗi trên tập Dev & Lý do chọn cấu hình:
1. **Lỗi 1 (`dev-06`):** Với câu hỏi *"Một thành viên độc lập HĐQT của Vinamilk đồng thời đảm nhiệm vị trí gì tại Công ty Cổ phần Xây dựng Coteccons?"*, phương pháp BM25 đơn lẻ bị loãng do từ khóa "Coteccons" xuất hiện thưa thớt, dẫn đến `Hit@1` sai. Trong khi đó, Hybrid RRF và Dense đã kéo được chunk chứa thông tin (trang 18) vào top ranking nhờ ngữ cảnh HĐQT.
2. **Lỗi 2 (`dev-08`):** Câu hỏi về số liệu tài chính *"Lợi nhuận sau thuế năm 2025 theo báo cáo HĐQT"*. BM25 gặp khó khăn khi thuật ngữ tài chính được viết dưới dạng bảng biểu hoặc diễn đạt tương đương; Dense giúp bổ trợ tìm đúng vùng trang báo cáo HĐQT (trang 48).
3. **Lý do chọn cấu hình:** **Hybrid RRF** đạt `Hit@1` cao nhất (66.7%), `Context Evidence Rate` cao nhất (66.7%) và cân bằng tốt giữa thời gian phản hồi (171ms) và chất lượng bằng chứng, vượt trội hơn hẳn so với Dense đơn lẻ.

---

## 3. Agent và đánh giá kết quả

Agent sử dụng cơ chế Tool Calling linh hoạt:
- Action `search`: Sinh câu truy vấn tối ưu và nhận ngữ cảnh trích đoạn.
- Action `answer`: Tổng hợp câu trả lời, trích dẫn rõ nguồn `[chunk_id]` và chủ động từ chối khi phát hiện không có dữ liệu.

### Bảng đánh giá Agent toàn diện (Chạy script `agent_metrics.py`):

| Cấu hình / Split | Tool use | Lượt gọi TB | Grounded answer (0 / 1 / 2) | Điểm TB (/2) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|---|
| **Baseline (Dev)** | 11/17 (64.7%) | 1.0 | 6 / 4 / 5 | 0.93 / 2 | 6/10 (60.0%) | 9/9 (100%) | 2/2 (100%) | 6/15 (40.0%) |
| **Cải thiện 1 - Hybrid (Dev)** | 13/17 (76.5%) | 1.0 | 5 / 4 / 6 | **1.07 / 2** | 6/10 (60.0%) | 10/11 (90.9%) | 2/2 (100%) | 4/15 (26.7%) |
| **Cải thiện 2 - Multi-hop (Dev)** | 13/17 (76.5%) | 1.4 | 5 / 4 / 6 | **1.07 / 2** | 6/12 (50.0%) | 10/10 (100%) | 2/2 (100%) | 5/15 (33.3%) |
| **Baseline (Test)** | 5/7 (71.4%) | 1.0 | 2 / 1 / 2 | 1.00 / 2 | 2/3 (66.7%) | 3/3 (100%) | 2/2 (100%) | 2/5 (40.0%) |
| **Cấu hình chọn - Hybrid (Test)** | 5/7 (71.4%) | 1.0 | **0 / 2 / 3** | **1.60 / 2** | **3/5 (60.0%)** | **5/5 (100%)** | **2/2 (100%)** | **0/5 (0.0%)** |

---

### Minh họa câu hỏi trên tập Test (Chi tiết Trace & Nhãn):

| Query ID | Loại câu | Tool Call & Kết quả | Câu trả lời & Citation | Nhãn, Lý do |
|---|---|---|---|---|
| `test-01` | Có đáp án | `search("chính sách tuyển dụng Vinamilk")` -> Trả về `strategy_b_p102_0718` (Trang 102) | *"Vinamilk áp dụng chính sách tuyển dụng công bằng, minh bạch, đa dạng, bình đẳng giới, không phân biệt đối xử... [strategy_b_p102_0718]"* | **Grade 2**: Đúng và đủ tuyệt đối theo BCTN trang 102. |
| `test-04` | Có đáp án | `search("quyền hạn Chủ tịch HĐQT trường hợp cần thiết")` -> Trả về `strategy_b_p90_0626` (Trang 90) | *"Trong trường hợp cần thiết, Chủ tịch HĐQT có thể triệu tập cuộc họp HĐQT bất thường hoặc lấy ý kiến bằng văn bản... [strategy_b_p90_0626]"* | **Grade 2**: Tìm đúng điều lệ và trích dẫn chuẩn xác. |
| `test-unans-01` | Thiếu bằng chứng | `search("giá bán lẻ đề xuất sữa tươi tiệt trùng Green Farm 1L")` -> Tìm không thấy giá bán lẻ | *"Tài liệu không có đủ bằng chứng để trả lời câu hỏi này."* (Citations: `[]`) | **Abstain Đúng**: BCTN không chứa bảng giá bán lẻ, agent từ chối trả lời chính xác, không hallucinate. |
| `test-unans-02` | Thiếu bằng chứng | `search("Vinamilk mua lại cổ phần TH True Milk")` -> Không có kết quả | *"Tài liệu không có thông tin về việc mua lại cổ phần của TH True Milk."* (Citations: `[]`) | **Abstain Đúng**: Nhận diện thông tin bịa đặt và từ chối trả lời an toàn. |

---

## 4. Kết luận & Giới hạn

- **Đánh đổi & Bài học kinh nghiệm:**
  - Hybrid RRF mang lại độ ổn định cao nhất, loại bỏ hoàn toàn hiện tượng từ chối nhầm trên tập Test (từ chối nhầm giảm từ 40% xuống 0%), nâng điểm Grounded Answer từ **1.0 lên 1.6/2.0**.
  - Multi-search cho phép model tự sửa query khi không tìm thấy dữ kiện ở lượt 1, nhưng tiêu tốn thêm latency và token context.
- **Giới hạn hệ thống:**
  - Một số bảng biểu phân bổ ngân sách phức tạp (như mức thù lao cụ thể từng cá nhân) nếu bị chia cắt giữa 2 chunks có thể khiến agent chỉ trả lời được một phần (Grade 1).
- **Vết dữ liệu (Artifacts):**
  - Toàn bộ kết quả thử nghiệm được lưu đầy đủ tại: `results/dev_baseline/`, `results/dev_imp1_hybrid/`, `results/dev_imp2_multisearch/`, `results/test_baseline/`, `results/test_selected_hybrid/`.
