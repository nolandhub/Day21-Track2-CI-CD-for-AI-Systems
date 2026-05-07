# BÁO CÁO THỰC HÀNH LAB MLOPS (DAY 21)

## 1. Phân tích và lựa chọn Siêu tham số (Hyperparameters)
*   **Bộ tham số tốt nhất**: `n_estimators: 200`, `max_depth: 20`, `min_samples_split: 2`.
*   **Lý do lựa chọn**: Thông qua các thí nghiệm trên MLflow ở Bước 1, bộ tham số này cho kết quả cân bằng nhất. Khi thử nghiệm với số lượng cây lớn hơn (1000+) hoặc độ sâu quá lớn, mô hình có dấu hiệu bão hòa và không cải thiện đáng kể Accuracy trên tập `eval.csv` (đạt ~0.6840). Bộ tham số này đảm bảo tốc độ huấn luyện nhanh mà vẫn giữ được độ chính xác tối ưu cho mô hình Random Forest trên tập dữ liệu gốc.

## 2. Các khó khăn gặp phải và giải pháp

### 2.1. Vấn đề vượt ngưỡng Accuracy 0.70 (Eval Gate)
*   **Khó khăn**: Với dữ liệu huấn luyện ban đầu (2998 mẫu), mô hình liên tục thất bại ở mức Accuracy 0.68, khiến Pipeline CI/CD tự động chặn bước Deploy.
*   **Giải pháp**: 
    - Áp dụng **Kỹ thuật đặc trưng (Feature Engineering)**: Tạo thêm các biến tương tác có ý nghĩa hóa học như `total_acidity`, `sugar_to_alcohol`, `alcohol_density` và `free_sulfur_ratio`.
    - **Tăng quy mô dữ liệu**: Gộp toàn bộ dữ liệu từ Phase 2 vào huấn luyện (tổng cộng 5996 mẫu).
*   **Kết quả**: Accuracy tăng vọt lên **0.7560**, giúp mô hình vượt qua cổng kiểm soát Eval Gate thành công.

### 2.2. Lỗi triển khai (Deploy Failed) do Health Check
*   **Khó khăn**: Bước `Deploy` trong GitHub Actions báo lỗi "Connection refused" dù code không sai. Qua kiểm tra Log trên VM bằng `journalctl`, phát hiện server mất khoảng 15 giây để tải model từ GCS, vượt quá thời gian `sleep 10` mặc định.
*   **Giải pháp**: Điều chỉnh file `mlops.yml`, tăng thời gian chờ sau khi restart service lên **30 giây**. Việc này đảm bảo server đã sẵn sàng hoàn toàn trước khi Pipeline thực hiện lệnh `curl` kiểm tra sức khỏe.

### 2.3. Lỗi Unit Test do không đồng nhất tên cột
*   **Khó khăn**: Sau khi thêm Feature Engineering, Unit Test bị lỗi `KeyError` do dữ liệu giả lập sử dụng tên cột có dấu gạch dưới (`fixed_acidity`), trong khi dữ liệu thật sử dụng khoảng trắng (`fixed acidity`).
*   **Giải pháp**: Cập nhật file `tests/test_train.py` để sử dụng đúng định dạng tên cột của tập dữ liệu UCI Wine Quality, đảm bảo tính nhất quán cho toàn bộ hệ thống.

---
**Kết luận**: Hệ thống đã hoạt động ổn định, tự động hóa hoàn toàn quy trình từ khi commit code/dữ liệu cho đến khi mô hình được cập nhật trên Cloud VM.
