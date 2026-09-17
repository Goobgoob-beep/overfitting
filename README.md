# Dự đoán giá nhà - Overfitting và cách khắc phục

## Nội dung script

1. **Tạo overfitting**: sinh đặc trưng đa thức (Polynomial Features) bậc 3 trên tập dữ liệu nhỏ, khiến số đặc trưng vượt xa số mẫu train => mô hình Linear Regression học thuộc lòng tập train (R2 train = 1.0000) nhưng dự đoán kém trên tập test.
2. **Cách 1**: so sánh sai số (R2, MAE) trên tập train và test khi tăng dần bậc đa thức, cho thấy xu hướng overfitting rõ rệt.
3. **Cách 2**: tách thêm tập validation từ tập train để chọn bậc đa thức phù hợp mà không dùng đến tập test.
4. **Cách 3**: dùng K-fold cross-validation (5-fold) để đánh giá ổn định hơn so với một lần chia validation duy nhất.
5. **Cách 4**: áp dụng Lasso regularization (với `LassoCV`) trên mô hình đã bị overfit (bậc 3), alpha được chọn bằng cross-validation trên tập train, không dùng tập test để chọn tham số. Kết quả cho thấy R2 trên test được cải thiện rõ rệt và nhiều đặc trưng không quan trọng bị triệt tiêu về 0.

## Dữ liệu

File `spreadsheet.xlsx` chứa dữ liệu giá nhà với các đặc trưng đầu vào và cột mục tiêu `gia` (giá).

## Chạy trên máy (local)

```
pip install pandas scikit-learn openpyxl
python overfitting.py
```

## Chạy trên GitHub Actions

1. Fork repo này.
2. Vào tab **Actions** > **Run House Price Script** > **Run workflow**.
3. Sau khi chạy xong, tải kết quả trong phần **Artifacts** của lần chạy đó.
