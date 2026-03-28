## Thực hành: Dự đoán Chuỗi Thời Gian với Mạng Nơ-ron Hồi quy (RNN)
#### SVTH : Trần Như Khả Ý 
#### GVHD : Nguyễn Thái Anh

---

## Giới thiệu 
Dự án này là một bài thực hành toàn diện về việc ứng dụng Mạng Nơ-ron Hồi quy (Recurrent Neural Network - RNN) để giải quyết bài toán dự đoán chuỗi thời gian. Bằng việc sử dụng thư viện PyTorch, mô hình được huấn luyện để nhận diện và học hỏi quy luật dao động của một tập dữ liệu giả lập (hàm sóng Sin có nhiễu), từ đó đưa ra các dự đoán cho tương lai.

---

## Công nghệ sử dụng
- **Ngôn ngữ lập trình:** Python 3.x
- **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`) - Dùng để thiết kế kiến trúc mạng RNN, tính toán gradient và tối ưu hóa mô hình.
- **Xử lý tính toán khoa học:** NumPy - Dùng để tạo lập dữ liệu giả lập, xử lý ma trận và biến đổi mảng (array).
- **Trực quan hóa dữ liệu:** Matplotlib (`matplotlib.pyplot`) - Dùng để vẽ biểu đồ theo dõi quá trình huấn luyện (Loss) và so sánh kết quả dự đoán.
- **Môi trường phát triển:** Jupyter Notebook / Google Colab.

---

## Cấu trúc bài LAB 
### Phần 1: Chuẩn bị dữ liệu
- Tạo lập dữ liệu giả lập dựa trên hàm sóng Sin và thêm nhiễu (noise) ngẫu nhiên để mô phỏng dữ liệu thực tế.
- Thực hiện chuẩn hóa dữ liệu (Min-Max Scaling) về khoảng `[0, 1]`.
- Ứng dụng kỹ thuật cửa sổ trượt (Sliding Window) để chia chuỗi dữ liệu dài thành các mẫu (sequences) ngắn.
- Phân chia dữ liệu thành 3 tập: Huấn luyện (Train - 70%), Xác thực (Validation - 15%), và Kiểm tra (Test - 15%).
- Code chính:
```python
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]
```
<img width="1635" height="912" alt="image" src="https://github.com/user-attachments/assets/824796c5-95b5-4616-9843-61ea662a8d3b" />

### Phần 2: Xây dựng và Huấn luyện mô hình cơ bản
- Thiết kế kiến trúc mạng RNN cơ bản sử dụng `torch.nn.RNN` kết hợp với một lớp Linear (`torch.nn.Linear`) để xuất kết quả.
- Thiết lập hàm mất mát `MSELoss` và thuật toán tối ưu `Adam` với learning rate khởi tạo là 0.01.
- Tiến hành huấn luyện mô hình qua 150 vòng lặp (epochs) và trực quan hóa sự suy giảm của hàm mất mát (Train Loss & Validation Loss).
- Code chính:
```python
 def __init__(self, input_size=3, hidden_size=32, output_size=1): 
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
----
# 3. Định nghĩa hàm mất mát và tối ưu hóa
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```
<img width="2394" height="1138" alt="image" src="https://github.com/user-attachments/assets/a355f955-1422-4c6e-a387-6d812f6ac967" />


### Phần 3: Đánh giá mô hình
- Sử dụng mô hình đã huấn luyện để dự đoán trên tập dữ liệu chưa từng thấy (Test set).
- Đo lường hiệu suất thực tế bằng các chỉ số sai số chuẩn: **MSE** (Mean Squared Error) và **MAE** (Mean Absolute Error).
- Vẽ biểu đồ so sánh trực quan giữa đường giá trị thực tế (Actual) và đường giá trị dự đoán (Predicted).
- Code chính:
```python
mse_criterion = nn.MSELoss()
mae_criterion = nn.L1Loss() # L1 Loss chính là Mean Absolute Error (MAE)

# Tính toán giá trị (dùng .item() để lấy số thực từ tensor)
mse_score = mse_criterion(predictions, y_test).item()
mae_score = mae_criterion(predictions, y_test).item()

print("--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ---")
print(f"1. MSE (Mean Squared Error)  : {mse_score:.6f}")
print(f"2. MAE (Mean Absolute Error) : {mae_score:.6f}")
print("-" * 38)

# Vẽ đường giá trị thực tế
plt.plot(y_test_plot, label="Giá trị thực tế (Actual)", color='blue', marker='o', markersize=4)
# Vẽ đường giá trị dự đoán
plt.plot(predictions_plot, label="Giá trị dự đoán (Predicted)", color='red', linestyle='--', marker='x', markersize=4)

```
<img width="2560" height="1379" alt="image" src="https://github.com/user-attachments/assets/0bd79032-7e9a-4d35-8f88-7297f407a89a" />


### Phần 4: Các yêu cầu nâng cao (Hyperparameter Tuning)
Thực hiện các thử nghiệm chuyên sâu để đánh giá sức mạnh và giới hạn của RNN thông qua việc thay đổi các siêu tham số:
- **Độ dài chuỗi (`seq_length`):** Thử nghiệm với các giá trị 10, 20, 30.
- **Kích thước lớp ẩn (`hidden_size`):** Thử nghiệm với 16, 32, 64 nơ-ron.
- **Learning Rate & Epochs:** Đánh giá tốc độ học (0.001 vs 0.1) và tăng số vòng lặp lên 300.
- **Kiến trúc mạng:** Thử nghiệm tăng số tầng ẩn (2 layers) và áp dụng Dropout (0.2).
- **Dự đoán đa bước (Multi-step):** Thử thách mô hình dự đoán 3 bước tiếp theo thay vì chỉ 1 bước.
- Code chính:
```python
# 1. Thử giá trị seq_length
mse_seq10, _, _, _ = experiment_rnn(seq_length=10)
mse_seq30, _, _, _ = experiment_rnn(seq_length=30)
print(f"[2] Thay đổi Seq_length = 10  -> MSE: {mse_seq10:.4f}")
print(f"    Thay đổi Seq_length = 30  -> MSE: {mse_seq30:.4f}")

# 2. Thử giá trị hidden_size
mse_h16, _, _, _ = experiment_rnn(hidden_size=16)
mse_h64, _, _, _ = experiment_rnn(hidden_size=64)
print(f"[3] Thay đổi Hidden_size = 16 -> MSE: {mse_h16:.4f}")
print(f"    Thay đổi Hidden_size = 64 -> MSE: {mse_h64:.4f}")

# 3. Tăng Epochs
mse_ep, _, _, _ = experiment_rnn(num_epochs=300)
print(f"[4] Tăng Epochs lên 300       -> MSE: {mse_ep:.4f}")

# 4. Thay đổi Learning Rate
mse_lr_low, _, _, _ = experiment_rnn(lr=0.001)
mse_lr_high, _, _, _ = experiment_rnn(lr=0.1)
print(f"[5] Learning Rate = 0.001     -> MSE: {mse_lr_low:.4f}")
print(f"    Learning Rate = 0.1       -> MSE: {mse_lr_high:.4f}")

# 5. Tăng số tầng (layers) và dùng Dropout
mse_drop, _, _, _ = experiment_rnn(num_layers=2, dropout=0.2)
print(f"[6] 2 Layers + Dropout(0.2)   -> MSE: {mse_drop:.4f}")

# 6. Dự đoán nhiều bước (Multi-step: 3 bước)
mse_multi, _, _, _ = experiment_rnn(output_steps=3)
print(f"[7] Dự đoán 3 bước tiếp theo  -> MSE: {mse_multi:.4f}\n")

# VẼ BIỂU ĐỒ SAI SỐ THEO TỪNG THỜI ĐIỂM (TẬP TEST)
# Tính sai số (Absolute Error) ở mỗi thời điểm cho mô hình Baseline
errors = np.abs(y_test_base.flatten() - preds_base.flatten())
```

<img width="1470" height="947" alt="image" src="https://github.com/user-attachments/assets/f378caa0-1dcf-4393-95cc-0f61068e00fe" />




