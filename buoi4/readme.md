# DEEP LEARNING – Tuần 4
## ANN 
### Sinh viên thực hiện: Trần Như Khả Ý _ 2374802010582
### GVHD : Nguyễn Thái Anh


## 1. Giới thiệu
Lab này xây dựng và huấn luyện một mô hình Artificial Neural Network (ANN) để phân loại dữ liệu 2D (điểm trong vòng tròn và vành đai).
Mục tiêu là minh họa trực quan cách một mạng nơ-ron học quy luật dữ liệu thông qua quá trình forward propagation, loss, và backpropagation.

---

## 2. Công nghệ và thư viện sử dụng
- Python 3.11.5
- PyTorch : Xây dựng kiến trúc ANN, tính toán gradient tự động (autograd), huấn luyện mô hình bằng backpropagation
- numpy : Tạo dữ liệu mẫu (synthetic dataset), xử lý ma trận và vector
- Matplotlib : trực quan hóa dữ liệu 
---

## 3. Cách hoạt động
## 3.1
### 1. Tạo dữ liệu đầu vào:
Chương trình tạo dữ liệu 2D gồm:
- Các điểm nằm trong vòng tròn → Class 0
- Các điểm nằm ngoài vòng tròn (vành đai) → Class 1

### 2. Xây dựng mô hình ANN
Kiến trúc mạng
Mạng gồm 3 lớp:
- Input layer: 2 nút (x, y)
- Hidden layer: 4 nút + ReLU
- Output layer: 1 nút + Sigmoid
- Code chính:
```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(2, 4)  # Đầu vào 2, ẩn 4
        self.relu = nn.ReLU()          # Công tắc ReLU
        self.layer2 = nn.Linear(4, 1)  # Ẩn 4, đầu ra 1
        self.sigmoid = nn.Sigmoid()    # Xác suất 0-1

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x
```

### 3. Lan truyền tiến (Forward Propagation)
Quy trình:
- Nhận input (x, y)
- Tính z = wx + b tại lớp ẩn
- Áp dụng ReLUTính output
- Áp dụng Sigmoid → xác suất lớp
- Code chính:
```python
model = ANN()
```
### 4. Hàm mất mát (Loss Function)
Sử dụng Binary Cross Entropy:
```python
criterion = nn.BCELoss()
```

### 5. Lan truyền ngược (Backpropagation)
PyTorch tự động tính gradient:
```python
loss.backward()
optimizer.step()
```
Quy trình:
- Tính loss
- Tính gradient bằng chain rule
- Cập nhật trọng số\

### 6. Huấn luyện mô hình
Mô hình được train qua nhiều epoch:
- Code chính:
```python
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
```
---

## 3.2 BTVN
### Mục tiêu
Sau khi hoàn thành Lab ANN, bài tập giúp sinh viên:
1. Thay đổi cấu trúc mạng ANN và quan sát ảnh hưởng đến hiệu suất.
2. Thử nghiệm các hàm mất mát (Loss Function) và thuật toán tối ưu (Optimizer).
3. Phân tích kết quả thông qua số liệu và biểu đồ trực quan.

### PHẦN 1: THAY ĐỔI CẤU TRÚC ANN
1. Tăng số nút trong lớp ẩn
Yêu cầu : 
- Sửa hidden layer từ **4 neuron → 8 neuron**
- Huấn luyện lại mô hình với cùng dữ liệu `X_train`, `y_train`
- Số epoch: 100
- Ghi lại:
  - Loss cuối cùng
  - Accuracy trên tập test (`X_test`, `y_test`)

Ví dụ chỉnh sửa:
```python
self.fc1 = nn.Linear(2, 8)
```
- KQ loss:
<img width="899" height="239" alt="image" src="https://github.com/user-attachments/assets/2e194024-4205-4e8f-a8d5-89ce4b1f726c" />
- KQ acc:
<img width="844" height="93" alt="image" src="https://github.com/user-attachments/assets/c70fbfd9-abee-4b24-99f9-ef971b67c9ff" />

  ---

  2. Thêm một hidden layer thứ hai
Kiến trúc mới
- Input: 2 neuron
- Hidden 1: 8 neuron + ReLU
- Hidden 2: 6 neuron + ReLU
- Output: 1 neuron + Sigmoid
- code chính:
```python
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(2, 8)  # Đầu vào 2, ẩn 8
        self.layer2= nn.Linear(8, 6)  
        self.relu = nn.ReLU()          # Công tắc ReLU  
        self.layer3 = nn.Linear(6,1) # ẩn 6 , đầu ra 1
        self.sigmoid = nn.Sigmoid()    # Xác suất 0-1

    def forward(self, x):
        x = self.layer1(x)
        x= self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x
```
- KQ loss:
<img width="811" height="202" alt="image" src="https://github.com/user-attachments/assets/e831aca7-4a05-4c6c-866a-13c46751b26c" />

- KQ ACC :
<img width="1685" height="418" alt="image" src="https://github.com/user-attachments/assets/ff44fba5-a2e3-47dc-85bd-57747fdad4ef" />

---

### PHẦN 2: HÀM MẤT MÁT & OPTIMIZER
1. Thay BCELoss bằng BCEWithLogitsLoss
Yêu cầu
- Thay:
```PYTHON
criterion = nn.BCELoss()
```
bằng:
```python
criterion = nn.BCEWithLogitsLoss()
```
- Bỏ Sigmoid ở output layer.
- Huấn luyện
- Cấu trúc: 2-4-1
- Epochs: 100
- Ghi lại: Loss, Accuracy

- KQ loss:
<img width="981" height="221" alt="image" src="https://github.com/user-attachments/assets/1d58a563-ad2b-46d4-9c87-0a9642584e12" />

- KQ ACC:
<img width="1242" height="387" alt="image" src="https://github.com/user-attachments/assets/e32c125a-7272-4cfc-aea0-6c0c164e372b" />

2.Thay Adam bằng SGD
Yêu cầu
- Giữ kiến trúc (2-4-1)
- Giữ BCELoss
- Thay optimizer:
```PYTHON
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
- Ghi lại
  - Loss cuối cùng
  - Accuracy
- KQ loss và acc:
<img width="1104" height="279" alt="image" src="https://github.com/user-attachments/assets/534c718a-ce17-4d11-b258-6676e11d5e31" />

---

### PHẦN 3: PHÂN TÍCH KẾT QUẢ
- Lưu loss mỗi epoch:
```python
losses.append(loss.item())
```
- Vẽ biểu đồ:
  - TH1 : 2-4-1 + Adam + BCELoss
```python
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), losses, label="2-4-1, Adam, BCELoss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss theo Epoch - 2-4-1 (Adam)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
  - KQ :
<img width="2129" height="1042" alt="image" src="https://github.com/user-attachments/assets/cc4e9a11-b9fb-4fbe-abfb-9e9202a064bd" />

  - TH2 : 2-8-1 + Adam + BCELoss
```python
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), losses_281, label="2-8-1, Adam, BCELoss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss theo Epoch - 2-8-1 (Adam)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
  - KQ :
<img width="1505" height="993" alt="image" src="https://github.com/user-attachments/assets/1803545f-7180-4529-a93e-feddafa9ea06" />

  - TH3 : 2-4-1 + SGD + BCELoss
```PYTHON
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), losses_sgd, label="2-4-1, SGD, BCELoss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss theo Epoch - 2-4-1 (SGD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
  - KQ :
<img width="1338" height="990" alt="image" src="https://github.com/user-attachments/assets/80b9a69f-447c-487c-9006-1b66354dedca" />

---

### 3.3. Kết quả
1. Mô hình học được biên phân loại phi tuyến
ANN có thể phân biệt:
- Điểm trong vòng tròn
- Điểm ngoài vòng tròn
- Điều này chứng minh ANN xử lý tốt dữ liệu không tuyến tính.

2. Loss giảm theo từng epoch
Trong quá trình huấn luyện:
- Loss giảm dần
- Mô hình học được quy luật dữ liệu

3. Trực quan hóa ranh giới phân loại
Sau khi train, mô hình tạo ra:
- Decision boundary cong
- Phân chia rõ 2 lớp dữ liệu
=> ANN đã học thành công cấu trúc vòng tròn
---

## 4. Kết luận:
Lab này minh họa đầy đủ quy trình huấn luyện một mạng nơ-ron nhân tạo:
- Tạo dữ liệu phi tuyến
- Xây dựng mô hình ANN với PyTorch
- Huấn luyện bằng backpropagation
- Đánh giá qua loss và trực quan hóa
- Kết quả cho thấy ANN có khả năng học các quy luật phức tạp mà mô hình tuyến tính không thể giải quyết.

---



