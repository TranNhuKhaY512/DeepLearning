## Thực hành môn Giới thiệu học sâu - Lab 1
### Sinh viên thực hiện:  Trần Như Khả Ý
### MSSV : 2374802010582
### GVHD: Nguyễn Thái Anh
---
### Lab 1 : Pytorch cơ bản & quy hồi tuyến tính
- Trong lab này, chủ yếu sử dụng thư viện Pytorch tập trung vào xử lý tensor, tính đạo hàm và triển khai thuật toán Gradient Descent để giải quyết bài toán tuyến tính.
---
### Công nghệ sử dụng
- Ngôn ngữ: Python 3.11.5
- Thư viện chính:
  - torch (PyTorch): Thư viện chính để xây dựng mô hình và tính toán trên Tensor.
  - pandas: Đọc và xử lý dataset (Iris.csv).
  - numpy: Xử lý mảng số học.
  - sklearn (scikit-learn): Tiền xử lý dữ liệu (LabelEncoder) và chia tập dữ liệu (train_test_split).
  - matplotlib: dùng để trực quan hóa dữ liệu.

---
Trong lab này gồm các phần như sau:
### 1. Khởi tạo và xử lý dữ liệu :
- Kiểm tra GPU/CUDA có hoạt động hay không
- Code chính:
  ```python
  torch.cuda.is_available()
  ```
- Đọc dữ liệu từ file Iris.csv
- KQ:
<img width="1313" height="387" alt="image" src="https://github.com/user-attachments/assets/524e7589-9b72-49b5-8ec1-e5a88b2d6316" />

- Mã hóa nhãn "Species" sang dạng số.
- Chia dữ liệu thành tập Train/Test (tỷ lệ 80/20)
- Chuyển đổi dữ liệu từ numpy array sang tensor (FloatTensor cho đầu vào và LongTensor cho nhãn).
- Code chính
```python
le = LabelEncoder()
X = df.drop(["Species"], axis = 1).values
y= le.fit_transform(df["Species"].values) 
# chia dữ liệu với test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=42 )
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train).reshape(-1,1)
y_test = torch.LongTensor(y_test).reshape(-1,1)
```
---
### 2. Tính đạo hàm bằng pytorch
Minh họa khả năng tự động gradient của pytorch thông qua thuộc tính `requires_grad=True`
- Tính đạo hàm của đa thức `y= 2x^4 + x^3 + 3x^2 + 5x + 1`
- Sử dụng `.backward()` để lan truyền ngược và tính `x.grad`.
- KQ :
<img width="1454" height="82" alt="image" src="https://github.com/user-attachments/assets/816eeb75-0a4b-475b-af5f-0722d4587b5d" />
---

### 3. Các BTVN:
#### 1. BTVN 1: cho y = 5x^5 + 6x^3 - 3x + 1. cho biết độ dốc của đa thức trên ở điểm nào.
-  Tính đạo hàm của đa thức `y = 5x^5 + 6x^3 - 3x + 1`
-  Sử dụng `.backward()` để lan truyền ngược và tính `x.grad`.
-  Code chính:
```python
x = torch.tensor(2.0, requires_grad=True)
y = 5*x**5 + 6*x**3 - 3*x + 1
y.grad_fn
y.backward()
x.grad
```
-  KQ :
<img width="1928" height="419" alt="image" src="https://github.com/user-attachments/assets/a06901ed-db64-430e-bd64-35e7fd13f024" />
---
#### 2. BTVN2: tạo 1 tensor ban đầu có giá trị là 2. định nghĩa hàm số và tính gradient y = x^3 + 2x^2 + 5x + 1. hãy tính dy/dx tại giá trị của x dùng phương pháp gradient descent với: learning_rate alpha =0.1 để cập nhật giá trị x trong 10 vòng lặp.
- Tính đạo hàm của đa thức `y = x^3 + 2x^2 + 5x + 1` tại x = 2
- Thực hiện thuật toán Gradient Descent với lr_rate_alpha = 0.1 qua 10 vòng lặp
- Code chính
```python
x = torch.tensor(2.0, requires_grad=True)
learning_rate_alpha = 0.1
for i in range(10):
    y = x**3 + 2*x**2 + 5*x + 1 
    y.backward()
    with torch.no_grad():
        x -= learning_rate_alpha * x.grad  #cập nhật x theo công thức gradient descent
```
- KQ:
<img width="1374" height="457" alt="image" src="https://github.com/user-attachments/assets/5022dd6d-9ac1-4c5b-a4ea-656aee818ef0" />

---
#### 3. BTVN 3: tạo 1 tập dữ liệu giả lập với x là số giờ học ngẫu nhiên từ 1 đến 10 và y là số điểm đc tính theo công thức y = 3x + 5+ noise , với noise là 1 giá trị ngẫu nhiên nhỏ. 
- Tạo dữ liệu giả lập x cho mối quan hệ tuyến tính: `y = 3x + 5 + noise`
- Khởi tạo trọng só w và b ngẫu nhiên
- Code chính:
```python
x = torch.rand(10,1) * 9 + 1
noise = torch.randn(1)
# 1. khởi tạo tham số w và b ngẫu nhiên 
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
```
- Xây dựng vòng lặp huấn luyện (100 vòng lặp ) tính toán MSE và cập nhật w, b bằng gradient Descent với Learning rate alpha = 0.01 để mô hình hội tụ về giá trị thực.
- Code chính: 
```python
for epoch in range(100):
    y_pred = w * x + b  # dự đoán theo mô hình tuyến tính
    loss = ((y_pred - y) ** 2).mean()  # cthuc mse = trung bình **2 sai số
    loss.backward()
    lr_alpha = 0.01
    with torch.no_grad():
        w -= lr_alpha * w.grad
        b -= lr_alpha * b.grad
```
- KQ:
<img width="1265" height="1138" alt="image" src="https://github.com/user-attachments/assets/6351cd77-d993-4a85-bd84-1a46c140652d" />

---
#### BTVN 4: giải thích 2 trường hợp trên
- Trường hợp 1: torch.from_numpy(arr): Tensor và Numpy dùng chung bộ nhớ nên  khi thay đổi arr thì x cũng thay đổi luôn
- Trường hợp 2: torch.tensor(arr): trường hợp này do tạo Tensor mới (copy dữ liệu) từ arr = np.arange(0,5) nên khi thay đổi arr thì x không đổi.

---
#### BTVN 5 : Tạo Tensor với Empyty, Zeros, Ones, Random, Reshape với view và view as
- Với Empyty: sử dụng câu lệnh torch.empty()
- KQ:
<img width="1036" height="294" alt="image" src="https://github.com/user-attachments/assets/b700796b-1e09-471b-ab28-97410c6278d7" />
- Với Zeros: sử dụng câu lệnh torch.zeros()
- KQ:
<img width="1234" height="315" alt="image" src="https://github.com/user-attachments/assets/46d4c324-8079-4c52-b694-67787cbee28b" />
- Với Ones : sử dụng câu lệnh torch.ones()
- KQ:
<img width="1257" height="354" alt="image" src="https://github.com/user-attachments/assets/9ae33af4-62a1-4aef-89a0-08af449d8ffa" />
- Với Random: sử dụng câu lệnh torch.rand()
- KQ:
<img width="1062" height="310" alt="image" src="https://github.com/user-attachments/assets/e49e7acb-dd22-4d89-b1b6-2165325ae82a" />
- Với Reshape: cần thực hiện biến đổi hình dạng của 1 vector từ 1 chiều sang 2 chiều(ma trận)
- KQ:
<img width="1081" height="414" alt="image" src="https://github.com/user-attachments/assets/3101162b-2cb4-494c-9b36-48ef642423f4" />












