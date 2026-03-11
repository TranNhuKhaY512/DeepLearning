
# Thực hành:  Convolutional Neural Network (CNN)
## Sinh viên thực hiện: 
- Trần Như Khả Ý  
- Mã SV: 2374802010582  
- Giảng viên: Nguyễn Thái Anh
---
## Công nghệ sử dụng
- **Ngôn ngữ**: Python 
- **Notebook**: Jupyter Notebook
- **Thư viện**:
  - `numpy`: Xử lý số học, mảng.
  - `matplotlib`: Trực quan hóa dữ liệu.
  - `torch`: Hỗ trợ xây dựng và huấn luyện mô hình học sâu bằng tensor và GPU.
  - `torch.optim`: Cung cấp các thuật toán tối ưu hóa trọng số mô hình như SGD, Adam.
  - `torchvision`: Cung cấp dữ liệu ảnh, mô hình sẵn có và công cụ xử lý ảnh cho PyTorch.
  - `torchvision.transforms`: Thực hiện các phép biến đổi ảnh như chuyển tensor, chuẩn hóa.
---
## Mục tiêu
- Hiểu rõ vai trò và cơ chế hoạt động của từng thành phần trong CNN.
- Biết cách sử dụng tích chập, ReLU, pooling để xử lý ảnh.
- Biết áp dụng CNN để trích xuất đặc trưng và phân loại ảnh.
- Có thể giải thích quy trình nhận diện ảnh số đơn giản như “7” và các số khác (chữ viết tay) bằng CNN.
---
## Tổng quan lý thuyết
## 1. CNN là gì?
- CNN (Convolutional Neural Network) là một loại mạng nơ-ron nhân tạo giúp máy tính "nhìn" và hiểu ảnh, tương tự cách con người nhận diện vật thể trong đời thực. Thay vì xem toàn bộ ảnh một lúc như mạng nơ-ron thông thường (fully connected), CNN chia nhỏ ảnh ra, tìm các đặc trưng như đường thẳng, góc, vòng tròn, rồi ghép lại để đoán xem ảnh đó là gì.
- **Ví dụ**: Khi ta nhìn một con mèo, không cần xem hết cả ảnh ngay lập tức. Ta nhận ra tai mèo (hình tam giác), mắt mèo (hình tròn), ria mèo (đường thẳng), rồi kết luận "Đây là mèo". CNN cũng làm như vậy bằng cách dùng các "kính lúp" nhỏ quét qua ảnh từng phần một.
---
## 2. Các thành phần chính của CNN
### 2.1. Tầng tích chập (Convolution Layer)
Đây là bước quan trọng nhất, giống như "đôi mắt" của CNN, giúp tìm các đặc trưng nhỏ trong ảnh như cạnh, góc, hoặc đường cong.
#### Ý tưởng cơ bản
- Chúng ta có một ảnh, giả sử kích thước là $6 \times 6$ pixel.
- Dùng một **bộ lọc** (filter/kernel), ví dụ $3 \times 3$, như một "kính lúp" nhỏ để quét qua ảnh.
- Kết quả là một **feature map** (bản đồ đặc trưng), cho biết chỗ nào trong ảnh có đặc trưng mà bộ lọc tìm được.
#### Công thức tích chập
Công thức toán học của tích chập là:
```math
S(i, j) = \sum_{m=0}^{F-1} \sum_{n=0}^{F-1} I(i+m, j+n) \cdot K(m, n)
```
- `I`: Ảnh đầu vào (input image).
- `K`: Bộ lọc (kernel/filter).
- `F`: Kích thước bộ lọc (ví dụ $F=3$ nếu là $3 \times 3$).
- `S(i, j)`: Giá trị tại vị trí $(i, j)$ trong feature map.

### 2.2. Hàm kích hoạt (ReLU)
- Sau khi có feature map từ tầng tích chập, ta dùng hàm ReLU để "lọc" nó, giữ lại các đặc trưng rõ ràng và loại bỏ những phần không quan trọng.
#### Công thức
```math
 \text{ReLU}(x) = \max(0, x) 
```
**Giải thích đơn giản**:
- Nếu số lớn hơn 0, giữ nguyên.
- Nếu số nhỏ hơn hoặc bằng 0, biến thành 0.
**Ý nghĩa**:
- Các giá trị âm (như -1) thường là những vùng không rõ đặc trưng, nên bị loại bỏ.
- Các giá trị dương (như 1) là những nét chính (đường ngang, đường chéo), được giữ lại.

### 2.3. Tầng Pooling (Pooling Layer)
- Pooling giống như "tóm tắt" feature map, giảm kích thước để tiết kiệm tính toán nhưng vẫn giữ được thông tin quan trọng.
#### Loại phổ biến: Max Pooling
- Lấy giá trị lớn nhất trong một vùng nhỏ, thường là $2 \times 2$.
#### Công thức
```math
P(i, j) = \max(I(2i:2i+2, 2j:2j+2)) 
```
**Giải thích đơn giản**: Chia feature map thành các ô $2 \times 2$, chọn số lớn nhất trong mỗi ô để tạo feature map nhỏ hơn.
**Ý nghĩa**:
- Kích thước giảm từ $4 \times 4$ xuống $2 \times 2$.
- Giữ lại các giá trị lớn (nét chính), bỏ bớt chi tiết nhỏ.
### 2.4. Tầng Fully Connected (FC Layer)
- Đây là bước cuối cùng, nơi CNN ghép tất cả đặc trưng lại để đoán xem ảnh là gì.
#### Công thức
```math
 y = Wx + b 
```
- `x`: Vector từ feature map duỗi ra.
- `W`: Ma trận trọng số.
- `b`: Bias (độ lệch).
- **Bước cuối - Softmax**: Chuyển $y$ thành xác suất dự đoán.
- **Giải thích đơn giản**: Lấy feature map cuối, "duỗi" thành một hàng số, rồi nhân với trọng số để ra kết quả phân loại.
- **Ý nghĩa**: Tầng này giống như "bộ não" quyết định, dựa trên các đặc trưng đã tìm được.
  ---
### 3. Tổng hợp lại cả quy trình CNN
1. **Tích chập**: Tìm đặc trưng như đường ngang → Feature map $S$.
2. **ReLU**: Lọc bỏ các nét mờ (giá trị âm) → Feature map $S_relu$.
3. **Pooling**: Tóm tắt, giảm kích thước → Feature map $S_pooled$.
4. **Fully Connected**: Ghép các đặc trưng, dự đoán.

### 4. Ứng dụng thực tế
CNN được dùng trong:
- **Nhận diện khuôn mặt**: Facebook dùng CNN để gắn thẻ bạn bè trong ảnh.
- **Xe tự lái**: Phát hiện biển báo, người đi bộ qua camera.
- **Y khoa**: Phân tích ảnh X-quang để tìm bệnh.
- ...
---

## Nội dung thực hành
Trong lab này sử dụng dữ liệu MNIST là một tập dữ liệu hình ảnh số viết tay làm ảnh đầu vào để train mô hình nhận dạng chữ viết tay.
### Một số hàm quan trọng:
 1. __init__ (trong class MNIST_CNN)
- Vai trò: Khởi tạo các tầng của mô hình CNN:
- conv1: tầng tích chập đầu tiên (1→16 kênh)
- conv2: tầng tích chập thứ hai (16→32 kênh)
- pool: tầng giảm kích thước (max pooling)
- fc1: tầng fully connected (ẩn → 10 lớp)
2. forward(x): Định nghĩa luồng dữ liệu từ đầu vào → ra dự đoán
  ```python
    x = self.pool(torch.relu(self.conv1(x)))  # Conv1 -> ReLU (loại giá trị âm) -> Pool (giảm kích thước)
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool, cuối cùng ra 32x5x5
        x = self.pool(torch.relu(self.conv3(x)))
        x.view(-1, 64 * 1 * 1)  # Duỗi tensor thành vector, -1 tự động tính batch size
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)  # Qua tầng fully connected, ra 10 giá trị (logits cho 0-9)
        return x  # Trả về kết quả dự đoán
  ```
3. visualize_prediction(): Vẽ 5 ảnh đầu tiên từ tập kiểm tra và hiển thị
<img width="1043" height="343" alt="image" src="https://github.com/user-attachments/assets/f5085c90-9ca7-4798-b5d6-aa5d231b9b21" />

5. visualize_feature_map()
Vai trò: Trực quan hóa:
- Ảnh gốc (ví dụ ảnh số 7)
- Feature map của tầng conv1 (2 kênh đầu)
- Dùng để hiểu CNN học được gì sau lớp tích chập đầu tiên.
<img width="1044" height="503" alt="image" src="https://github.com/user-attachments/assets/6ec12350-d0da-452d-b381-57b83ecbbdc1" />

5. Vòng lặp huấn luyện:
- Vai trò: Huấn luyện mô hình (trong 10 lần,... - epoch).
- Trong đó:
  - loss.backward() và optimizer.step() là bước cập nhật trọng số.
  - loss_values, accuracy_values lưu lại quá trình huấn luyện để vẽ biểu đồ.
<img width="1051" height="464" alt="image" src="https://github.com/user-attachments/assets/b44ea9ac-0d5e-4627-b715-cc135a9494d3" />


6. Đánh giá mô hình:
```python
correct = 0  # Đếm số dự đoán đúng
total = 0  # Đếm tổng số mẫu
with torch.no_grad():  # Tắt tính gradient vì chỉ đánh giá, không huấn luyện
    for images, labels in test_loader:  # Lặp qua từng batch trong tập kiểm tra
        images, labels = images.to(device), labels.to(device)  # Chuyển dữ liệu sang GPU/CPU
        outputs = model(images)  # Dự đoán trên batch ảnh
        _, predicted = torch.max(outputs.data, 1)  # Lấy chỉ số lớp có giá trị lớn nhất (dự đoán)
        total += labels.size(0)  # Cộng số mẫu trong batch vào tổng
        correct += (predicted == labels).sum().item()  # Cộng số dự đoán đúng
print(f"Độ chính xác trên tập test: {100 * correct / total:.2f}%")  # Tính và in độ chính xác (%)
```
- Vai trò: Đánh giá độ chính xác mô hình trên tập test.
- Không tính gradient để tiết kiệm tài nguyên.

 7. Vẽ biểu đồ Loss & Accuracy
Vai trò: Hiển thị quá trình học theo thời gian (qua các epoch).
  <img width="1070" height="502" alt="image" src="https://github.com/user-attachments/assets/630c176e-9c8c-4219-8567-2eef65a21a15" />
---

## Hướng dẫn chạy
1. Mở file Jupyter Notebook:  
   `CNN-LAB.ipynb`
2. Chạy lần lượt từng cell để xem kết quả.
3. Kết quả cuối cùng sẽ hiển thị:
  - Độ chính xác (Accuracy) trên tập kiểm tra.
  - Biểu đồ Learning Curve:  hai biểu đồ: Loss theo từng epoch, Accuracy theo từng epoch
  - Trực quan hoá ảnh và dự đoán: hiển thị 5 ảnh viết tay từ tập test.
  - Trực quan hoá Feature Map: hiển thị: ảnh gốc từ tập test và Feature maps đầu ra từ tầng.
---



