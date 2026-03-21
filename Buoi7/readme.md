## THỰC HÀNH MÔN DEEP LEARNING - BUỔI 7
## LAB CNN tiếp theo
### Sinh viên thực hiện : Trần Như Khả Ý_ 2374802010582
### GVHD : Nguyễn Thái Anh
---

## Dự án Phân loại Hình ảnh với kiến trúc CNN thuần
Dự án này triển khai các mô hình Mạng thần kinh tích chập (CNN) tự định nghĩa để giải quyết 3 bài toán phân loại hình ảnh phổ biến. Mục tiêu cốt lõi là tối ưu hóa tham số để đạt độ chính xác trên 90% và kiểm soát overfitting mà không sử dụng các mô hình pre-trained như ResNet hay ConvNext.

### Công nghệ sử dụng
- Ngôn ngữ: Python.
- Deep Learning Framework: PyTorch.
- Thư viện bổ trợ:
  + Torchvision: Tải dữ liệu và thực hiện các phép biến đổi hình ảnh.
  + Matplotlib: Vẽ biểu đồ mất mát (Loss) và độ chính xác (Accuracy).
  + PIL (Pillow): Xử lý tệp hình ảnh đầu vào cho tập dữ liệu tùy chỉnh.

- Tối ưu hóa: AdamW và SGD kết hợp với ReduceLROnPlateau để điều chỉnh learning rate tự động.

 --- 
 
### BTVN
1. Với dataset CIFAR-10
- Mục tiêu: Phân loại 10 lớp đối tượng (máy bay, ô tô, chim, mèo,...).
- Kiến trúc: Sử dụng 3 khối (Block) tích chập với số kênh tăng dần (64 -> 128 -> 256).
- Code chính :
```python
self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
```

- Kỹ thuật: RandomCrop và RandomRotation để tăng cường dữ liệu nhỏ (32x32).
```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
```
- Cấu hình huấn luyện:
  - Dùng nn.CrossEntropyLoss() : Tối ưu cho phân loại đa lớp. Nó kết hợp LogSoftmax và NLLLoss trong một lớp duy nhất, giúp tính toán sự chênh lệch giữa dự đoán của mô hình và nhãn thực tế một cách hiệu quả.
  - optim.SGD (Stochastic Gradient Descent): Được cấu hình với các tham số nâng cao:
      - lr=0.01: Tốc độ học khởi đầu giúp mô hình hội tụ ổn định.
      - momentum=0.9: Giúp mô hình vượt qua các cực tiểu địa phương và tăng tốc độ hội tụ.
      - weight_decay=5e-4: Một kỹ thuật Regularization (L2 penalty) giúp kiểm soát độ lớn của trọng số, ngăn chặn hiện tượng Overfitting.
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
```
- Mô hình được huấn luyện trong 100 Epoch gồm các bước :
  - Lan truyền xuôi (Forward Pass): Hình ảnh được đưa qua mạng CNN để dự đoán nhãn.
  - Tính toán mất mát: Sử dụng CrossEntropyLoss để đo lường sai số giữa dự đoán và thực tế.
  - Lan truyền ngược (Backward Pass): Tính toán Gradient thông qua loss.backward() để xác định hướng cập nhật trọng số.
  - Tối ưu hóa: optimizer.step() cập nhật trọng số mô hình và optimizer.zero_grad() xóa bộ nhớ đệm gradient cho bước sau.
  - Theo dõi hiệu suất: Lưu lại giá trị Loss và Accuracy trung bình sau mỗi Epoch để vẽ biểu đồ trực quan.
  - Cập nhật thông minh: scheduler.step(epoch_acc) tự động giảm tốc độ học nếu độ chính xác không cải thiện, giúp mô hình hội tụ tốt hơn.
- KQ :
<img width="2403" height="1161" alt="image" src="https://github.com/user-attachments/assets/7b64cd76-00e5-49d3-a458-e0109a50844b" />
- Vẽ biểu đồ loss và acc:
<img width="2144" height="968" alt="image" src="https://github.com/user-attachments/assets/53211387-e8aa-4157-a169-ec523439bfd2" />
- Độ chính xác trên tập test: 
<img width="1269" height="142" alt="image" src="https://github.com/user-attachments/assets/afa62362-2c98-462f-a02b-4a02c4b05996" />
- Hiển thị kết quả thực tế :
<img width="1764" height="480" alt="image" src="https://github.com/user-attachments/assets/18c36ceb-1224-4305-a918-1a1afb0ccfa2" />

---

2. Với dataset Cats vs Dogs
- Mục tiêu: Phân loại nhị phân giữa chó và mèo.
- Tiền xử lý: Ảnh được đưa về kích thước 128x128 để giữ chi tiết tốt hơn so với CIFAR.
- Code chính :
```python
transform_train = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
```
- Cấu trúc mạng :
  - 4 Khối Tích chập (Convolutional Blocks): Mỗi khối gồm Conv2d (tăng dần từ 32 lên 256 kênh), kết hợp với BatchNorm2d để ổn định dữ liệu và ReLU làm hàm kích hoạt.
  - Giảm chiều dữ liệu: tầng MaxPool2d giúp giảm kích thước không gian ảnh xuống 2 lần, giúp mô hình tập trung vào các đặc trưng quan trọng.
  - Lớp Phân loại (Fully Connected): * Flatten: Chuyển đổi dữ liệu từ dạng 2D (256 kênh 8x8) sang 1D ($256 \times 8 \times 8$).
    - Dropout (0.5): Tắt ngẫu nhiên 50% neuron trong quá trình huấn luyện để chống "học vẹt" (overfitting).
    - Linear Layers: Hai tầng liên kết đầy đủ (512 unit và 2 unit đầu ra) để quyết định nhãn là "Mèo" hay "Chó".
```python
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)
```

- Thiết lập Huấn luyện:
  - Hàm mất mát (nn.CrossEntropyLoss): Sử dụng để đo lường sai lệch giữa dự đoán của mô hình và nhãn thực tế, tối ưu cho bài toán phân loại nhiều lớp.
  - Bộ tối ưu hóa (optim.AdamW):
    - lr=0.001: Tốc độ học khởi tạo giúp mô hình hội tụ nhanh và ổn định.
    - weight_decay=1e-2: Kỹ thuật Regularization (L2) mạnh mẽ giúp kiểm soát trọng số, trực tiếp ngăn chặn hiện tượng Overfitting.
  - Bộ điều chỉnh tốc độ học (ReduceLROnPlateau):
    - Cơ chế: Theo dõi độ chính xác (mode='max').
    - Điều kiện: Nếu độ chính xác không tăng sau 5 Epoch (patience=5), tốc độ học sẽ giảm đi một nửa (factor=0.5).
```python
model = CatDog_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
```
- Quy trình Huấn luyện Cat vs Dog (50 Epochs)
  - Chế độ Huấn luyện (model.train()): Kích hoạt các lớp đặc biệt như BatchNorm và Dropout để mô hình học cách tổng quát hóa dữ liệu, tránh học vẹt.
  - Các bước tối ưu hóa mỗi Batch:
    - Forward Pass: Chuyển hình ảnh lên thiết bị tính toán (GPU/MPS/CPU) và đưa qua mạng để dự đoán nhãn.
    - Tính Loss & Backpropagation: Tính toán sai số bằng CrossEntropyLoss, sau đó dùng loss.backward() để lan truyền ngược lỗi về các trọng số.
    - Cập nhật trọng số: optimizer.step() thực hiện điều chỉnh các tham số dựa trên Gradient đã tính.
  - Quản lý bộ nhớ: optimizer.zero_grad() được gọi đầu mỗi batch để đảm bảo gradient không bị cộng dồn từ các bước trước.
  - Giám sát tiến trình: Lưu trữ giá trị Loss trung bình và Accuracy (%) sau mỗi Epoch để theo dõi sự hội tụ của mô hình qua biểu đồ.
  - Điều chỉnh linh hoạt: scheduler.step(epoch_acc) tự động tinh chỉnh tốc độ học dựa trên độ chính xác đạt được, giúp mô hình ổn định ở các Epoch cuối.
- KQ :
- Train với 50 epochs
<img width="1857" height="1144" alt="image" src="https://github.com/user-attachments/assets/a5acfc79-f464-4121-86e2-a1b03430d62c" />
- Biểu đồ loss và acc:
<img width="1749" height="568" alt="image" src="https://github.com/user-attachments/assets/f1d11df7-f7c2-44ab-90ca-079d7ac8ea6b" />
- Độ chính xác :
<img width="704" height="127" alt="image" src="https://github.com/user-attachments/assets/027731b7-5643-4c09-99bb-085629ada4ab" />
- Hiện thị kết quả thực tế :
<img width="1723" height="564" alt="image" src="https://github.com/user-attachments/assets/6209aebb-ee31-4afa-a1aa-e0d4a32ba5f9" />

---

3. Với dataset PlantVillage
- Mục tiêu: Nhận diện các loại bệnh trên lá cây trồng.
- Quy mô: Phân loại đa lớp với số lượng lớp lớn hơn (tùy thuộc vào dữ liệu trong thư mục color).
- Tối ưu: Sử dụng cấu trúc conv_layer đóng gói sẵn (Conv + BatchNorm + ReLU + MaxPool) để mã nguồn sạch sẽ và hiệu quả.
```python
def conv_layer(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
        self.features = nn.Sequential(
            conv_layer(3, 32),   
            conv_layer(32, 64),  
            conv_layer(64, 128), 
            conv_layer(128, 256) 
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
```
- Cấu hình huấn luyện:
  - Hàm mất mát (nn.CrossEntropyLoss): Lựa chọn tối ưu cho phân loại đa lớp, giúp tính toán sai số giữa các loại bệnh cây trồng khác nhau.
  - Bộ tối ưu hóa (optim.AdamW):
    - lr=0.001: Tốc độ học khởi tạo cân bằng giữa việc học nhanh và ổn định.
    - weight_decay=1e-2: Kỹ thuật Regularization L2 giúp kiểm soát các trọng số mạng, ngăn chặn hiện tượng Overfitting khi học trên tập dữ liệu hình ảnh lá cây phức tạp.
  - Bộ điều chỉnh tốc độ học (ReduceLROnPlateau):
    - patience=3: Khắt khe hơn (chỉ chờ 3 Epoch thay vì 5). Nếu độ chính xác không tăng, hệ thống sẽ can thiệp ngay lập tức.
    - factor=0.5: Giảm một nửa tốc độ học để giúp mô hình hội tụ cực chi tiết vào các đặc trưng nhỏ trên lá.
```python
model = PlantCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
```

- Quy trình Huấn luyện PlantVillage (40 Epochs): 
  - Thiết lập Chế độ (model.train()): Đảm bảo các lớp BatchNorm và Dropout hoạt động để tối ưu hóa khả năng tổng quát hóa trên các loại lá khác nhau
  - Chu trình Tối ưu hóa:
    - Xóa Gradient: optimizer.zero_grad() ngăn chặn việc cộng dồn sai số từ các vòng lặp trước.
    - Tính toán & Lan truyền: Dự đoán nhãn (out), tính toán hàm mất mát (loss), và thực hiện lan truyền ngược (loss.backward()) để cập nhật trọng số (optimizer.step()).
  - Thống kê Hiệu suất:
    - Sử dụng out.max(1) để xác định nhãn dự đoán có xác suất cao nhất.
    - Tính toán độ chính xác (Accuracy %) dựa trên số lượng dự đoán đúng (correct) trên tổng mẫu (total).
  - Điều chỉnh Tốc độ Học: scheduler.step(epoch_acc) can thiệp ngay sau mỗi Epoch để tinh chỉnh Learning Rate, giúp mô hình vượt qua các vùng hội tụ chậm và đạt độ chính xác tối ưu.
- KQ :
<img width="1817" height="1158" alt="image" src="https://github.com/user-attachments/assets/032f0533-92d9-45f9-9a6f-3267b2fea408" />
- Biểu đồ loss và acc:
<img width="1749" height="800" alt="image" src="https://github.com/user-attachments/assets/e8bfd766-866c-4800-a671-81351853cd17" />
- Độ chính xác :
<img width="910" height="115" alt="image" src="https://github.com/user-attachments/assets/641fab66-6592-4169-81e4-957f8bcfff6a" />
- Hiển thị kết quả thực tế :
<img width="1738" height="489" alt="image" src="https://github.com/user-attachments/assets/df0e17dd-8b5a-47e7-940b-a0745901d720" />

---








