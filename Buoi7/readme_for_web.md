
# Multi-Model Vision AI Platform

Một ứng dụng web tích hợp trí tuệ nhân tạo mạnh mẽ, cho phép nhận diện vật thể đa lớp và chẩn đoán bệnh lý cây trồng thông qua hình ảnh. Ứng dụng được xây dựng dựa trên framework **PyTorch** cho mô hình Deep Learning và **Gradio** cho giao diện người dùng.

## Tính năng nổi bật

Trang web bao gồm hai công cụ phân tích thị giác máy tính chuyên biệt:

### 1. Phân loại vật thể tổng quát (CIFAR-10)
* **Khả năng:** Nhận diện 10 loại đối tượng phổ biến (máy bay, ô tô, chim, mèo, nai, chó, ếch, ngựa, tàu thủy, xe tải).
* **Công nghệ tích hợp (Grad-CAM):** Không chỉ đưa ra kết quả, hệ thống còn hiển thị "Bản đồ nhiệt" (Heatmap) giải thích vùng hình ảnh nào mà mô hình đang tập trung vào để đưa ra quyết định. Điều này giúp tăng tính minh bạch và tin cậy của AI.

### 2. Chẩn đoán bệnh lý cây trồng (Plant Disease)
* **Khả năng:** Phân tích hình ảnh lá cây để phát hiện các dấu hiệu bệnh lý sớm.
* **Mục tiêu:** Hỗ trợ người làm nông nghiệp xác định nhanh tình trạng sức khỏe của cây trồng để có biện pháp xử lý kịp thời.

---

## 🛠 Công nghệ sử dụng

* **Ngôn ngữ:** Python
* **Deep Learning:** PyTorch (Kiến trúc CNN nâng cao với BatchNorm, Dropout và Residual Blocks).
* **Xử lý ảnh:** OpenCV, Pillow, Torchvision.
* **Giao diện:** Gradio (Thiết kế Responsive, hỗ trợ Tabs tiện lợi).
* **Giải thích mô hình:** Kỹ thuật Grad-CAM (Gradient-weighted Class Activation Mapping).

---

## Hướng dẫn cài đặt
Vì file trọng số của mô hình có kích thước lớn (>25MB), vui lòng tải file trực tiếp từ Google Drive theo đường dẫn dưới đây và lưu vào thư mục gốc của dự án trước khi chạy:

Mô hình CNN (Chó & Mèo): [catdog_model.pth](https://drive.google.com/file/d/13pX_eKTrATqG0-h6vfcbpjlZ7WFIf9T1/view?usp=sharing)

Mô hình CNN (CIFAR-10): [cifar10_model.pth](https://drive.google.com/file/d/1VVDSx6RB2XnYpHbyAws8NiIBOWTSxiOP/view?usp=sharing)

Mô hình Plant Disease: [plant_model.pth](https://drive.google.com/file/d/1bYqR4h6ytnm-kayHIr2EhiynL2oE57GK/view?usp=sharing)

Để chạy dự án trên máy cục bộ, hãy thực hiện các bước sau:

### 1. Cài đặt thư viện cần thiết
```bash
pip install torch torchvision gradio pillow numpy opencv-python
```

### 2. Chuẩn bị file Model
Đảm bảo bạn đã có các file trọng số đã được huấn luyện nằm cùng thư mục với mã nguồn:
* `cifar10_model.pth`
* `plant_model.pth`

### 3. Khởi chạy ứng dụng
```bash
python CNN_web.py
```
Sau khi chạy, một đường dẫn local sẽ xuất hiện. Bạn chỉ cần copy và dán vào trình duyệt để bắt đầu trải nghiệm.

---

## 📸 Giao diện ứng dụng

Giao diện được chia làm 2 Tab rõ rệt:
* **Tab chính:** Ô tải ảnh lên (Drag & Drop) và nút điều khiển.
* **Bảng kết quả:** Hiển thị xác suất của các lớp (Top Predictions) bằng biểu đồ trực quan.
* **Cửa sổ Attention:** Hiển thị ảnh Grad-CAM cho thấy tư duy của máy tính.
<img width="2552" height="1356" alt="image" src="https://github.com/user-attachments/assets/31297b96-7c2b-4187-a66a-83585d0f7794" />
<img width="2548" height="1268" alt="image" src="https://github.com/user-attachments/assets/e0cb98ac-5c02-45d7-97f1-2f7dd6a16b8e" />


---

