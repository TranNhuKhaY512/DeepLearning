
#  Cat vs Dog Image Classifier

Ứng dụng web nhận diện Chó và Mèo sử dụng **Deep Learning (CNN)** với framework **PyTorch** và giao diện **Gradio**. Ứng dụng cho phép người dùng tải lên hình ảnh và nhận về kết quả dự đoán độ tin cậy theo thời gian thực.

## Tính năng nổi bật

  * **Kiến trúc CNN nâng cao:** Sử dụng 4 tầng tích chập (Convolutional Layers) kết hợp BatchNorm và Dropout để tối ưu độ chính xác.
  * **Giao diện trực quan:** Tương tác dễ dàng qua trình duyệt web với Gradio.
  * **Xử lý thông minh:** Tự động nạp cấu trúc từ file `.pth` (input size, normalization, class names).
  * **Tính tương thích cao:** Hỗ trợ chạy trên cả CPU và GPU.

## Công nghệ sử dụng

  * **Ngôn ngữ:** Python 3.11+
  * **Framework:** PyTorch, Torchvision
  * **Giao diện:** Gradio
  * **Xử lý ảnh:** Pillow (PIL), NumPy

## Cấu trúc Mô hình (Architecture)

Mặc dù file được đặt tên là `ANN_web`, kiến trúc bên trong sử dụng mạng tích chập **CNN** để đạt hiệu suất cao nhất cho dữ liệu hình ảnh:

1.  **Convolutional Blocks:** 4 tầng (32 -\> 64 -\> 128 -\> 256 filters) kèm BatchNorm và ReLU.
2.  **Pooling:** MaxPool2d để giảm chiều dữ liệu.
3.  **Fully Connected:** 2 tầng Dense layer (512 units -\> 2 output classes).
4.  **Regularization:** Dropout (0.5) để chống Overfitting.

## Giao diện ứng dụng (Dashboard)
Giao diện được xây dựng bằng Gradio Blocks, tối ưu cho việc trình diễn và báo cáo khoa học:

Khu vực Input: Cho phép kéo thả ảnh hoặc tải lên từ bộ nhớ máy tính.

Khu vực Output: Hiển thị kết quả dưới dạng biểu đồ thanh (Confidence Score) trực quan.
<img width="2557" height="1258" alt="Screenshot 2026-04-03 213342" src="https://github.com/user-attachments/assets/a7529ae0-3258-4fd6-9181-0ff492fcba71" />
<img width="2559" height="1241" alt="image" src="https://github.com/user-attachments/assets/bda84321-a294-40a2-9306-d82d2319eb6e" />


## Cài đặt thư viện

```bash
pip install torch torchvision gradio pillow
```

### Tải file trọng số (Model Weights)

Do giới hạn dung lượng của GitHub, file `catdog_model.pth` (>25mb) cần được tải riêng:

Mô hình CNN (Chó & Mèo): [catdog_model.pth](https://drive.google.com/file/d/13pX_eKTrATqG0-h6vfcbpjlZ7WFIf9T1/view?usp=sharing)


> *Lưu ý: Đặt file này vào cùng thư mục với file `ANN_web.py`.*

## Hướng dẫn sử dụng

Chạy ứng dụng bằng lệnh sau:

```bash
python ANN_web.py
```

Sau khi chạy, truy cập vào đường dẫn mặc định: `http://127.0.0.1:7860`


