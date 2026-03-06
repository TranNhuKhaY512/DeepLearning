# DEEP LEARNING – Tuần 5
## ANN 
### Sinh viên thực hiện: Trần Như Khả Ý _ 2374802010582
### GVHD : Nguyễn Thái Anh

## Giới thiệu
Bài lab này tập trung vào việc xây dựng và huấn luyện **Artificial Neural Network (ANN)** để thực hiện bài toán **phân loại ảnh (image classification)**.

Hai bộ dữ liệu được sử dụng:
- **MNIST Dataset**: tập dữ liệu chữ số viết tay từ 0–9.
- **Dogs vs Cats Dataset**: tập dữ liệu phân loại ảnh chó và mèo.

Mục tiêu của bài lab:
- Hiểu cách xây dựng mô hình **ANN bằng PyTorch**.
- Thực hiện quá trình **train và đánh giá mô hình**.
- So sánh hiệu quả mô hình trên hai loại dữ liệu khác nhau.

---

# Công nghệ sử dụng

- **Ngôn ngữ:** Python 3

### Thư viện
- **PyTorch (torch)** : Framework học sâu dùng để xây dựng và huấn luyện mạng neural.
- **Torchvision** : Cung cấp các dataset và công cụ xử lý ảnh.
- **NumPy** : Xử lý dữ liệu dạng mảng.
- **Matplotlib** : Hiển thị kết quả huấn luyện và trực quan hóa dữ liệu.
---

# Phần 1: Huấn luyện ANN với MNIST

## Dataset
MNIST là bộ dữ liệu gồm:

- 60,000 ảnh train
- 10,000 ảnh test
- Kích thước ảnh: **28 × 28**
- 10 lớp (digits 0–9)

---

## Các bước huấn luyện:
1. Chuẩn bị dữ liệu
2. Xây dựng ANN bằng PyTorch
3. Kiểm tra mô hình ANN
4. Visualization
5. Trực quan hóa kết quả
   
## Kết quả và biểu đồ mất mát
Sau khi train:
- Mô hình học được đặc trưng của chữ số viết tay
- Accuracy thường đạt khoảng 95–98%
<img width="656" height="802" alt="image" src="https://github.com/user-attachments/assets/4dc41be2-c006-43b1-99a8-1c5724914104" />
<img width="629" height="808" alt="image" src="https://github.com/user-attachments/assets/e803b569-4d7e-4268-8fc0-1bdbd1889799" />
<img width="773" height="811" alt="image" src="https://github.com/user-attachments/assets/13070020-2a7a-4a3c-b3a2-66592156cffc" />
<img width="822" height="544" alt="image" src="https://github.com/user-attachments/assets/97acfb98-1e1b-41ec-ab60-0f72fecbc978" />

<img width="1432" height="713" alt="image" src="https://github.com/user-attachments/assets/66df2206-5936-4a13-841d-661f18a5ecdf" />

# Phần 2: Huấn luyện ANN với Dogs vs Cats
## Dataset
- Dogs vs Cats là dataset phân loại ảnh gồm 2 lớp:
  + Dog
  + Cat
- Ảnh thường có kích thước lớn nên cần resize trước khi đưa vào mô hình.

## Các bước huấn luyện:
1. Chuẩn bị dữ liệu
2. Xây dựng ANN bằng PyTorch
3. Kiểm tra mô hình ANN
4. Visualization
5. Trực quan hóa kết quả

## Kết quả và biểu đồ mất mát
Sau khi huấn luyện:
  + Mô hình có thể phân biệt dog và cat
  + Accuracy phụ thuộc vào: số lượng dữ liệu, số epoch, kiến trúc mạng
<img width="857" height="297" alt="image" src="https://github.com/user-attachments/assets/f1c22d8e-b20e-4b56-b704-d824f244f77e" />

<img width="1350" height="578" alt="image" src="https://github.com/user-attachments/assets/3aad4a43-7b21-4e66-a46a-42b863c21433" />




