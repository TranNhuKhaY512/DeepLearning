# DEEP LEARNING – Tuần 2
## Numpy, Pandas & Matplotlib
### Sinh viên thực hiện: Trần Như Khả Ý _ 2374802010582
### GVHD : Nguyễn Thái Anh

## Giới thiệu
Trong bài lab này tập trung vào việc làm quen và thực hành với **NumPy, Pandas và Matplotlib** trong Python.
Nội dung chủ yếu xoay quanh:

* Xử lý mảng và ma trận
* Thao tác dữ liệu dạng bảng
* Rèn luyện tư duy logic thông qua bài toán nhỏ (ví dụ trò chơi dạng caro )

---

## 🛠 Công nghệ sử dụng

* **Python 3**
* **NumPy** : Làm việc với mảng, ma trận, phép toán số học, xử lý dữ liệu số.

---

## Cách hoạt động & nội dung chính

### NumPy: 
1. BTVN 1:
- Khởi tạo ma trận 3x3 với giá trị 99 (ô trống).
- Quy ước:
  - 1 → người chơi X
  - 0 → người chơi O
- Hai người chơi nhập tọa độ (row, col) luân phiên.
- Nếu ô đã được điền → yêu cầu nhập lại.
- Sau mỗi lượt, ma trận được cập nhật.
- Kiểm tra nếu một người có đủ 3 ô liên tiếp thì dừng trò chơi.
  
- Kết quả
  - Ma trận được cập nhật đúng theo lượt chơi.
  - Không cho phép ghi đè lên ô đã đánh.
  - In ra trạng thái bàn cờ sau mỗi lượt.
    
<img width="930" height="493" alt="image" src="https://github.com/user-attachments/assets/d21d7f1c-6644-435d-a572-4857a78fead5" />

---

2. BTVN2
![Uploading image.png…]()

4. BTVN3:
- Cách hoạt động
    - Dùng if trong vòng lặp for
    - Dùng list comprehension
- Kết quả: Các số chẵn được xuất ra
<img width="933" height="540" alt="image" src="https://github.com/user-attachments/assets/88afac27-425b-4e4d-8073-3e74f67eff96" />

5. BTVN4:
- Cách hoạt động
  - Tạo mảng NumPy 150 × 5 (giả lập dữ liệu sinh viên):
    - 4 cột đầu: đặc trưng (X)
    - 1 cột cuối: nhãn (y)
  - Tách dữ liệu:
    - X = data[:, :4]
    - y = data[:, 4]
  - Chia tập dữ liệu:
    - 70% train
    - 30% test(dùng train_test_split)

- Kết quả:
  - X_train, X_test, y_train, y_test được tạo đúng kích thước.
  - Dữ liệu sẵn sàng cho bước huấn luyện mô hình
<img width="932" height="540" alt="image" src="https://github.com/user-attachments/assets/c03d8ddb-4a71-470c-b187-347c4dfa8559" />

<img width="886" height="350" alt="image" src="https://github.com/user-attachments/assets/5e669b4c-955d-4b26-9c73-d8cc2b8172d7" />


## Cách chạy
1. Cài đặt Python
2. Cài các thư viện cần thiết:

```bash
pip install numpy 
```

3. Mở Jupyter Notebook:

```bash
jupyter notebook
```

4. Mở file `numpy_pandas.ipynb` và chạy từng cell từ trên xuống.

---

## Kết quả đạt được
* Hiểu và sử dụng được NumPy cho xử lý ma trận

---
