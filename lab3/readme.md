# Pandas_06022026 – Thực hành Pandas với Series và DataFrame
# DEEP LEARNING – Tuần 3
## Pandas 
### Sinh viên thực hiện: Trần Như Khả Ý _ 2374802010582
### GVHD : Nguyễn Thái Anh


## 1. Giới thiệu
Notebook `Pandas_06022026.ipynb` được xây dựng nhằm mục đích thực hành thư viện Pandas trong Python. Nội dung tập trung vào hai cấu trúc dữ liệu quan trọng là Series và DataFrame, giúp người học hiểu cách tổ chức, truy cập và thao tác dữ liệu dạng bảng.

---

## 2. Công nghệ và thư viện sử dụng
- Python 3.11.5
- Pandas : Thao tác và phân tích dữ liệu dạng bảng
- numpy : Hỗ trợ mảng số học và so sánh với Pandas
---

## 3. Nội dung và cách hoạt động
## 3.1. Pandas Series
### 3.1.1. Mục tiêu
- Hiểu khái niệm Pandas Series
- Biết cách tạo, truy cập và thao tác dữ liệu một chiều
- So sánh Series với list Python và NumPy array

### 3.1.2. Cách hoạt động

1. **Tạo Series từ list Python**  
Dữ liệu đầu vào là một danh sách Python. Pandas tự động sinh index mặc định cho từng phần tử.
```python
data_pd = pd.Series([0.25, 0.5, 0.75, 1.0])
print("Pandas series from list:\n", data_pd)
```
<img width="585" height="217" alt="image" src="https://github.com/user-attachments/assets/565df310-f79b-4dfe-8257-81ae51a6d53d" />


2. **Tạo Series từ NumPy array**  
Series được tạo từ mảng NumPy nhưng có thêm index, giúp quản lý dữ liệu tốt hơn so với mảng thuần NumPy.
```python
import numpy as np

numpy_arr = np.arange(5)
data_pd = pd.Series(numpy_arr)
print("Pandas series from numpy array:\n", data_pd)
```
<img width="639" height="244" alt="image" src="https://github.com/user-attachments/assets/8bb944b4-f16b-445b-bc64-9cde5e06088d" />


3. **Truy cập dữ liệu trong Series**
- Truy cập theo vị trí
- Truy cập theo nhãn (label)
- Cắt dữ liệu (slicing)
```python
print("Data[1]:", data_pd[1])
print("Data[-2:]:\n", data_pd[-2:])
```
<img width="487" height="232" alt="image" src="https://github.com/user-attachments/assets/421bcef7-76f0-41e0-b2f8-08e2dd4d4770" />

4. Series với index tùy chỉnh
- Index được gán thủ công giúp truy cập dữ liệu theo nhãn.
```python
data_pd = pd.Series([0.25, 0.5, 0.75, 1.0],
                    index=['a', 'b', 'c', 'd'])
print(data_pd)
```

5. Truy cập dữ liệu theo label
- Pandas cho phép truy cập bằng label thay vì chỉ số.
```python
print("data_pd['a']:", data_pd['a'])
print("data_pd['b':'d']:\n", data_pd['b':'d'])
```

6. Phép toán trên Series
- Phép toán được áp dụng cho từng phần tử trong Series.
```python
data_pd * 2
data_pd + 1
```

7. Series và NumPy array
- np.exp(data_pd)
- Pandas tương thích tốt với các hàm NumPy.

8. Kết quả phần Series
- Hiểu Series là dữ liệu một chiều có index.
- Biết tạo Series từ list và NumPy array.
- Truy cập, cắt, lọc và tính toán trên Series.
- Hiểu sự khác biệt giữa Series và NumPy array
  
### 3.1.3. Kết quả
- Hiểu rõ cấu trúc Series gồm values và index
- Thực hiện được các thao tác cơ bản trên dữ liệu một chiều
- Nhận thấy ưu điểm của Series so với list và NumPy array

---

## 3.2. Pandas DataFrame

### 3.2.1. Mục tiêu
- Hiểu khái niệm Pandas DataFrame
- Làm việc với dữ liệu dạng bảng (tương tự Excel hoặc bảng SQL)
- Chuẩn bị nền tảng cho các bước phân tích dữ liệu nâng cao

### 3.2.2. Cách hoạt động
1. **Tạo DataFrame từ dictionary**  
Mỗi key trong dictionary tương ứng với một cột, mỗi value là danh sách dữ liệu của cột đó.
```python
some_population_dict = {'Sai Gon': 11111, 
                        'Vung Tau': 22222,
                        'Phan Thiet': 33333,
                        'Vinh Long': 44444}
some_area_dict = {'Sai Gon': 99999, 
                'Vung Tau': 88888,
                'Phan Thiet': 77777,
                'Vinh Long': 66666,
                 'Ben Tre': 33333}

states = pd.DataFrame({'population': some_population_dict,
                       'area': some_area_dict})
print(states)
```
<img width="633" height="218" alt="image" src="https://github.com/user-attachments/assets/4d442469-53bb-462a-94a1-53f1f965180b" />


2. **Khám phá cấu trúc DataFrame**
- Xem dữ liệu đầu và cuối bảng
- Kiểm tra số dòng (shape), số cột(columns)
- Lấy danh sách cột và index
```python
print("Index:        ", states.index)  # pandas index object
print("Index[-1]:    ", states.index[-1])  # pandas index object is similar to numpy array
print("Columns:      ", states.columns)  # pandas index object
print("Columns[0:1]: ", states.columns[0:1]) # notice how 1 is not included
```

3. **Truy cập dữ liệu**
- Truy cập theo cột
- Truy cập nhiều cột
- Truy cập theo dòng và cột
```python
print("Only area column with everything")
print(states['area'])  # first way
print(states[:]['area']) # second way
```
<img width="711" height="445" alt="image" src="https://github.com/user-attachments/assets/f176a832-c0ac-460f-9f93-ea47d0cb043c" />

4. **Thêm và xóa cột**
- Thêm cột mới bằng phép gán
- Xóa cột không cần thiết để phục vụ tiền xử lý dữ liệu

5. **Phép toán trên DataFrame**
- Thực hiện các phép toán trên từng cột
- Áp dụng toán học cho toàn bộ bảng dữ liệu

### 3.2.3. Kết quả
- Hiểu DataFrame là cấu trúc dữ liệu trung tâm của Pandas
- Thao tác được với dữ liệu dạng bảng
- Sẵn sàng cho các bước xử lý và phân tích dữ liệu thực tế

---

## 4. Kết quả đạt được
Sau khi hoàn thành notebook, có thể:
- Nắm vững khái niệm Series và DataFrame
- Tạo và thao tác dữ liệu bằng Pandas
- Chuẩn bị nền tảng cho các chủ đề nâng cao.

---

## 5. Cách chạy notebook

```bash
pip install pandas numpy
jupyter notebook Pandas_06022026.ipynb
