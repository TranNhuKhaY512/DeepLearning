# LSTM Applications: Time Series Forecasting & Next Word Prediction

## Giới thiệu

Bài lab này triển khai hai ứng dụng tiêu biểu của **mạng LSTM (Long Short-Term Memory)**:

1. **Dự đoán chuỗi thời gian (Time Series Forecasting)**
2. **Dự đoán từ tiếp theo (Next Word Prediction)**

Mục tiêu là giúp hiểu cách LSTM xử lý dữ liệu tuần tự trong hai lĩnh vực:

* Phân tích dữ liệu số theo thời gian
* Xử lý ngôn ngữ tự nhiên (NLP)

---

## ⚙️ Công nghệ sử dụng

* **Python** - 3.11.5
* **PyTorch** – xây dựng và huấn luyện mô hình LSTM
* **NumPy** – xử lý dữ liệu số
* **Matplotlib** – trực quan hóa kết quả
* **Scikit-learn** – chuẩn hóa dữ liệu (MinMaxScaler)

---

# 📈 Bài 1: Dự đoán chuỗi thời gian

## Mô tả: Sử dụng LSTM để dự đoán giá trị tiếp theo của một chuỗi dữ liệu liên tục.
## Dữ liệu
- Dữ liệu giả lập:
  + Chuỗi **sin + nhiễu (noise)**
- Mô phỏng dữ liệu thực tế như:
  + Nhiệt độ
  + Giá cổ phiếu
  + Doanh thu

---

##  Cách hoạt động

### 1. Chuẩn bị dữ liệu (Sliding Window) : 
```python
t = np.linspace(0, 50, 300)
data = np.sin(t) + 0.1 * np.random.randn(300)

def create_dataset(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 5
X, y = create_dataset(data, window_size)
```
---

### 2. Chia dữ liệu

* 80%: Train
* 20%: Test
```python
train_size = int(len(X) * 0.8)

X_train = X[:train_size]
X_test  = X[train_size:]

y_train = y[:train_size]
y_test  = y[train_size:]
```
---

### 3. Chuẩn hóa

* Sử dụng **MinMaxScaler**
* Giúp mô hình học nhanh và ổn định hơn
```python
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_X.fit_transform(X_train)
X_test  = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train.reshape(-1,1))
y_test  = scaler_y.transform(y_test.reshape(-1,1))
```
---

### 4. Kiến trúc mô hình

* 1 lớp **LSTM**
* 1 lớp **Fully Connected**
```python
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
X_test  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)

class TimeSeriesLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)
model = TimeSeriesLSTM()
```
---

### 5. Huấn luyện

* Loss: **MSE (Mean Squared Error)**
* Optimizer: **Adam**
```python
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
```
---

## 📊 Kết quả

* Mô hình học được xu hướng của dữ liệu
* Dự đoán gần với giá trị thực tế
* Sai số xuất hiện tại vùng có nhiễu cao
---

## Biểu đồ

* Biểu đồ **Loss giảm dần theo epoch**
<img width="1812" height="809" alt="image" src="https://github.com/user-attachments/assets/dfdf42fa-4a2b-4e44-8850-264a6873f328" />
* Biểu đồ **So sánh thực tế vs dự đoán**
<img width="1824" height="1023" alt="image" src="https://github.com/user-attachments/assets/e6601f9c-937a-4b8a-b90b-9e6e03e0b1ee" />

---

## Nhận xét

* LSTM phù hợp với dữ liệu chuỗi thời gian
* Hiệu quả với dữ liệu có tính tuần hoàn
* Cần cải thiện nếu dữ liệu phức tạp hơn

---

# Bài 2: Dự đoán từ tiếp theo

##  Mô tả

Sử dụng LSTM để dự đoán từ tiếp theo trong câu dựa trên ngữ cảnh.

---

##  Dữ liệu

```id="a2"
toi thich nghe nhac
toi thich xem phim
ban thich doc sach
chung toi thich an com
```

---

##  Cách hoạt động

### 1. Tiền xử lý

* Tokenization (tách từ)
* Tạo vocabulary
* Chuyển từ → số
```python
tokens = [sentence.split() for sentence in sentences]

# ===== Tạo vocabulary =====
vocab = set()
for sentence in tokens:
    vocab.update(sentence)

word2idx = {word: i+1 for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

vocab_size = len(word2idx) + 1
```
---

### 2. Tạo dữ liệu huấn luyện

```id="a3"
toi → thich  
thich → nghe  
nghe → nhac  
```
```python
X = []
y = []

for sentence in tokens:
    for i in range(len(sentence) - 1):
        X.append(word2idx[sentence[i]])
        y.append(word2idx[sentence[i+1]])

X = torch.tensor(X).unsqueeze(1)  # (samples, seq_len=1)
y = torch.tensor(y)
```

---

### 3. Kiến trúc mô hình

* **Embedding Layer**
* **LSTM Layer**
* **Fully Connected Layer**
```python
class LSTM_NextWord(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 10)
        self.lstm = nn.LSTM(10, 20, batch_first=True)
        self.fc = nn.Linear(20, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTM_NextWord(vocab_size)
```

---

### 4. Huấn luyện

* Loss: **CrossEntropyLoss**
* Optimizer: **Adam**
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item()
```

---

### 5. Dự đoán

```id="a4"
Input: "toi" → Output: "thich"  
Input: "thich" → Output: "nghe" / "xem" / "an"
```
```python
def predict(word):
    x = torch.tensor([[word2idx[word]]])
    out = model(x)
    pred = torch.argmax(out, dim=1).item()
    return idx2word.get(pred, "?")

print("toi ->", predict("toi"))
print("thich ->", predict("thich"))
```

---

### Kết quả

* Mô hình học được cấu trúc câu đơn giản
* Dự đoán hợp lý các từ tiếp theo
<img width="1955" height="1033" alt="image" src="https://github.com/user-attachments/assets/6799fbd7-6cee-4d39-8026-30e46be5f0ef" />

* Biểu đồ **Loss giảm dần theo epoch**
<img width="1559" height="1017" alt="image" src="https://github.com/user-attachments/assets/584d9114-674f-4a57-8724-3805fed50919" />

---

## Kết luận

LSTM là mô hình mạnh mẽ trong việc xử lý dữ liệu tuần tự:

* Với Time Series: học xu hướng theo thời gian
* Với NLP: học ngữ cảnh giữa các từ

Bài lab  giúp hiểu rõ cách áp dụng LSTM trong thực tế và là nền tảng cho các bài toán AI nâng cao.

