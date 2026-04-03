import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import os

# ===== 1. ĐỊNH NGHĨA ĐÚNG CẤU TRÚC MÔ HÌNH (Dựa trên lỗi báo) =====
# Lưu ý: Mặc dù bạn đặt tên file là ANN_web, nhưng file .pth của bạn là CNN
class CatDog_Model_Fixed(nn.Module):
    def __init__(self):
        super().__init__()
        # Các lớp Conv và BN dựa trên thông tin "Unexpected key" trong lỗi của bạn
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Các lớp Fully Connected (fc)
        # 8*8 là kích thước sau 4 lần MaxPool từ ảnh 128x128
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ===== 2. LOAD MODEL =====
device = torch.device("cpu")
model = CatDog_Model_Fixed()

try:
    checkpoint = torch.load("catdog_model.pth", map_location=device)
    
    # Load trọng số từ key "model_state_dict"
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ Đã khớp trọng số thành công!")
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    
    # Lấy các thông số phụ trợ
    class_names = checkpoint.get("class_names", ["Mèo 🐈", "Chó 🐕"])
    input_size = checkpoint.get("input_size", 128)
    mean = checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalize_std", [0.229, 0.224, 0.225])

except Exception as e:
    print(f"❌ Lỗi: {e}")
    class_names = ["Mèo 🐈", "Chó 🐕"]
    input_size = 128
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

# ===== 3. TRANSFORMS & PREDICT =====
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def predict(image):
    if image is None: return None
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

# ===== 4. GIAO DIỆN =====
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="🐶🐱 Cat vs Dog Classifier",
    description="Nhập ảnh chó mèo"
)

if __name__ == "__main__":
    demo.launch()