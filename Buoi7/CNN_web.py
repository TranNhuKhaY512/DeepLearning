import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# ==========================================
# 1. ĐỊNH NGHĨA CÁC KIẾN TRÚC MÔ HÌNH
# ==========================================

# Mô hình CIFAR-10
class CIFAR10_AdvancedCNN(nn.Module):
    def __init__(self):
        super(CIFAR10_AdvancedCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Mô hình Plant Disease
class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ==========================================
# 2. LOAD MODELS & CONFIG
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CIFAR-10
cifar_model = CIFAR10_AdvancedCNN().to(device)
try:
    cifar_model.load_state_dict(torch.load("cifar10_model.pth", map_location=device))
    cifar_model.eval()
except: print("Warning: cifar10_model.pth not found.")

cifar_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
cifar_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

# Load Plant Disease
try:
    plant_checkpoint = torch.load("plant_model.pth", map_location=device)
    plant_model = PlantCNN(plant_checkpoint["num_classes"]).to(device)
    plant_model.load_state_dict(plant_checkpoint["model_state_dict"])
    plant_model.eval()
    plant_classes = plant_checkpoint["class_names"]
except:
    print("Warning: plant_model.pth not found.")
    plant_classes = []

plant_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ==========================================
# 3. HELPER FUNCTIONS (GRAD-CAM & PREDICT)
# ==========================================
features_blobs = []
def hook_fn(module, input, output):
    features_blobs.append(output)

cifar_model.block3.register_forward_hook(hook_fn)

def run_cifar(image):
    features_blobs.clear()
    img_tensor = cifar_transform(image).unsqueeze(0).to(device)
    img_tensor.requires_grad = True

    output = cifar_model(img_tensor)
    probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
    pred = output.argmax()

    # Grad-CAM logic
    cifar_model.zero_grad()
    output[0, pred].backward()
    
    fmap = features_blobs[0].detach().cpu().numpy()[0]
    cam = np.mean(fmap, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (32,32))
    cam = cam / (cam.max() + 1e-8)

    img_np = np.array(image.resize((32,32))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = (heatmap * 0.5 + img_np * 0.5)
    
    # Format labels for Top 5
    idx = np.argsort(probs)[::-1][:5]
    res_labels = {cifar_classes[i]: float(probs[i]) for i in idx}
    
    return res_labels, (overlay * 255).astype(np.uint8)

def run_plant(image):
    img = plant_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = plant_model(img)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    
    return {plant_classes[i]: float(probs[i]) for i in range(len(plant_classes))}

# ==========================================
# 4. GRADIO UI LAYOUT
# ==========================================
css = """
.main-header {text-align: center; color: #2563eb;}
.card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; background: white;}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Multi-Model Vision AI Platform", elem_classes="main-header")
    
    with gr.Tabs():
        # --- TAB 1: CIFAR-10 ---
        with gr.TabItem("🖼️ General Object Detection (CIFAR-10)"):
            with gr.Row():
                with gr.Column():
                    c_input = gr.Image(type="pil", label="Upload Image")
                    c_btn = gr.Button("Analyze Image", variant="primary")
                with gr.Column():
                    c_label = gr.Label(label="Top Predictions", num_top_classes=5)
                    c_cam = gr.Image(label="Model Attention (Grad-CAM)")
            
            c_btn.click(run_cifar, inputs=c_input, outputs=[c_label, c_cam])

        # --- TAB 2: PLANT DISEASE ---
        with gr.TabItem("🌱 Plant Disease Diagnosis"):
            gr.Markdown("### Upload a leaf image to identify potential diseases.")
            with gr.Row():
                with gr.Column():
                    p_input = gr.Image(type="pil", label="Leaf Image")
                    p_btn = gr.Button("Diagnose", variant="primary")
                with gr.Column():
                    p_label = gr.Label(label="Diagnosis Results", num_top_classes=3)
            
            p_btn.click(run_plant, inputs=p_input, outputs=p_label)

if __name__ == "__main__":
    demo.launch()