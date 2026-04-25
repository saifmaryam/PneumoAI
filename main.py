from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import base64
import io
import os
import cv2
import uvicorn

app = FastAPI(title="Chest X-Ray AI Diagnosis API")

# ── CORS (Frontend se connect karne ke liye) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve Frontend ──
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("../frontend/index.html")

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Model Load ──
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    # Model weights load karo agar available ho
    weights_path = "../model/best_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(
            torch.load(weights_path, map_location=device))
        print("✅ Model weights loaded!")
    else:
        print("⚠️ No weights found — using untrained model")
    model.eval()
    return model.to(device)

model = load_model()

# ── Transform ──
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CLASSES = ["NORMAL", "PNEUMONIA"]
CLASS_LABELS = {
    "NORMAL": "Normal — No Pneumonia Detected",
    "PNEUMONIA": "Pneumonia Detected"
}
CLASS_COLORS = {
    "NORMAL": "#00f5c4",
    "PNEUMONIA": "#ff4757"
}
CLASS_EMOJI = {
    "NORMAL": "✅",
    "PNEUMONIA": "🔴"
}

def image_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ── Validate file ──
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")

    # ── Read Image ──
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    img_resized = img_pil.resize((224, 224))
    img_array = np.array(img_resized) / 255.0

    # ── Prepare tensor ──
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # ── Prediction ──
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(1).item()

    pred_class = CLASSES[pred_idx]
    confidence = probabilities[pred_idx].item() * 100
    normal_prob = probabilities[0].item() * 100
    pneumonia_prob = probabilities[1].item() * 100

    # ── Grad-CAM ──
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_visualization = show_cam_on_image(
        img_array.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    # ── Original image base64 ──
    original_b64 = image_to_base64(np.array(img_resized))
    # ── Grad-CAM image base64 ──
    gradcam_b64 = image_to_base64(cam_visualization)
    # ── Heatmap only ──
    heatmap = cv2.applyColorMap(
        np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_b64 = image_to_base64(heatmap_rgb)

    return {
        "prediction": pred_class,
        "label": CLASS_LABELS[pred_class],
        "confidence": round(confidence, 2),
        "color": CLASS_COLORS[pred_class],
        "emoji": CLASS_EMOJI[pred_class],
        "probabilities": {
            "normal": round(normal_prob, 2),
            "pneumonia": round(pneumonia_prob, 2)
        },
        "images": {
            "original": original_b64,
            "gradcam": gradcam_b64,
            "heatmap": heatmap_b64
        }
    }

@app.get("/health")
def health():
    return {
        "status": "running",
        "device": str(device),
        "model": "ResNet18 + Grad-CAM XAI"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
