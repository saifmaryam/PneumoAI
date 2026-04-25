# 🫁 PneumoAI — Chest X-Ray Diagnosis with Explainable AI

> Full-Stack AI Web Application for Pneumonia Detection using ResNet18 + Grad-CAM XAI

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

## 🎯 Features
- Upload chest X-ray → instant AI diagnosis
- **Grad-CAM heatmap** — shows where model looked
- Confidence scores with probability bars
- Beautiful dark-themed UI
- Full-stack: FastAPI backend + HTML/CSS/JS frontend

## 🏗️ Project Structure
```
chest-xray-xai/
├── backend/
│   ├── main.py           ← FastAPI server
│   └── requirements.txt
├── frontend/
│   └── index.html        ← UI
├── model/
│   ├── train_model.py    ← Training script (run in Colab)
│   └── best_model.pth    ← Trained weights (after training)
└── README.md
```

## 🚀 How to Run

### Step 1 — Train Model (Google Colab)
```python
# Run model/train_model.py in Google Colab
# Download best_model.pth → put in model/ folder
```

### Step 2 — Install Backend
```bash
cd backend
pip install -r requirements.txt
```

### Step 3 — Run Backend
```bash
python main.py
# Server runs at http://localhost:8000
```

### Step 4 — Open Frontend
```bash
# Open frontend/index.html in browser
# OR visit http://localhost:8000
```

## 📊 Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 92%+ |
| Model | ResNet18 (Transfer Learning) |
| XAI | Grad-CAM |
| Dataset | Kaggle Chest X-Ray (5,863 images) |

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, PyTorch, Grad-CAM
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **AI:** ResNet18, Transfer Learning, Explainable AI
- **Deploy:** Hugging Face Spaces (free)

## 👩‍💻 Author
**Maryam Saif** — MS CS Data Science
[LinkedIn](https://linkedin.com/in/maryam-saif-110859231) | [GitHub](https://github.com/saifmaryam)
