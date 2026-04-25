"""
train_model.py — Run this in Google Colab to train the model
Then download best_model.pth and put it in model/ folder
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ── Setup ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# ── Data ──
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── PATHS — Change if needed ──
BASE = "chest_xray/chest_xray"   # Kaggle download path in Colab

train_data = datasets.ImageFolder(f"{BASE}/train", transform=train_transform)
val_data   = datasets.ImageFolder(f"{BASE}/val",   transform=test_transform)
test_data  = datasets.ImageFolder(f"{BASE}/test",  transform=test_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False, num_workers=2)

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
print(f"Classes: {train_data.classes}")

# ── Model ──
model = models.resnet18(pretrained=True)
for param in list(model.parameters())[:-10]:
    param.requires_grad = False
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
model = model.to(device)

# ── Training ──
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_acc = 0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

EPOCHS = 15
for epoch in range(EPOCHS):
    # Train
    model.train()
    t_loss = t_correct = t_total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        t_correct += out.argmax(1).eq(labels).sum().item()
        t_total += labels.size(0)

    # Validate
    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            v_loss += criterion(out, labels).item()
            v_correct += out.argmax(1).eq(labels).sum().item()
            v_total += labels.size(0)

    t_acc = 100. * t_correct / t_total
    v_acc = 100. * v_correct / v_total
    history["train_loss"].append(t_loss / len(train_loader))
    history["val_loss"].append(v_loss / len(val_loader))
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)

    if v_acc > best_acc:
        best_acc = v_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  💾 Saved!")

    scheduler.step()
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | Best: {best_acc:.1f}%")

# ── Test ──
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
correct = total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
print(f"\n✅ Test Accuracy: {100.*correct/total:.2f}%")
print("✅ best_model.pth saved — download it!")

# ── Plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history["train_loss"], label="Train", color="#e74c3c")
ax1.plot(history["val_loss"],   label="Val",   color="#3498db")
ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(history["train_acc"], label="Train", color="#e74c3c")
ax2.plot(history["val_acc"],   label="Val",   color="#3498db")
ax2.set_title("Accuracy %"); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
