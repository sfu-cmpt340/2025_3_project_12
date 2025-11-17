import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ==============================
# CONFIG
# ==============================
PH2_IMAGES_ROOT = r"C:/Users/User/Desktop/matlab/Project/PH2Dataset/PH2 Dataset images"   # IMDxxx folders live here
BENIGN_CSV_PATH = r"C:/Users/User/Desktop/matlab/Project/PH2Dataset/false.csv"            # benign IDs (IMDxxx) in one column

MODEL_SAVE_PATH = "real_resnet18_best.pth"

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_EPOCHS  = 10
LR          = 1e-3
NUM_WORKERS = 0   # keep 0 on Windows
SEED        = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = ["benign", "malignant"]   # 0, 1


# ==============================
# 1. Load benign IDs from false.csv
# ==============================
def load_benign_ids(csv_path):
    # One column, maybe with or without header; keep only things that look like IMDxxx
    df = pd.read_csv(csv_path, header=None)
    ids_raw = df.iloc[:, 0].astype(str).str.strip()
    benign_ids = [x for x in ids_raw if x.startswith("IMD")]
    print(f"Loaded {len(benign_ids)} benign IDs from {csv_path}")
    return set(benign_ids)


# ==============================
# 2. Build (img_path, label) list from folder structure
# ==============================
def build_samples(images_root, benign_ids):
    """
    images_root:
        PH2Dataset/PH2 Dataset images/
            IMD003/
              IMD003_Dermoscopic_Image/
                IMD003_Dermoscopic_Image.bmp
    """
    images_root = Path(images_root)
    samples = []

    for lesion_dir in images_root.iterdir():
        if not lesion_dir.is_dir():
            continue

        imd_id = lesion_dir.name  # e.g. "IMD003"

        subdir = lesion_dir / f"{imd_id}_Dermoscopic_Image"
        img_path = subdir / f"{imd_id}.bmp"

        if not img_path.exists():
            print(f"Warning: expected image not found: {img_path}")
            continue

        if imd_id in benign_ids:
            label = 0  # benign
        else:
            label = 1  # malignant (everything not listed as benign)

        samples.append((str(img_path), label))

    print(
        f"Total samples: {len(samples)} "
        f"(benign={sum(1 for _, l in samples if l == 0)}, "
        f"malignant={sum(1 for _, l in samples if l == 1)})"
    )
    return samples


# ==============================
# 3. Custom Dataset
# ==============================
class RealLesionDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (img_path, label)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ==============================
# 4. Transforms
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ==============================
# 5. Training / eval helpers
# ==============================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def eval_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


# ==============================
# 6. Main training pipeline
# ==============================
def main():
    benign_ids = load_benign_ids(BENIGN_CSV_PATH)
    samples = build_samples(PH2_IMAGES_ROOT, benign_ids)

    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # 70/15/15 split with stratification
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
    )

    print(f"Train size: {len(train_paths)}, "
          f"Val size: {len(val_paths)}, "
          f"Test size: {len(test_paths)}")

    train_samples = list(zip(train_paths, train_labels))
    val_samples   = list(zip(val_paths, val_labels))
    test_samples  = list(zip(test_paths, test_labels))

    train_dataset = RealLesionDataset(train_samples, transform=train_transform)
    val_dataset   = RealLesionDataset(val_samples,   transform=eval_transform)
    test_dataset  = RealLesionDataset(test_samples,  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    # Model: ResNet18, 2-class output
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, criterion)
        elapsed = time.time() - start

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Time: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  🔹 New best model saved (val_acc={val_acc:.4f})")

    print("\nBest val accuracy:", best_val_acc)

    # ----- Test evaluation -----
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss, test_acc, y_true, y_pred = eval_model(model, test_loader, criterion)

    print(f"\n✅ Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()
