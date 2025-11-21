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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms, datasets
from PIL import Image



SKETCH_ROOT = r"C:/Users/User/Desktop/matlab/Project/sketch_splits"


PH2_IMAGES_ROOT = r"C:/Users/User/Desktop/matlab/Project/PH2Dataset/PH2 Dataset images"
BENIGN_CSV_PATH = r"C:/Users/User/Desktop/matlab/Project/PH2Dataset/false.csv"


MODEL_SAVE_PATH = "combined_resnet18_best.pth"

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_EPOCHS  = 10
LR          = 1e-3
NUM_WORKERS = 0
SEED        = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = ["benign", "malignant"]

# ==============================
# 1. Sketch datasets via ImageFolder (train / val)
# ==============================
def make_sketch_datasets(sketch_root, train_transform, eval_transform):
    sketch_train_dir = Path(sketch_root) / "train"
    sketch_val_dir   = Path(sketch_root) / "val"

    # Raw ImageFolder datasets
    sketch_train_raw = datasets.ImageFolder(root=sketch_train_dir, transform=train_transform)
    sketch_val_raw   = datasets.ImageFolder(root=sketch_val_dir,   transform=eval_transform)

    print("Sketch classes:", sketch_train_raw.classes)   # e.g. ['benign','cancer'] or ['cancer','false']

    # Map original class indices to {0=benign, 1=malignant}
    benign_like = {"benign", "false", "noncancer", "non_cancer", "negative"}
    idx_to_binary = {}
    for idx, cls_name in enumerate(sketch_train_raw.classes):
        if cls_name.lower() in benign_like:
            idx_to_binary[idx] = 0
        else:
            idx_to_binary[idx] = 1
    print("Sketch index mapping:", idx_to_binary)  # e.g. {0:1, 1:0} etc.

    class SketchWrapper(Dataset):
        def __init__(self, base_dataset, idx_map):
            self.base = base_dataset
            self.idx_map = idx_map

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            img, orig_label = self.base[i]
            new_label = self.idx_map[int(orig_label)]
            return img, new_label

    sketch_train = SketchWrapper(sketch_train_raw, idx_to_binary)
    sketch_val   = SketchWrapper(sketch_val_raw, idx_to_binary)
    return sketch_train, sketch_val



def load_benign_ids(csv_path):
    df = pd.read_csv(csv_path, header=None)
    ids = df.iloc[:, 0].astype(str).str.strip().tolist()
    benign_ids = [x for x in ids if x.startswith("IMD")]
    print(f"Loaded {len(benign_ids)} benign IDs from {csv_path}")
    return set(benign_ids)


def build_real_samples(images_root, benign_ids):

    root = Path(images_root)
    samples = []

    for lesion_dir in root.iterdir():
        if not lesion_dir.is_dir():
            continue
        imd_id = lesion_dir.name
        img_dir = lesion_dir / f"{imd_id}_Dermoscopic_Image"
        img_path = img_dir / f"{imd_id}.bmp"

        if not img_path.exists():
            print(f"Warning: missing image: {img_path}")
            continue

        label = 0 if imd_id in benign_ids else 1
        samples.append((str(img_path), label))

    print(
        f"Real samples: {len(samples)} "
        f"(benign={sum(1 for _, l in samples if l == 0)}, "
        f"malignant={sum(1 for _, l in samples if l == 1)})"
    )
    return samples


class RealLesionDataset(Dataset):
    def __init__(self, samples, transform=None):
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


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


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

    return running_loss / total, running_correct / total


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

    return running_loss / total, running_correct / total, np.array(all_labels), np.array(all_preds)


def main():
    sketch_train, sketch_val = make_sketch_datasets(SKETCH_ROOT, train_transform, eval_transform)

    benign_ids = load_benign_ids(BENIGN_CSV_PATH)
    real_samples = build_real_samples(PH2_IMAGES_ROOT, benign_ids)

    paths = [s[0] for s in real_samples]
    labels = [s[1] for s in real_samples]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
    )

    print(f"Real train size: {len(train_paths)}, val size: {len(val_paths)}, test size: {len(test_paths)}")

    real_train_ds = RealLesionDataset(list(zip(train_paths, train_labels)), transform=train_transform)
    real_val_ds   = RealLesionDataset(list(zip(val_paths, val_labels)),   transform=eval_transform)
    real_test_ds  = RealLesionDataset(list(zip(test_paths, test_labels)),  transform=eval_transform)

    combined_train_ds = ConcatDataset([sketch_train, real_train_ds])
    combined_val_ds   = ConcatDataset([sketch_val,   real_val_ds])

    print(f"Combined train size: {len(combined_train_ds)}, Combined val size: {len(combined_val_ds)}")

    train_loader = DataLoader(combined_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(combined_val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    real_test_loader = DataLoader(real_test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS)

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
            print(f"  🔹 New best combined model saved (val_acc={val_acc:.4f})")

    print("\nBest combined val accuracy:", best_val_acc)

    # ---- Final test on REAL images only ----
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss, test_acc, y_true, y_pred = eval_model(model, real_test_loader, criterion)

    print(f"\n Combined model test on REAL images only:")
    print(f"   Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    main()