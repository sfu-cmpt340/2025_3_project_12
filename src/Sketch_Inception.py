import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import file_paths
from image_dataset_loader import MelanomaImageDatasetLoader, LoaderType
from model import load_inception_v3

# ------
# Set Up like Model_Sketches.py from Tim

BASE_DIR = Path(__file__).resolve().parent.parent
SKETCH_ROOT = BASE_DIR / "sketch_splits" 

BATCH_SIZE  = 32
NUM_EPOCHS  = 10
LR          = 1e-4
NUM_WORKERS = 0
SEED        = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = ["benign", "malignant"]
MODEL_SAVE_PATH = "sketch_inception_v3_best.pth"


# ------
# Data transforms for Inception V3

def get_inception_v3_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_inception_v3_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


#------
# Make dataset map and training functions
def make_sketch_datasets(sketch_root: Path,
                         train_transform: transforms.Compose,
                         eval_transform: transforms.Compose):
    train_dir = sketch_root / "train"
    val_dir   = sketch_root / "val"
    test_dir  = sketch_root / "test"

    sketch_train_raw = datasets.ImageFolder(root=train_dir, transform=train_transform)
    sketch_val_raw   = datasets.ImageFolder(root=val_dir,   transform=eval_transform)
    sketch_test_raw  = datasets.ImageFolder(root=test_dir,  transform=eval_transform)

    print("Sketch classes:", sketch_train_raw.classes)

    benign_like = {"benign", "false", "noncancer", "non_cancer", "negative"}
    idx_to_binary = {}
    for idx, cls_name in enumerate(sketch_train_raw.classes):
        if cls_name.lower() in benign_like:
            idx_to_binary[idx] = 0   # benign
        else:
            idx_to_binary[idx] = 1   # malignant
    print("Sketch index mapping:", idx_to_binary)

    class SketchWrapper(Dataset):
        def __init__(self, base_dataset, idx_map):
            self.base = base_dataset
            self.idx_map = idx_map

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            img, orig_label = self.base[i]
            new_label = self.idx_map[int(orig_label)]
            return img, torch.tensor(new_label, dtype=torch.long)

    sketch_train = SketchWrapper(sketch_train_raw, idx_to_binary)
    sketch_val   = SketchWrapper(sketch_val_raw,   idx_to_binary)
    sketch_test  = SketchWrapper(sketch_test_raw,  idx_to_binary)
    return sketch_train, sketch_val, sketch_test


def _forward_inception(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    out = model(images)
    if isinstance(out, tuple):
        out = out[0]
    return out


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = _forward_inception(model, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


def eval_model_collect(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = _forward_inception(model, images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # prob of class 1 (malignant)
            _, preds = torch.max(outputs, 1)

            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    loss = running_loss / total
    acc  = running_correct / total
    return loss, acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics_block(title: str, y_true, y_pred, y_prob):
    acc        = accuracy_score(y_true, y_pred)
    precision  = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # malignant recall

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1    = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_prob)

    print(f"\n========== {title} ==========")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Sensitivity:   {sensitivity:.4f}")
    print(f"Specificity:   {specificity:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"AUROC:         {auroc:.4f}")
    print("=============================================\n")

    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report (benign=0, malignant=1):")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


def train_and_evaluate_sketch_only():
    # Build sketch dataset
    sketch_train, sketch_val, sketch_test = make_sketch_datasets(
        SKETCH_ROOT,
        train_transform=get_inception_v3_train_transform(),
        eval_transform=get_inception_v3_eval_transform(),
    )

    train_loader = DataLoader(sketch_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(sketch_val,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    sketch_test_loader = DataLoader(sketch_test, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=NUM_WORKERS)

    #Use Inception V3 model
    model = load_inception_v3(num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _, _ = eval_model_collect(model, val_loader, criterion)
        elapsed = time.time() - start

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Time: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" New best sketch-only Inception V3 saved (val_acc={val_acc:.4f})")

    print("\nBest sketch-only val accuracy:", best_val_acc)

    # Load best model for evaluation
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)

    # Evaluate on sketch test only
    sketch_test_loss, sketch_test_acc, y_true_s, y_pred_s, y_prob_s = eval_model_collect(
        model, sketch_test_loader, criterion
    )
    print(f"\nSketch-only Inception model test on SKETCH images:")
    print(f"   Test Loss: {sketch_test_loss:.4f}, Test Accuracy: {sketch_test_acc:.4f}")
    print_metrics_block("SKETCH-ONLY MODEL ON SKETCH TEST", y_true_s, y_pred_s, y_prob_s)

    # #vaulate on real test
    melanoma_loader = MelanomaImageDatasetLoader(
        metadata_csv_path=file_paths.METADATA_FILE,
        images_zip_paths=(file_paths.IMAGES_ZIP_PATH_1, file_paths.IMAGES_ZIP_PATH_2),
    )
    real_test_loader = melanoma_loader.get_melanoma_image_dataset_loader(LoaderType.TEST)

    real_test_loss, real_test_acc, y_true_r, y_pred_r, y_prob_r = eval_model_collect(
        model, real_test_loader, criterion
    )
    print(f"\nSketch-only Inception model test on REAL images:")
    print(f"   Test Loss: {real_test_loss:.4f}, Test Accuracy: {real_test_acc:.4f}")
    print_metrics_block("SKETCH-ONLY MODEL ON REAL TEST", y_true_r, y_pred_r, y_prob_r)


def main():
    print("Using device:", device)
    train_and_evaluate_sketch_only()


if __name__ == "__main__":
    main()
