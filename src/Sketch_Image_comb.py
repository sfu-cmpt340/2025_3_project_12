import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

from . import file_paths
from .image_dataset_loader import MelanomaImageDatasetLoader, LoaderType
from .model import load_inception_v3



MODEL_SAVE_PATH = "sketches_and_images_inception_v3.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 0
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = ["benign", "malignant"]



def get_inception_v3_train_transform() -> transforms.Compose:

    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_inception_v3_eval_transform() -> transforms.Compose:

    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])



def make_sketch_datasets(sketch_root: str,
                         train_transform: transforms.Compose,
                         eval_transform: transforms.Compose):

    sketch_train_dir = Path(sketch_root) / "train"
    sketch_val_dir = Path(sketch_root) / "val"

    sketch_train_raw = datasets.ImageFolder(root=sketch_train_dir, transform=train_transform)
    sketch_val_raw = datasets.ImageFolder(root=sketch_val_dir, transform=eval_transform)

    print("Sketch classes:", sketch_train_raw.classes)

    benign_like = {"benign", "false", "noncancer", "non_cancer", "negative"}
    idx_to_binary = {}
    for idx, cls_name in enumerate(sketch_train_raw.classes):
        if cls_name.lower() in benign_like:
            idx_to_binary[idx] = 0
        else:
            idx_to_binary[idx] = 1
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
    sketch_val = SketchWrapper(sketch_val_raw, idx_to_binary)
    return sketch_train, sketch_val



def _forward_inception(model: nn.Module, images: torch.Tensor) -> torch.Tensor:

    outputs = model(images)

    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return outputs


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

            outputs = _forward_inception(model, images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return running_loss / total, running_correct / total, np.array(all_labels), np.array(all_preds)



def train_combined_model():


    sketch_train, sketch_val = make_sketch_datasets(
        file_paths.SKETCH_DATA_ROOT,
        train_transform=get_inception_v3_train_transform(),
        eval_transform=get_inception_v3_eval_transform(),
    )


    melanoma_loader = MelanomaImageDatasetLoader(
        metadata_csv_path=file_paths.METADATA_FILE,
        images_zip_paths=(file_paths.IMAGES_ZIP_PATH_1, file_paths.IMAGES_ZIP_PATH_2),
    )


    real_train_loader = melanoma_loader.get_melanoma_image_dataset_loader(LoaderType.TRAINING)
    real_val_loader = melanoma_loader.get_melanoma_image_dataset_loader(LoaderType.VALIDATION)
    real_test_loader = melanoma_loader.get_melanoma_image_dataset_loader(LoaderType.TEST)


    real_train_ds = real_train_loader.dataset
    real_val_ds = real_val_loader.dataset
    real_test_ds = real_test_loader.dataset

    combined_train_ds = ConcatDataset([sketch_train, real_train_ds])
    combined_val_ds = ConcatDataset([sketch_val, real_val_ds])

    print(f"Sketch train size:  {len(sketch_train)}")
    print(f"Real train size:    {len(real_train_ds)}")
    print(f"Combined train size:{len(combined_train_ds)}")
    print(f"Sketch val size:    {len(sketch_val)}")
    print(f"Real val size:      {len(real_val_ds)}")
    print(f"Combined val size:  {len(combined_val_ds)}")
    print(f"Real test size:     {len(real_test_ds)}")


    train_loader = DataLoader(
        combined_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(combined_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    real_test_loader = DataLoader(real_test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS)


    model = load_inception_v3(num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, criterion)
        elapsed = time.time() - start

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
              f"Time: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" New best combined Inception V3 model saved (val_acc={val_acc:.4f})")

    print("\nBest combined val accuracy:", best_val_acc)


    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    test_loss, test_acc, y_true, y_pred = eval_model(model, real_test_loader, criterion)

    print(f"\n Combined Inception V3 model test on REAL images only:")
    print(f"   Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))



def main():
    train_combined_model()


if __name__ == "__main__":
    main()
