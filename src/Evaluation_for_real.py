import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from .image_dataset_loader import MelanomaImageDatasetLoader, LoaderType
from .model import load_inception_v3
from . import file_paths
from .train import MODEL_SAVE_PATH  # reuse path from train.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["benign", "malignant"]


def _forward_inception(model, images: torch.Tensor) -> torch.Tensor:
    """
    Handle Inception outputs (InceptionOutputs, tuple, or plain logits).
    """
    outputs = model(images)
    if hasattr(outputs, "logits"):
        outputs = outputs.logits
    elif isinstance(outputs, tuple):
        outputs = outputs[0]
    return outputs


def evaluate_real_only_model():
    # Build real-image test loader
    loader = MelanomaImageDatasetLoader(
        metadata_csv_path=file_paths.METADATA_FILE,
        images_zip_paths=(file_paths.IMAGES_ZIP_PATH_1, file_paths.IMAGES_ZIP_PATH_2),
    )
    test_loader = loader.get_melanoma_image_dataset_loader(LoaderType.TEST)

    # loead model architecture with its best real-only weights
    model = load_inception_v3()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []  # probability of class 1 (malignant)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = _forward_inception(model, images)
            probs = F.softmax(logits, dim=1)[:, 1]  # p(malignant)
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # 3) Compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # malignant recall

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_prob)

    # printt
    print("\n========== REAL-ONLY MODEL EVALUATION ==========")
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


if __name__ == "__main__":
    evaluate_real_only_model()
