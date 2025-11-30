from pathlib import Path

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

import file_paths
from real_real_image_dataset_loader import MelanomaImageDatasetLoader, LoaderType
from model import load_inception_v3
from Sketch_Image_comb import MODEL_SAVE_PATH  # use save path from training combined model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["benign", "malignant"]

# ------
# Build real-image test loader 

def evaluate_combined_model():
    melanoma_loader = MelanomaImageDatasetLoader(
        metadata_csv_path=file_paths.METADATA_FILE,
        images_zip_paths=(file_paths.IMAGES_ZIP_PATH_1, file_paths.IMAGES_ZIP_PATH_2),
    )
    real_test_loader = melanoma_loader.get_melanoma_image_dataset_loader(
        LoaderType.TEST
    )
    # ------
    # Load model architecture with its best combined weights

    model = load_inception_v3(num_classes=2)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []  # prob of class 1 (malignant)

    with torch.no_grad():
        for images, labels in real_test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # ----
    # Calculate metrics for report 
    acc = accuracy_score(y_true, y_pred) #accurary
    precision = precision_score(y_true, y_pred, zero_division=0) #precision for malignant
    sensitivity = recall_score(y_true, y_pred, zero_division=0)  # recall for malignant

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred) 
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    #f1 score and auroc
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_prob)

    # Print evaluations 
    print("\n========== SKETCH COMBINATION MODEL EVALUATION ==========")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Sensitivity:   {sensitivity:.4f}")
    print(f"Specificity:   {specificity:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"AUROC:         {auroc:.4f}")
    print("=============================================\n")

    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    evaluate_combined_model()
