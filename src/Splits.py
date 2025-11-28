import os
import random
from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent.parent
BENIGN_SRC = BASE_DIR / "data" / "image_data" / "sketches_false_aug"
CANCER_SRC = BASE_DIR / "data" / "image_data" / "aug_sketch"


OUT_ROOT = BASE_DIR / "sketch_splits"


TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


MODE = "copy"


SEED = 42
random.seed(SEED)



IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    folder = Path(folder)
    return [p for p in folder.glob("*") if p.suffix.lower() in IMG_EXTS]


def split_list(items, train_ratio, val_ratio, test_ratio):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    n = len(items)
    random.shuffle(items)

    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    n_test  = n - n_train - n_val

    train_items = items[:n_train]
    val_items   = items[n_train:n_train + n_val]
    test_items  = items[n_train + n_val:]

    return train_items, val_items, test_items


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def transfer_file(src, dst, mode="copy"):
    ensure_dir(os.path.dirname(dst))
    if mode == "move":
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def make_splits():

    benign_paths = list_images(BENIGN_SRC)
    cancer_paths = list_images(CANCER_SRC)



    b_train, b_val, b_test = split_list(benign_paths, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    c_train, c_val, c_test = split_list(cancer_paths, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)



    splits = {
        "train": {"false": b_train, "cancer": c_train},
        "val":   {"false": b_val,   "cancer": c_val},
        "test":  {"false": b_test,  "cancer": c_test},
    }

    for split_name, class_dict in splits.items():
        for cls_name, paths in class_dict.items():
            out_dir = Path(OUT_ROOT) / split_name / cls_name
            ensure_dir(out_dir)
            for p in paths:
                dst = out_dir / p.name
                transfer_file(str(p), str(dst), mode=MODE)



if __name__ == "__main__":
    make_splits()
