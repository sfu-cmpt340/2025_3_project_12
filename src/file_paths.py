import os

from pathlib import Path

def _get_data_directory() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data"

REAL_IMAGE_DIRECTORY = _get_data_directory() / "image_data"
SKETCH_DIRECTORY = _get_data_directory() / "sketch_data"

# real images
IMAGES_ZIP_PATH_1 = REAL_IMAGE_DIRECTORY / "images1.zip"
IMAGES_ZIP_PATH_2 = REAL_IMAGE_DIRECTORY / "images2.zip"
METADATA_FILE = REAL_IMAGE_DIRECTORY / "meta/meta.csv"
TRAIN_INDICES_FILE = REAL_IMAGE_DIRECTORY / "meta/train_indexes.csv"
VALIDATION_INDICES_FILE = REAL_IMAGE_DIRECTORY / "meta/valid_indexes.csv"
TEST_INDICES_FILE = REAL_IMAGE_DIRECTORY / "meta/test_indexes.csv"

# sketches
MALIGNANT_SKETCHES_ZIP = SKETCH_DIRECTORY / "malignant_sketches.zip"
BENIGN_SKETCHES_ZIP = SKETCH_DIRECTORY / "benign_sketches.zip"
MALIGNANT_SKETCHES_AUGMENTED_OUTPUT = SKETCH_DIRECTORY / "malignant_sketches_augmented"
BENIGN_SKETCHES_AUGMENTED_OUTPUT = SKETCH_DIRECTORY / "benign_sketches_augmented"
SKETCH_SPLITS_ZIP = SKETCH_DIRECTORY / "sketch_splits.zip"
SKETCH_SPLITS = SKETCH_DIRECTORY / "sketch_splits"
