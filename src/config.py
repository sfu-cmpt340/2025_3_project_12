from .util import get_path

# image file paths
IMAGES_ZIP_PATH_1 = get_path("data/image_data/images1.zip")
IMAGES_ZIP_PATH_2 = get_path("data/image_data/images2.zip")
METADATA_FILE = get_path("data/image_data/meta/meta.csv")
TRAIN_INDICES_FILE = get_path("data/image_data/meta/train_indexes.csv")
VALIDATION_INDICES_FILE = get_path("data/image_data/meta/valid_indexes.csv")
TEST_INDICES_FILE = get_path("data/image_data/meta/test_indexes.csv")

# meta.csv values
DIAGNOSIS_COLUMN_NAME = "diagnosis"
IMAGE_FILEPATH_COLUMN_NAME = "derm"
POSITIVE_CLASS = "melanoma"

# sketch file paths
SKETCH_DATA_ROOT = get_path("sketch_splits/")
