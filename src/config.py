from .util import get_path

# file paths
IMAGE_DIRECTORY = get_path("data/image_data/images")
METADATA_FILE = get_path("data/image_data/meta/meta.csv")
TRAIN_INDICES_FILE = get_path("data/image_data/meta/train_indexes.csv")
VALIDATION_INDICES_FILE = get_path("data/image_data/meta/valid_indexes.csv")
TEST_INDICES_FILE = get_path("data/image_data/meta/test_indexes.csv")

# meta.csv values
DIAGNOSIS_COLUMN_NAME = "diagnosis"
IMAGE_FILEPATH_COLUMN_NAME = "derm"
POSITIVE_CLASS = "melanoma"
