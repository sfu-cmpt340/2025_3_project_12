from .util import get_path


# real images
IMAGES_ZIP_PATH_1 = get_path("data/image_data/images1.zip")
IMAGES_ZIP_PATH_2 = get_path("data/image_data/images2.zip")
METADATA_FILE = get_path("data/image_data/meta/meta.csv")
TRAIN_INDICES_FILE = get_path("data/image_data/meta/train_indexes.csv")
VALIDATION_INDICES_FILE = get_path("data/image_data/meta/valid_indexes.csv")
TEST_INDICES_FILE = get_path("data/image_data/meta/test_indexes.csv")

# sketches
SKETCH_SPLITS_ZIP = get_path("sketch_splits.zip")  
SKETCH_DATA_ROOT  = get_path("sketch_splits") 
