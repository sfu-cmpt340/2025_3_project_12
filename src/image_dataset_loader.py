from image_dataset import MelanomaImageDataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from enum import Enum
from pathlib import Path
import csv
import config

class LoaderType(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3

def get_loader(loader_type: LoaderType) -> DataLoader:
    dataset = MelanomaImageDataset(
        metadata_csv_path = config.METADATA_FILE,
        images_dir= config.IMAGE_DIRECTORY,
        transform=_get_inception_v3_image_transform()
    )

    loader_map = {
        LoaderType.TRAINING: (config.TRAIN_INDICES_FILE, True),
        LoaderType.VALIDATION: (config.VALIDATION_INDICES_FILE, False),
        LoaderType.TEST: (config.TEST_INDICES_FILE, False)
    }

    file_path, shuffle = loader_map[loader_type]
    indices = _load_indices(file_path)
    subset_dataset = Subset(dataset, indices)
    return DataLoader(subset_dataset, batch_size=32, shuffle=shuffle)

def _load_indices(csv_path: Path) -> list:
    indices = []
    with open(csv_path) as file:
        reader = csv.reader(file)
        for row in reader:
            indices.append(int(row[0]))

    return indices

def _get_inception_v3_image_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])