from pathlib import Path
from image_dataset import MelanomaImageDataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import csv
from enum import Enum
from util import get_path

class LoaderType(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3

class ImageDatasetLoader:
    def get_loader(self, loader_type: LoaderType) -> DataLoader:
        dataset = MelanomaImageDataset(
            metadata_csv_path = get_path("data/image_data/meta/meta.csv"),
            images_dir= get_path("data/image_data/images/"),
            transform=self._get_inception_v3_image_transform()
        )

        loader_map = {
            LoaderType.TRAINING: ("train_indexes.csv", True),
            LoaderType.VALIDATION: ("valid_indexes.csv", False),
            LoaderType.TEST: ("test_indexes.csv", False)
        }

        file_name, shuffle = loader_map[loader_type]
        indices = self._load_indices(get_path(f"data/image_data/meta/{file_name}"))
        subset_dataset = Subset(dataset, indices)
        return DataLoader(subset_dataset, batch_size=32, shuffle=shuffle)

    @staticmethod
    def _load_indices(csv_path: Path) -> list:
        indices = []
        with open(csv_path) as file:
            reader = csv.reader(file)
            for row in reader:
                indices.append(int(row[0]))

        return indices

    @staticmethod
    def _get_inception_v3_image_transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])