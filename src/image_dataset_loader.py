from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from enum import Enum
from pathlib import Path
import csv
import string
from .image_dataset import MelanomaImageDataset
from . import config

class LoaderType(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3

class MelanomaImageDatasetLoader:
    def __init__(self, metadata_csv_path: Path, images_directory_str: string):
        print('Loading Melanoma Image Dataset')
        self.dataset = MelanomaImageDataset(
            metadata_csv_path = metadata_csv_path,
            images_directory_str= images_directory_str,
            transform=self._get_inception_v3_image_transform()
        )
        self.loader_map = {
            LoaderType.TRAINING: (config.TRAIN_INDICES_FILE, True),
            LoaderType.VALIDATION: (config.VALIDATION_INDICES_FILE, False),
            LoaderType.TEST: (config.TEST_INDICES_FILE, False)
        }

    def get_melanoma_image_dataset_loader(self, loader_type: LoaderType) -> DataLoader:
        """
        Gets data loader for any phase of AI pipeline.
        :param loader_type: The phase (TRAINING, VALIDATION, or TEST).
        :return: The data loader.
        """
        print(f'Partitioning Melanoma Image Dataset for {loader_type.name}')

        file_path, shuffle = self.loader_map[loader_type]
        indices = self._load_indices(file_path)
        loader_type_subset = Subset(self.dataset, indices)
        return DataLoader(loader_type_subset, batch_size=32, shuffle=shuffle)

    def _load_indices(self, csv_path: Path) -> list:
        """
        Converts indices of images to use for phase of AI pipeline (as suggested by SFU dataset) to list.
        :param csv_path: Path object of CSV file contain indices.
        :return: CSV file as a list of ints.
        """
        indices = []
        with open(csv_path) as file:
            reader = csv.reader(file)
            next(reader) # skip header
            for row in reader:
                indices.append(int(row[0]))

        return indices

    def _get_inception_v3_image_transform(self) -> transforms.Compose:
        """
        Gets transformation for use of images in Inception V3 model. The model requires 299x299 images.
        :return: A function performing the transformation.
        """
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])