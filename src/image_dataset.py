import pandas as pd
from PIL import Image
from io import BytesIO
from zipfile import ZipFile
from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import Callable, Optional, Union
from . import config

class MelanomaImageDataset(Dataset):
    """
    PyTorch Dataset class to represent SFU dermascopic criteria melanoma image set.
    Implements minimum required functions for Dataset interface.
    """

    def __init__(self, metadata_csv_path: Path, images_zip_paths: tuple[Path, Path], transform: Optional[Callable] = None):
        """
        Initializes MelanomaImageDataset with metadata, image directories, and optional transform.
        :param metadata_csv_path: Path object to CSV file containing image metadata.
        :param images_zip_paths: Tuple containing paths object to 2 zip files containing all images.
        :param transform: Optional function to transform images to normalize for use with classifier model.
        """
        self.metadata_dataframe = pd.read_csv(metadata_csv_path)
        self.images_zip1 = ZipFile(images_zip_paths[0], 'r')
        self.images_zip2 = ZipFile(images_zip_paths[1], 'r')
        self.transform = transform

    def __len__(self) -> int:
        """
        Get size of dataset.
        :return: Size.
        """
        return len(self.metadata_dataframe)

    def __getitem__(self, index: int) -> tuple[Union[Image.Image, torch.Tensor], torch.Tensor]:
        """
        Get one data element from the dataset.
        :param index: Index of data element.
        :return: The element's image and label (melanoma vs not melanoma).
        """
        metadata_row = self.metadata_dataframe.iloc[index]
        image_relative_path_str = metadata_row[config.IMAGE_FILEPATH_COLUMN_NAME]
        image_bytes = self._get_images_bytes(image_relative_path_str)

        image = Image.open(image_bytes).convert("RGB")
        label = 1 if config.POSITIVE_CLASS in metadata_row[config.DIAGNOSIS_COLUMN_NAME] else 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def _get_images_bytes(self, image_relative_path_str: str):
        for zip_file in (self.images_zip1, self.images_zip2):
            try:
                image_bytes = zip_file.read(image_relative_path_str)
                return BytesIO(image_bytes)
            except KeyError:
                continue
        raise FileNotFoundError(f'File not found in image dataset: {image_relative_path_str}')