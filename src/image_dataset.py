import pandas as pd
from PIL import Image
from io import BytesIO
from zipfile import ZipFile
from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
from . import config

class MelanomaImageDataset(Dataset):
    """
    PyTorch Dataset class to represent SFU dermascopic criteria melanoma image set.
    Implements minimum required functions for Dataset interface.
    """

    def __init__(self, metadata_csv_path: Path, images_zip_path: Path, transform: Optional[Callable] = None):
        """
        Initializes MelanomaImageDataset with metadata, image directories, and optional transform.
        :param metadata_csv_path: Path object to CSV file containing image metadata.
        :param images_zip_path: Path object to zip file containing all images.
        :param transform: Optional function to transform images to normalize for use with classifier model.
        """
        self.metadata_dataframe = pd.read_csv(metadata_csv_path)
        self.images_zip = ZipFile(images_zip_path, 'r')
        self.transform = transform

    def __len__(self) -> int:
        """
        Get size of dataset.
        :return: Size.
        """
        return len(self.metadata_dataframe)

    def __getitem__(self, index: int) -> Tuple[Union[Image.Image, torch.Tensor], torch.Tensor]:
        """
        Get one data element from the dataset.
        :param index: Index of data element.
        :return: The element's image and label (melanoma vs not melanoma).
        """
        metadata_row = self.metadata_dataframe.iloc[index]
        image_relative_path_str = metadata_row[config.IMAGE_FILEPATH_COLUMN_NAME]
        image_bytes = self.images_zip.read(image_relative_path_str)
        images_bytes_IO = BytesIO(image_bytes)

        image = Image.open(images_bytes_IO).convert("RGB")
        label = 1 if config.POSITIVE_CLASS in metadata_row[config.DIAGNOSIS_COLUMN_NAME] else 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)