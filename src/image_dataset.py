import string
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
from .util import get_path
from .util import _get_project_root
from . import config

class MelanomaImageDataset(Dataset):
    """
    PyTorch Dataset class to represent SFU dermascopic criteria melanoma image set.
    Implements minimum required functions for Dataset interface.
    """

    def __init__(self, metadata_csv_path: Path, images_directory_str: string, transform: Optional[Callable] = None):
        """
        Initializes MelanomaImageDataset with metadata, image directories, and optional transform.
        :param metadata_csv_path: Path object to CSV file containing image metadata.
        :param images_directory_str: String of directory containing all images.
        :param transform: Optional function to transform images to normalize for use with classifier model.
        """
        self.metadata_dataframe = pd.read_csv(metadata_csv_path)
        self.images_directory_str = images_directory_str
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
        image_path_str = self.images_directory_str + metadata_row[config.IMAGE_FILEPATH_COLUMN_NAME]
        image_path = get_path(image_path_str)

        image = Image.open(image_path).convert("RGB")
        label = 1 if config.POSITIVE_CLASS in metadata_row[config.DIAGNOSIS_COLUMN_NAME] else 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)