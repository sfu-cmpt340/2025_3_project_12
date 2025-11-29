import pandas as pd
from PIL import Image
from io import BytesIO
from zipfile import ZipFile
from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import Callable, Optional, Union
from . import file_paths


# meta.csv values
DIAGNOSIS_COLUMN_NAME = "diagnosis"
IMAGE_FILEPATH_COLUMN_NAME = "derm"
POSITIVE_CLASS = "melanoma"

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

        # Maps to perform case-insensitive searches (to handle case mismatches between meta.csv and actual data folders)
        self.images_zip1_case_map = {name.lower(): name for name in self.images_zip1.namelist()}
        self.images_zip2_case_map = {name.lower(): name for name in self.images_zip2.namelist()}

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
        # Image
        image_metadata = self.metadata_dataframe.iloc[index]
        image = self._get_image(image_metadata)

        # Image transformation
        if self.transform:
            image = self.transform(image)

        # Image label
        label = 1 if POSITIVE_CLASS in image_metadata[DIAGNOSIS_COLUMN_NAME] else 0

        return image, torch.tensor(label, dtype=torch.long)

    def _get_image(self, image_metadata: pd.Series) -> Image.Image:
        image_relative_path_str = image_metadata[IMAGE_FILEPATH_COLUMN_NAME]

        for zip_file, case_map in [(self.images_zip1, self.images_zip1_case_map),
                                   (self.images_zip2, self.images_zip2_case_map)]:
            correct_case_name = case_map.get(image_relative_path_str.lower())
            if correct_case_name:
                image_bytes = BytesIO(zip_file.read(correct_case_name))
                return Image.open(image_bytes).convert("RGB")

        raise FileNotFoundError(f'File not found in image dataset: {image_relative_path_str}')