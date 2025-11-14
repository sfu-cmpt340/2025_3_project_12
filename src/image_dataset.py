import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import config

class MelanomaImageDataset(Dataset):
    def __init__(self, metadata_csv_path: Path, images_dir: Path, transform: Optional[Callable] = None):
        self.metadata_dataframe = pd.read_csv(metadata_csv_path)
        self.images_directory = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata_dataframe)

    def __getitem__(self, index: int) -> Tuple[Union[Image.Image, torch.Tensor], torch.Tensor]:
        metadata_row = self.metadata_dataframe.iloc[index]
        image_relative_path = metadata_row[config.IMAGE_FILEPATH_COLUMN_NAME]
        image_absolute_path = self.images_directory.joinpath(image_relative_path)

        image = Image.open(image_absolute_path).convert("RGB")
        label = 1 if config.POSITIVE_CLASS in metadata_row[config.DIAGNOSIS_COLUMN_NAME] else 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)