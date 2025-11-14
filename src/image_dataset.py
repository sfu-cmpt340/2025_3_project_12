import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

DIAGNOSIS_COLUMN_NAME = "diagnosis"
POSITIVE_CLASS = "melanoma"
IMAGE_FILEPATH_COLUMN_NAME = "derm"

class MelanomaImageDataset(Dataset):

    def __init__(self, metadata_csv_path, images_dir, transform=None):
        self.metadata_dataframe = pd.read_csv(metadata_csv_path)
        self.images_directory = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata_dataframe)

    def __getitem__(self, index):
        metadata_row = self.metadata_dataframe.iloc[index]
        image_relative_path = metadata_row[IMAGE_FILEPATH_COLUMN_NAME]
        image_absolute_path = os.path.join(self.images_directory, image_relative_path)

        image = Image.open(image_absolute_path).convert("RGB")
        label = 1 if POSITIVE_CLASS in metadata_row[DIAGNOSIS_COLUMN_NAME] else 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)