import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

DIAGNOSIS_COLUMN_NAME = "diagnosis"
CLASS_LABEL_COLUMN_NAME = "label"
POSITIVE_CLASS = "melanoma"

class MelanomaImageDataset(Dataset):

    def __init__(self, metadata_csv_path, images_dir, transform=None):
        self.metadata_dataframe = pd.read_csv(metadata_csv_path)
        self.images_dir = images_dir
        self.transform = transform

        # normalize
        self.metadata_dataframe[DIAGNOSIS_COLUMN_NAME] = (self.metadata_dataframe[DIAGNOSIS_COLUMN_NAME].fillna("").str.strip().str.lower())

        # add class label
        self.metadata_dataframe[CLASS_LABEL_COLUMN_NAME] = self.metadata_dataframe[DIAGNOSIS_COLUMN_NAME].apply(
            lambda d: 1 if POSITIVE_CLASS in d else 0
        )

    def __len__(self):
        return 0

    def __getitem__(self, index):
        return None