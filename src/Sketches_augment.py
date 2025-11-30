import os
import random

from pathlib import Path
from PIL import Image
from zipfile import ZipFile
from io import BytesIO
import torch
import torchvision.transforms as T

import file_paths


file_paths.MALIGNANT_SKETCHES_AUGMENTED_OUTPUT.mkdir(parents=True, exist_ok=True)


AUGS_PER_IMAGE = 20

random.seed(42)
torch.manual_seed(42)


augment = T.Compose([
    T.ConvertImageDtype(torch.float32),

    # Geometric transforms
    T.RandomRotation(degrees=20),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomResizedCrop(
        size=(256, 256),
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
    ),


    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
    ),


    T.ToPILImage(),
])


def load_image_as_rgb(path):
    img = Image.open(path).convert("RGB")
    return img


def augment_sketches():
    with ZipFile(file_paths.MALIGNANT_SKETCHES_ZIP, 'r') as zip_file:
        input_paths = [
            name for name in zip_file.namelist()
            if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]


        for img_path in input_paths:
            image_bytes = BytesIO(zip_file.read(img_path))
            img = Image.open(image_bytes).convert("RGB")
            stem = Path(img_path).stem  # filename without extension

            for i in range(AUGS_PER_IMAGE):
                # Convert PIL -> tensor in [0,1] range
                img_tensor = T.functional.pil_to_tensor(img)
                aug_img = augment(img_tensor)

                out_name = f"{stem}_aug_{i:02d}.png"
                out_path = os.path.join(file_paths.MALIGNANT_SKETCHES_AUGMENTED_OUTPUT, out_name)
                aug_img.save(out_path)



    


if __name__ == "__main__":
    augment_sketches()

