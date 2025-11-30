import os
import random

from pathlib import Path
from PIL import Image
from zipfile import ZipFile
from io import BytesIO
import torch
import torchvision.transforms as T

import file_paths

file_paths.BENIGN_SKETCHES_AUGMENTED_OUTPUT.mkdir(parents=True, exist_ok=True)

AUGS_PER_IMAGE = 30
OUT_SIZE = 256
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


augment = T.Compose([
    T.Resize((OUT_SIZE, OUT_SIZE)),


    T.RandomRotation(degrees=20),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomResizedCrop(
        size=OUT_SIZE,
        scale=(0.85, 1.0),
        ratio=(0.9, 1.1),
    ),


    T.ColorJitter(
        brightness=0.15,
        contrast=0.15,
    ),


])


def is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in IMG_EXTS



def augment_benign_sketches(input_dir, output_dir, augs_per_image=AUGS_PER_IMAGE):
    with ZipFile(file_paths.BENIGN_SKETCHES_ZIP, 'r') as zip_file:
        input_paths = [name for name in zip_file.namelist() if is_image_file(name)]


        for img_path in input_paths:
            try:
                image_bytes = BytesIO(zip_file.read(img_path))
                img = Image.open(image_bytes).convert("RGB")
            except Exception as e:
                print(f"Skipping {img_path} ({e})")
                continue

            stem = Path(img_path).stem


            base_out = img.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
            base_out_path = os.path.join(file_paths.BENIGN_SKETCHES_AUGMENTED_OUTPUT, f"{stem}_base.png")
            base_out.save(base_out_path)


            for i in range(augs_per_image):
                aug_img = augment(img)
                out_name = f"{stem}_aug_{i:02d}.png"
                out_path = os.path.join(file_paths.BENIGN_SKETCHES_AUGMENTED_OUTPUT, out_name)
                aug_img.save(out_path)



    


if __name__ == "__main__":
    augment_benign_sketches(file_paths.BENIGN_SKETCHES_ZIP, file_paths.BENIGN_SKETCHES_AUGMENTED_OUTPUT, AUGS_PER_IMAGE)
