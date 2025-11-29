import os
import random
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as T


BASE_DIR = Path(__file__).resolve().parent.parent  

INPUT_DIR = BASE_DIR / "data" / "image_data" / "sketches_orig" # Directory containing original sketch images (should be unziped if you want to run this)
OUTPUT_DIR = BASE_DIR / "data" / "image_data" / "aug_sketch" # Directory to save augmented images (create if not exist)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


AUGS_PER_IMAGE = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    input_paths = [
        p for p in Path(INPUT_DIR).glob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]



    for img_path in input_paths:
        img = load_image_as_rgb(img_path)
        stem = img_path.stem  # filename without extension

        for i in range(AUGS_PER_IMAGE):
            # Convert PIL -> tensor in [0,1] range
            img_tensor = T.functional.pil_to_tensor(img)
            aug_img = augment(img_tensor)

            out_name = f"{stem}_aug_{i:02d}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            aug_img.save(out_path)



    


if __name__ == "__main__":
    augment_sketches()

