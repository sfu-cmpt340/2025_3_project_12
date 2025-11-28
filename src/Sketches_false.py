import os
import random
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms as T


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "image_data" / "sketches_false" # Directory containing benign sketch images (should be unziped if you want to run this)
OUTPUT_DIR = BASE_DIR / "data" / "image_data" / "sketches_false_aug" # Where the false ugmented images will be saved (create if not exist)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUGS_PER_IMAGE = 30
OUT_SIZE = 256
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
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


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS



def augment_benign_sketches(input_dir, output_dir, augs_per_image=AUGS_PER_IMAGE):
    input_paths = [p for p in Path(input_dir).glob("*") if is_image_file(p)]


    for img_path in input_paths:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_path} ({e})")
            continue

        stem = img_path.stem


        base_out = img.resize((OUT_SIZE, OUT_SIZE), Image.BILINEAR)
        base_out_path = os.path.join(OUTPUT_DIR, f"{stem}_base.png")
        base_out.save(base_out_path)


        for i in range(augs_per_image):
            aug_img = augment(img)
            out_name = f"{stem}_aug_{i:02d}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            aug_img.save(out_path)



    


if __name__ == "__main__":
    augment_benign_sketches(INPUT_DIR, OUTPUT_DIR, AUGS_PER_IMAGE)
