import cv2
import albumentations as A
from pathlib import Path

# Folder asal & tujuan
input_dir = Path(r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\efishery_yolov8\dataset\images\val")
output_dir = input_dir.parent / "train_aug"   # hasil simpan di train_aug
output_dir.mkdir(parents=True, exist_ok=True)

# Definisi augmentasi (fix RandomResizedCrop pakai tuple)
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.6),
    A.HueSaturationValue(p=0.5),
    A.GaussNoise(p=0.4),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.4),
    A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), p=0.5),  # ✅ fix error
    A.RandomFog(p=0.2),       # versi default (tanpa parameter extra)
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.RandomShadow(p=0.2)     # versi default
])

# Loop semua gambar
for img_path in input_dir.glob("*.jpg"):  # bisa diganti ke *.png kalau dataset png
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # hanya 1 variasi per gambar
    augmented = transform(image=img)
    aug_img = augmented["image"]

    out_path = output_dir / f"aug_{img_path.name}"
    cv2.imwrite(str(out_path), aug_img)

print("✅ Augmentasi selesai (1 variasi per gambar)")
print("📂 Hasil ada di:", output_dir)
