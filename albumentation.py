import cv2
import albumentations as A
from pathlib import Path

# Folder asal & tujuan
input_dir = Path(r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\efishery_yolov8\dataset\images\train")
output_dir = input_dir.parent / "test_aug"   # hasil simpan di test_aug
output_dir.mkdir(parents=True, exist_ok=True)

# Definisi augmentasi
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.6),
    A.HueSaturationValue(p=0.5),
    A.GaussNoise(p=0.4),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), p=0.5)  # ✅ fix
])

# Loop semua gambar
for img_path in input_dir.glob("*.*"):  # bisa .jpg/.png
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # bikin beberapa variasi augmentasi
    for i in range(3):  # bikin 3 variasi per gambar
        augmented = transform(image=img)
        aug_img = augmented["image"]

        out_path = output_dir / f"aug_{i}_{img_path.name}"
        cv2.imwrite(str(out_path), aug_img)

print("✅ Augmentasi selesai")
print("📂 Hasil ada di:", output_dir)
