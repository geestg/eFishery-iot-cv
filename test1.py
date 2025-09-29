from pathlib import Path
from ultralytics import YOLO

# Tentukan folder weight relatif ke file ini
base_dir = Path(__file__).resolve().parent
weights_dir = base_dir / "train_ikan_mas_v2" / "weights"

# Path file best dan last
best_pt = weights_dir / "best.pt"
last_pt = weights_dir / "last.pt"

# Pilih weight yang tersedia
if best_pt.exists():
    weights_path = best_pt
    print(f"[INFO] Load model dari: {weights_path}")
elif last_pt.exists():
    weights_path = last_pt
    print(f"[INFO] Load model dari: {weights_path}")
else:
    raise FileNotFoundError(f"Tidak ada weights di {weights_dir}")

# Load YOLO model
model = YOLO(str(weights_path))

# Path ke video
video_path = r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\efishery_yolov8\videos_raw\mas01.mp4"

# Jalankan prediksi pada video
results = model.predict(
    source=video_path,
    show=True,        # tampilkan hasil deteksi di jendela
    save=True,        # simpan hasil ke runs/detect/predict
    conf=0.5          # confidence threshold (atur sesuai kebutuhan)
)

print(f"[INFO] Hasil prediksi disimpan di folder runs/detect/predict")
