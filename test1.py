from ultralytics import YOLO

# Load model hasil training
model = YOLO(r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\train_ikan_mas_v1\weights\best.pt")

# Uji model ke video
results = model.predict(
    source=r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\efishery_yolov8\videos_raw\mas.mp4",  # video input
    conf=0.5,               # confidence threshold
    save=True,              # simpan hasil video
    show=True,              # tampilkan langsung
    device='cpu',           # AMD Radeon tanpa CUDA
    project=r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\efishery_yolov8\videos_raw",  # lokasi hasil disimpan
    name="hasil_deteksi"    # nama folder hasil deteksi
)

print("✅ Deteksi selesai!")
print("📂 Hasil disimpan di: D:\\SEMESTER 5\\TA 1\\eFishery-iot-cv\\efishery_yolov8\\videos_raw\\hasil_deteksi")
