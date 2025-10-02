from ultralytics import YOLO

# Load model hasil training
model = YOLO(r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\train_ikan_kolam\weights\best.pt")

# Path video input
video_path = r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\efishery_yolov8\videos_raw\mas02.mp4"

# Prediksi pada video
results = model.predict(
    source=video_path,
    conf=0.25,   # turunkan threshold biar lebih banyak ikan terdeteksi
    iou=0.45,    # intersection-over-union threshold
    save=True,   # simpan hasil video
    show=True,   # tampilkan langsung
    project=r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\train_ikan_kolam",  # folder hasil
    name="predict_video"  # subfolder hasil
)

print("✅ Proses selesai. Hasil tersimpan di:")
print(r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\train_ikan_kolam\predict_video")

# Hitung jumlah ikan per frame
print("\n=== Jumlah ikan terdeteksi per frame ===")
for i, result in enumerate(results):
    boxes = result.boxes  # bounding box hasil deteksi
    num_fish = len(boxes) if boxes is not None else 0
    print(f"Frame {i+1}: {num_fish} ikan")
