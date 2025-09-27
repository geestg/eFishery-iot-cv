from ultralytics import YOLO

# Arahkan ke path yang benar (sesuai screenshot)
model = YOLO("D:/SEMESTER 5/TA 1/eFishery-iot-cv/train_ikan_mas_v2/weights/best.pt")

# Kalau best.pt tidak ada, gunakan last.pt
# model = YOLO("D:/SEMESTER 5/TA 1/eFishery-iot-cv/train_ikan_mas_v2/weights/last.pt")

results = model.predict(
    source="D:/SEMESTER 5/TA 1/eFishery-iot-cv/efishery_yolov8/videos_raw",
    show=True,
    save=True,
    project="D:/SEMESTER 5/TA 1/eFishery-iot-cv/output",
    name="prediksi_video",
    exist_ok=True
)
