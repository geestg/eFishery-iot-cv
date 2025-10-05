from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data=r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\efishery_yolov8\data.yaml",
    epochs=150,
    imgsz=640,
    batch=8,               # lebih kecil, aman untuk CPU
    workers=0,
    device='cpu',          # AMD Radeon tanpa CUDA → CPU mode
    project=r"D:\SEMESTER 5\TA 1\eFishery-iot-cv",
    name="train_ikan_mas_v1",
    optimizer="AdamW",
    lr0=0.001,
    patience=30,
    augment=True
)

print("✅ Training selesai")
print("📂 Hasil tersimpan di:", results.save_dir)
