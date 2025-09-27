from ultralytics import YOLO

# Load model YOLOv8s (lebih besar dari nano, biar recall lebih baik)
model = YOLO("yolov8s.pt")

# Train model
results = model.train(
    data="D:/SEMESTER 5/TA 1/eFishery-iot-cv/efishery_yolov8n/data.yaml",  # path ke data.yaml
    epochs=50,          # kurangi epoch biar tidak overfitting
    imgsz=640,          # ukuran gambar default optimal (bisa coba 640/736)
    batch=8,            # batch size
    patience=10,        # early stopping lebih cepat
    workers=0,          # aman di Windows
    project="D:/SEMESTER 5/TA 1/eFishery-iot-cv",  # folder utama hasil
    name="train_ikan_mas_v2",  # subfolder hasil training
    augment=True,       # aktifkan augmentasi data
    fl_gamma=2.0,       # focal loss → bantu kalau dataset imbalance
    lr0=0.01,           # initial learning rate
    lrf=0.01,           # final learning rate
    weight_decay=0.0005 # regularisasi untuk cegah overfitting
)

print("✅ Training selesai")
print("📂 Hasil tersimpan di:", results.save_dir)
