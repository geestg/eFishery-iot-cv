from ultralytics import YOLO

# Load model YOLOv8n (nano → ringan, cocok untuk realtime)
model = YOLO("yolov8n.pt")

# Training
results = model.train(
    data="D:/SEMESTER 5/TA 1/eFishery-iot-cv/efishery_yolov8/data.yaml",  # path ke data.yaml
    epochs=150,              # epoch lebih lama untuk dataset kecil
    imgsz=640,               # resolusi optimal YOLO
    batch=16,                # sesuaikan dengan VRAM
    patience=30,             # early stopping
    workers=0,               # Windows → set 0
    project="D:/SEMESTER 5/TA 1/eFishery-iot-cv", 
    name="train_ikan_kolam", # folder hasil training

    # Hyperparameter
    optimizer="AdamW",        # lebih akurat untuk dataset kecil
    lr0=0.001,                # initial learning rate
    lrf=0.01,                 # final learning rate fraction
    momentum=0.937,           
    weight_decay=0.0005,      
    warmup_epochs=3.0,        

    # Augmentasi khusus kondisi kolam
    augment=True,             
    hsv_h=0.015,              # hue
    hsv_s=0.7,                # saturation
    hsv_v=0.5,                # brightness (siang/malam)
    degrees=5.0,              # rotasi kecil
    translate=0.1,            # translasi
    scale=0.5,                # zoom ±50% (harus float tunggal)
    shear=2.0,                # sedikit shear
    perspective=0.0,          
    flipud=0.0,               # ikan tidak jungkir balik
    fliplr=0.5,               # ikan bisa kanan/kiri
    mosaic=0.5,               # mosaik sedang
    mixup=0.1,                # campur gambar
    copy_paste=0.1            # tempel ikan lain di frame
)

print("✅ Training selesai")
print("📂 Hasil tersimpan di:", results.save_dir)
