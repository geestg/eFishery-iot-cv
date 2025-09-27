from ultralytics import YOLO

# Load model hasil training (best.pt)
model = YOLO("D:/SEMESTER 5/TA 1/eFishery-iot-cv/efishery_yolov8/runs/detect/train_ikan_mas_v2/weights/best.pt")

# Path ke video input
video_path = "D:/SEMESTER 5/TA 1/eFishery-iot-cv/efishery_yolov8/videos_raw/video1.mp4"

# Jalankan prediksi pada video
results = model.predict(
    source=video_path,     # bisa video atau folder video
    conf=0.5,              # threshold confidence
    save=True,             # simpan hasil deteksi ke file
    save_txt=False,        # kalau mau simpan bounding box ke txt set True
    show=True              # tampilkan jendela video dengan bounding box
)

print("✅ Deteksi selesai, hasil tersimpan di:", results[0].save_dir)
