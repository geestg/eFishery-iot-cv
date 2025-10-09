from ultralytics import YOLO
import cv2
import time
import requests
import os
import datetime
import numpy as np
from sort import Sort  # tracker untuk melacak ikan antar-frame

# === 1. Load model YOLO hasil training ikan mas ===
MODEL_PATH = r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\train_ikan_mas_v1\weights\best.pt"
model = YOLO(MODEL_PATH)

# === 2. Pilih kamera endoscope ===
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("❌ Kamera tidak dapat dibuka!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("✅ Kamera aktif. Tekan 'q' untuk berhenti.")

# === 3. Siapkan penyimpanan video dengan nama rapi ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = f"mas_{timestamp}.mp4"   # format baru
output_file = video_name

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

# === 4. Tracker (SORT) ===
tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)
unique_ids = set()
prev_time = 0

# === 5. Jalankan deteksi YOLO + tracking ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Tidak ada frame yang terbaca!")
        break

    current_time = time.time()
    if current_time - prev_time < 0.1:
        continue
    prev_time = current_time

    results = model(frame, conf=0.5, device='cpu', verbose=False)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        detections.append([x1, y1, x2, y2, conf])

    if len(detections) > 0:
        tracked_objects = tracker.update(np.array(detections))
        for x1, y1, x2, y2, obj_id in tracked_objects:
            obj_id = int(obj_id)
            unique_ids.add(obj_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{obj_id}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Ikan unik: {len(unique_ids)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    out.write(frame)
    cv2.imshow("🎥 Deteksi Ikan Mas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Program dihentikan oleh user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Video hasil tersimpan sebagai {output_file}")
print(f"🐟 Total ikan unik: {len(unique_ids)}")

# === 6. Kirim video hasil ke server Flask ===
server_url = "http://localhost:5000/upload"

try:
    with open(output_file, "rb") as f:
        print(f"📤 Mengirim {output_file} ke server {server_url} ...")
        response = requests.post(server_url, files={"file": f})
        print("✅ Respons server:", response.text)
except Exception as e:
    print("❌ Gagal mengirim file:", e)
