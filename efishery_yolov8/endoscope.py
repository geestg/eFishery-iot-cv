from ultralytics import YOLO
import cv2
import time
import requests
import os

# === 1. Load model YOLO hasil training ikan mas ===
model = YOLO(r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\train_ikan_mas_v1\weights\best.pt")

# === 2. Pilih kamera endoscope ===
cap = cv2.VideoCapture(0)  # ganti ke 1 jika endoscope tidak di index 0
if not cap.isOpened():
    print("❌ Kamera tidak dapat dibuka!")
    exit()
print("✅ Kamera aktif. Tekan 'q' untuk keluar.")

# === 3. Siapkan penyimpanan hasil video ===
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = 'hasil_endoscope.avi'
out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

# === 4. Jalankan deteksi YOLO ===
prev_time = 0
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
    boxes = results[0].boxes
    jumlah = len(boxes)

    annotated = results[0].plot()
    cv2.putText(annotated, f"Ikan terdeteksi: {jumlah}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(annotated)
    cv2.imshow("Deteksi Ikan Mas (Endoscope)", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Program dihentikan oleh user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Video hasil tersimpan sebagai {output_file}")

# === 5. Kirim video ke server kamu sendiri ===
server_url = "http://127.0.0.1:5000/upload"  # ubah ke localhost server kamu

try:
    with open(output_file, "rb") as f:
        print(f"📤 Mengirim file {output_file} ke server {server_url} ...")
        response = requests.post(server_url, files={"file": f})
        print("✅ Respons server:", response.text)
except Exception as e:
    print("❌ Gagal mengirim file:", e)
