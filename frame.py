import cv2
import os

# Folder input (video)
video_folder = r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\efishery_yolov8\videos_raw"

# Folder output (frame hasil ekstraksi)
output_folder = r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\efishery_yolov8\frames"
os.makedirs(output_folder, exist_ok=True)

# Interval frame yang mau disimpan (misalnya setiap 10 frame = sekitar 0.3 detik kalau video 30fps)
frame_interval = 10

# Loop semua file video di folder
for filename in os.listdir(video_folder):
    if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mov"):
        video_path = os.path.join(video_folder, filename)
        cap = cv2.VideoCapture(video_path)

        # Ambil nama video tanpa ekstensi
        video_name = os.path.splitext(filename)[0]

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Nama file frame
                frame_filename = f"{video_name}_frame_{saved_count}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"✅ {saved_count} frames disimpan dari {filename}")

print("🎉 Selesai ekstraksi semua video!")
