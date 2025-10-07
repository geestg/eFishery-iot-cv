from flask import Flask, request, jsonify, send_from_directory
import os, datetime, json, cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

# === konfigurasi model dan skala px→cm ===
MODEL_PATH = "model/best.pt"  # nanti mount model YOLO kamu ke container
PIXEL_PER_CM = 50              # contoh nilai kalibrasi (ubah sesuai hasil nyata)
model = YOLO(MODEL_PATH)

@app.route('/')
def index():
    return "<h2>🐟 eFishery Docker Server with Pixel-to-CM Processing</h2>"

# === Upload video ===
@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['file']
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    save_path = os.path.join(SAVE_DIR, filename)
    file.save(save_path)

    # Jalankan analisis setelah upload
    analysis_result = analyze_video(save_path)

    # Simpan hasil analisis ke file JSON
    analysis_filename = f"analysis_{timestamp}.json"
    with open(os.path.join(SAVE_DIR, analysis_filename), "w") as f:
        json.dump(analysis_result, f, indent=4)

    return jsonify({
        "status": "success",
        "message": "Video uploaded & analyzed successfully",
        "video_file": filename,
        "analysis_file": analysis_filename,
        "summary": analysis_result["summary"]
    })

# === Analisis ukuran ikan (Pixel → CM) ===
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fish_lengths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ambil setiap 10 frame agar tidak berat
        if frame_count % 10 != 0:
            continue

        results = model(frame, conf=0.5, device="cpu", verbose=False)
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            pixel_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            cm_length = round(pixel_length / PIXEL_PER_CM, 2)
            fish_lengths.append(cm_length)

    cap.release()

    summary = {
        "total_frames": frame_count,
        "detected_fish": len(fish_lengths),
        "avg_length_cm": round(np.mean(fish_lengths), 2) if fish_lengths else 0
    }

    return {"summary": summary, "lengths_cm": fish_lengths}

# === Upload log manual (opsional) ===
@app.route('/upload_data', methods=['POST'])
def upload_data():
    data = request.get_json()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"log_{timestamp}.json"
    save_path = os.path.join(SAVE_DIR, filename)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)
    return jsonify({
        "status": "success",
        "message": "Log saved successfully",
        "filename": filename
    })

# === List file & download ===
@app.route('/videos')
def list_videos():
    files = os.listdir(SAVE_DIR)
    vids = [f for f in files if f.endswith(('.avi','.mp4'))]
    return jsonify({"videos": vids})

@app.route('/logs')
def list_logs():
    files = os.listdir(SAVE_DIR)
    logs = [f for f in files if f.endswith('.json')]
    return jsonify({"logs": logs})

@app.route('/download/<path:fname>')
def download(fname):
    return send_from_directory(SAVE_DIR, fname, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
