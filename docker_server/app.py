from flask import Flask, request, jsonify, render_template, send_from_directory
from ultralytics import YOLO
import cv2, os, datetime, json, numpy as np

app = Flask(__name__)

SAVE_DIR = "data"
UPLOAD_DIR = os.path.join(SAVE_DIR, "uploads")
PROCESSED_DIR = os.path.join(SAVE_DIR, "processed")
MODEL_PATH = "model/best.pt"
RESULT_FILE = os.path.join(SAVE_DIR, "results.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
CM_PER_PIXEL = 0.05  # ubah sesuai hasil kalibrasi kamu

# Muat hasil lama
if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "r") as f:
        results_data = json.load(f)
else:
    results_data = []

@app.route("/")
def home():
    return "<h3>Server YOLO aktif ✅ — buka <a href='/dashboard'>/dashboard</a></h3>"

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    # Jalankan YOLO deteksi dan simpan hasil video
    results = model.predict(
        source=filepath, conf=0.5, save=True,
        project=PROCESSED_DIR, name=timestamp
    )

    fish_lengths = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box[:4]
            fish_lengths.append(float((x2 - x1) * CM_PER_PIXEL))

    avg_length = np.mean(fish_lengths) if fish_lengths else 0
    num_fish = len(fish_lengths)
    processed_path = os.path.join(PROCESSED_DIR, timestamp)
    
    # cari file hasil (biasanya .mp4 di folder YOLO)
    video_output = None
    for f in os.listdir(processed_path):
        if f.endswith(".mp4"):
            video_output = f"/processed/{timestamp}/{f}"
            break

    entry = {
        "filename": filename,
        "num_fish": num_fish,
        "avg_length_cm": round(float(avg_length), 2),
        "timestamp": timestamp,
        "video_path": video_output
    }

    results_data.append(entry)
    with open(RESULT_FILE, "w") as f:
        json.dump(results_data, f, indent=2)

    return jsonify(entry)

@app.route("/processed/<path:filename>")
def serve_processed(filename):
    """Agar video hasil YOLO bisa diakses dari browser"""
    return send_from_directory(PROCESSED_DIR, filename)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", results=results_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
