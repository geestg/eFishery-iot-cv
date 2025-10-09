from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from ultralytics import YOLO
import cv2, os, datetime, json, numpy as np

app = Flask(__name__)

# === 1. Direktori utama ===
SAVE_DIR = "data"
UPLOAD_DIR = os.path.join(SAVE_DIR, "uploads")
PROCESSED_DIR = os.path.join(SAVE_DIR, "processed")
MODEL_PATH = "model/best.pt"
RESULT_FILE = os.path.join(SAVE_DIR, "results.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# === 2. Load model YOLO ===
print("📦 Memuat model YOLO...")
model = YOLO(MODEL_PATH)
print("✅ Model YOLO siap digunakan!")

# === 3. Kalibrasi sementara (akan diupdate nanti) ===
CM_PER_PIXEL = 0.05  # default dummy value

# === 4. Muat hasil lama jika ada ===
if os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "r") as f:
        results_data = json.load(f)
else:
    results_data = []

# === 5. Helper function untuk penamaan otomatis ===
def generate_video_name():
    count = len(results_data) + 1
    return f"mas{count}.mp4"

# === 6. ROUTE ===

# redirect langsung ke dashboard
@app.route("/")
def home():
    return redirect(url_for('dashboard'))

# upload video hasil endoscope
@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]

    # waktu realtime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # buat nama file unik
    custom_name = generate_video_name()
    filename = f"{custom_name}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    print(f"🎥 File diterima: {filename}")

    # Jalankan deteksi YOLO
    results = model.predict(
        source=filepath,
        conf=0.5,
        save=True,
        project=PROCESSED_DIR,
        name=os.path.splitext(custom_name)[0],
        verbose=False
    )

    # hitung jumlah ikan (jumlah box)
    fish_lengths = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box[:4]
            fish_lengths.append(float((x2 - x1) * CM_PER_PIXEL))

    num_fish = len(fish_lengths)
    avg_length = 0  # Belum diketahui (belum kalibrasi)
    processed_path = os.path.join(PROCESSED_DIR, os.path.splitext(custom_name)[0])

    # cari file hasil video
    video_output = None
    for f in os.listdir(processed_path):
        if f.endswith(".mp4"):
            video_output = f"/processed/{os.path.splitext(custom_name)[0]}/{f}"
            break

    entry = {
        "filename": filename,
        "num_fish": num_fish,
        "avg_length_cm": "Belum diketahui",
        "timestamp": timestamp,
        "video_path": video_output
    }

    results_data.append(entry)
    with open(RESULT_FILE, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"✅ Analisis selesai: {filename}, Jumlah ikan: {num_fish}")
    return jsonify(entry)

# agar video hasil bisa ditonton langsung
@app.route("/processed/<path:filename>")
def serve_processed(filename):
    return send_from_directory(PROCESSED_DIR, filename)

# dashboard utama
@app.route("/dashboard")
def dashboard():
    sorted_results = sorted(results_data, key=lambda x: x["timestamp"], reverse=True)
    return render_template("dashboard.html", results=sorted_results)

# === 7. Jalankan server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
