import cv2
import numpy as np
from ultralytics import YOLO

# Load model YOLOv8n (ikan mas)
model = YOLO("runs/detect/train_ikan_kolam/weights/best.pt")

# Buka kamera (0 = webcam, atau ganti path video)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ======== 1. Preprocessing ========

    # Convert ke grayscale lalu CLAHE (tingkatkan kontras di air keruh)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Denoising (kurangi noise di malam/keruh)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

    # Sedikit blur → hilangkan partikel air kecil
    preprocessed = cv2.GaussianBlur(denoised, (3,3), 0)

    # ======== 2. YOLO Detection ========
    results = model.predict(preprocessed, conf=0.5, imgsz=640)

    # Visualisasi hasil
    annotated = results[0].plot()

    cv2.imshow("Preprocessed Input", preprocessed)
    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
