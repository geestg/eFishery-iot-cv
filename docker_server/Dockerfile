FROM python:3.10-slim

WORKDIR /app
COPY . .

# Install pip terbaru agar kompatibel
RUN pip install --upgrade pip

# Install dependensi utama dari PyPI
RUN pip install --no-cache-dir flask ultralytics opencv-python-headless numpy

# Install PyTorch & torchvision CPU-only (gunakan repo PyTorch)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

VOLUME ["/app/data"]
EXPOSE 5000
CMD ["python", "app.py"]
