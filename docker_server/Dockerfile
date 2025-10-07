# Gunakan image Python yang ringan
FROM python:3.10-slim

# Install dependency sistem untuk OpenCV & Flask (kompatibel Debian Trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja di dalam container
WORKDIR /app

# Copy semua file ke container
COPY . .

# Perbaikan instalasi pip untuk mencegah error hash mismatch
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Buat folder penyimpanan data hasil video
RUN mkdir -p /app/data

# Buka port 5000 untuk web server Flask
EXPOSE 5000

# Jadikan folder /app/data sebagai volume (agar data bisa disimpan di host)
VOLUME ["/app/data"]

# Jalankan Flask app
CMD ["python", "app.py"]
