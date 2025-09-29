import os

# Path ke folder labels (ganti sesuai struktur dataset kamu)
base_label_dir = r"D:/SEMESTER 5/TA 1/eFishery-iot-cv/efishery_yolov8/dataset/labels"

splits = ["train", "val"]

for split in splits:
    folder = os.path.join(base_label_dir, split)
    if not os.path.exists(folder):
        print(f"❌ Folder tidak ditemukan: {folder}")
        continue

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            new_lines = []
            with open(path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "0"  # force semua class_id jadi 0
                        new_lines.append(" ".join(parts) + "\n")
            with open(path, "w") as f:
                f.writelines(new_lines)

    print(f"✅ Semua label di {folder} sudah diperbaiki ke class 0 (ikan_mas)")
