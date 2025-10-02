from pathlib import Path

# ganti path ke folder labels anda
labels_dir = Path(r"D:\SEMESTER 5\TA 1\eFishery-iot-cv\efishery_yolov8\dataset\labels")

for txt_file in labels_dir.rglob("*.txt"):  # scan train dan val
    lines = txt_file.read_text().strip().splitlines()
    fixed_lines = []
    for line in lines:
        parts = line.split()
        if parts:
            parts[0] = "0"  # ganti semua class index ke 0
            fixed_lines.append(" ".join(parts))
    txt_file.write_text("\n".join(fixed_lines))
    print(f"Fixed: {txt_file}")
