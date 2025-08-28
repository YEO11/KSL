import os
import shutil
import random
from pathlib import Path

# -------------------------
# 설정
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
ANNOTATED_DIR = DATA_DIR / "annotated"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

SPLIT_RATIO = 0.8  # 80% train, 20% val

# -------------------------
# 함수
# -------------------------
def ensure_dirs(base_dir: Path):
    for sub in ["images", "labels"]:
        (base_dir / sub).mkdir(parents=True, exist_ok=True)

def split_dataset():
    ensure_dirs(TRAIN_DIR)
    ensure_dirs(VAL_DIR)

    classes = [d.name for d in ANNOTATED_DIR.iterdir() if d.is_dir()]
    print("Classes:", classes)

    for cls in classes:
        label_files = list((ANNOTATED_DIR / cls).glob("*.txt"))
        random.shuffle(label_files)

        split_idx = int(len(label_files) * SPLIT_RATIO)
        train_labels = label_files[:split_idx]
        val_labels = label_files[split_idx:]

        for phase, files in [("train", train_labels), ("val", val_labels)]:
            for lbl in files:
                img_name = lbl.stem + ".jpg"
                img_path = RAW_DIR / cls / img_name
                if not img_path.exists():
                    print(f"⚠️ 이미지 없음: {img_path}")
                    continue

                # 새 파일 이름 (클래스 접두어 붙임)
                new_img_name = f"{cls}_{img_name}"
                new_lbl_name = f"{cls}_{lbl.name}"

                out_img = DATA_DIR / phase / "images" / new_img_name
                out_lbl = DATA_DIR / phase / "labels" / new_lbl_name

                shutil.copy2(img_path, out_img)
                shutil.copy2(lbl, out_lbl)

        print(f"[{cls}] → Train: {len(train_labels)}, Val: {len(val_labels)}")

if __name__ == "__main__":
    split_dataset()
