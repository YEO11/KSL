"""
Train YOLOv8 on Korean fingerspelling dataset.
pip install ultralytics opencv-python pyyaml
"""
from ultralytics import YOLO
from src.config.config import load_config
from src.utils.paths import TRAIN_DIR, VAL_DIR, TEST_DIR, RUNS_DIR
from src.utils.seed import set_seed
from src.utils.data_yaml import write_data_yaml
from src.utils.jamo import save_names_txt
import torch
import os
import yaml  # ← nc 읽기 위해 추가

print(f"🔷 Is Cuda Available? == {torch.cuda.is_available()}")
print(f"🔷 {torch.cuda.get_device_name(0)}")

def check_dataset(images_dir, labels_dir, nc):
    """
    이미지/라벨 쌍 확인 + 라벨 내용 체크
    """
    img_files = set(os.listdir(images_dir))
    label_files = set(os.listdir(labels_dir))

    # 1. 이미지와 라벨 매칭 확인
    missing_labels = [f for f in img_files if f.replace(".jpg", ".txt") not in label_files]
    missing_images = [f for f in label_files if f.replace(".txt", ".jpg") not in img_files]

    if missing_labels:
        print("⚠️ 이미지에 대응 라벨 없는 파일:", missing_labels)
    if missing_images:
        print("⚠️ 라벨에 대응 이미지 없는 파일:", missing_images)

    # 2. 라벨 내용 체크
    for lbl in label_files:
        path = os.path.join(labels_dir, lbl)
        with open(path) as f:
            lines = f.readlines()
            for l in lines:
                parts = l.strip().split()
                if len(parts) != 5:
                    print("⚠️ 잘못된 라벨 형식:", lbl, l)
                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= nc:
                        print("⚠️ 클래스 id 범위 이상:", lbl, l)
                    coords = list(map(float, parts[1:]))
                    if not all(0.0 <= x <= 1.0 for x in coords):
                        print("⚠️ 좌표 범위 이상:", lbl, l)
                except:
                    print("⚠️ 라벨 파싱 오류:", lbl, l)

def get_nc_from_yaml(data_yaml_path):
    """data.yaml에서 nc 읽어오기"""
    with open(data_yaml_path) as f:
        data_dict = yaml.safe_load(f)
    return data_dict['nc']

def main(cfg_path: str | None = None):
    cfg = load_config(cfg_path)
    set_seed(cfg.seed)

    # Prepare data.yaml and names.txt
    write_data_yaml(cfg.data_yaml_path, TRAIN_DIR / "images", VAL_DIR / "images", TEST_DIR / "images")
    save_names_txt(cfg.names_txt)

    # 🔍 data.yaml에서 nc 읽기
    nc = get_nc_from_yaml(cfg.data_yaml_path)

    # 🔍 데이터셋 검증
    print("🔎 Checking train dataset...")
    check_dataset(TRAIN_DIR / "images", TRAIN_DIR / "labels", nc=nc)
    print("🔎 Checking val dataset...")
    check_dataset(VAL_DIR / "images", VAL_DIR / "labels", nc=nc)

    model = YOLO(cfg.model)
    results = model.train(
        data=cfg.data_yaml_path,
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        device=cfg.device,
        workers=0,  # Windows에서는 0~2 추천
        project=str(RUNS_DIR),
        name="train-jamo",
        seed=cfg.seed,
        cos_lr=True,
        amp=False,
        val=True,
        plots=True
    )

    print(results)
    print(results.metrics)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=None, help="Path to YAML config (optional)")
    args = ap.parse_args()
    main(args.cfg)
