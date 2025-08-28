from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class TrainConfig:
    project_root: str = "."
    dataset_root: str = "data"
    data_yaml_path: str = "data/data.yaml"
    names_txt: str = "data/names.txt"

    model: str = "yolov8n.pt"
    imgsz: int = 640
    epochs: int = 100
    batch: int = 16
    device: str = "0"        # or "cpu"
    workers: int = 8
    seed: int = 42

    save_dir: str = "runs/train"

@dataclass
class RealtimeConfig:
    weights: str = "models/checkpoints/best.pt"
    conf: float = 0.5
    iou: float = 0.5
    imgsz: int = 640
    device: str = "0"
    class_names_path: str = "data/names.txt"
    show_fps: bool = True

def load_config(path: str | Path | None = None) -> TrainConfig:
    if path is None:
        return TrainConfig()
    path = Path(path)
    if not path.exists():
        return TrainConfig()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    valid = {k: v for k, v in raw.items() if k in TrainConfig.__annotations__}
    return TrainConfig(**valid)
