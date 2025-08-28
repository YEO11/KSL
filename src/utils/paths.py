from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ANNOTATED_DIR = DATA_DIR / "annotated"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
CKPT_DIR = MODELS_DIR / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"

for p in [DATA_DIR, RAW_DIR, ANNOTATED_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MODELS_DIR, PRETRAINED_DIR, CKPT_DIR, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
