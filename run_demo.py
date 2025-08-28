"""
Convenience runner to (1) ensure data.yaml and names.txt, (2) run a quick predict if weights exist.
"""
from ultralytics import YOLO
from src.utils.data_yaml import write_data_yaml
from src.utils.jamo import save_names_txt
from src.utils.paths import TRAIN_DIR, VAL_DIR, TEST_DIR
from pathlib import Path

def main():
    write_data_yaml("data/data.yaml", TRAIN_DIR / "images", VAL_DIR / "images", TEST_DIR / "images")
    save_names_txt("data/names.txt")
    weights = Path("models/checkpoints/best.pt")
    if weights.exists():
        YOLO(str(weights)).predict(source=TEST_DIR / "images", save=True)
        print("Ran a quick prediction on test images.")
    else:
        print("Prepared data files. Train your model to create weights first.")

if __name__ == "__main__":
    main()
