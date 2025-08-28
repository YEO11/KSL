from pathlib import Path
from .jamo import JAMO_CLASSES

TEMPLATE = (
    "train: {train}\n"
    "val: {val}\n"
    "test: {test}\n\n"
    f"nc: {{nc}}\n"
    f"names: {{names}}\n"
)

def write_data_yaml(out_path, train_dir, val_dir, test_dir=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = TEMPLATE.format(
        train=Path(train_dir).as_posix(),
        val=Path(val_dir).as_posix(),
        test=Path(test_dir).as_posix() if test_dir else Path(val_dir).as_posix(),
        nc=len(JAMO_CLASSES),
        names=JAMO_CLASSES,
    )
    out_path.write_text(payload, encoding="utf-8")
    return out_path
