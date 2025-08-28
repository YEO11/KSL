"""
Convert a simple JSON annotation into YOLO format.
Each JSON file (in a folder) should be:
{
  "image": "path/to/img.jpg",
  "width": 1280, "height": 720,
  "objects": [{"cls": "ㅅ", "bbox": [x1,y1,x2,y2]}, ...]
}
"""
from pathlib import Path
import json
from src.utils.jamo import NAME2ID

def xyxy_to_yolo(x1,y1,x2,y2,w,h):
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1), (y2-y1)
    return cx/w, cy/h, bw/w, bh/h

def convert_folder(json_dir):
    json_dir = Path(json_dir)
    for jf in json_dir.glob("*.json"):
        data = json.loads(jf.read_text(encoding="utf-8"))
        img_path = Path(data["image"])
        w, h = data["width"], data["height"]
        lines = []
        for obj in data.get("objects", []):
            cls = NAME2ID.get(obj["cls"])
            if cls is None:
                print(f"[WARN] Unknown class '{obj['cls']}' in {jf}"); continue
            x1,y1,x2,y2 = obj["bbox"]
            x,y,bw,bh = xyxy_to_yolo(x1,y1,x2,y2,w,h)
            lines.append(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
        label_path = img_path.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="utf-8")
        print("Wrote", label_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("json_dir", help="Folder with .json files")
    args = ap.parse_args()
    convert_folder(args.json_dir)
