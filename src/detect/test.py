"""
Batch inference on a folder of images or a video file.
Usage: python -m src.detect.test --weights models/checkpoints/best.pt --source data/test/images
"""
from ultralytics import YOLO

def run(weights: str, source: str, conf: float = 0.5, iou: float = 0.5, imgsz: int = 640, device: str = "0"):
    model = YOLO(weights)
    results = model.predict(source=source, conf=conf, iou=iou, imgsz=imgsz, device=device, save=True, save_txt=True)
    print(results)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/checkpoints/best.pt")
    ap.add_argument("--source", default="data/test/images")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    run(args.weights, args.source, args.conf, args.iou, args.imgsz, args.device)
