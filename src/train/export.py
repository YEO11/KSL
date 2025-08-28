# Export trained weights to various formats (onnx, torchscript, etc.).
from ultralytics import YOLO

def export(weights: str, fmt: str = "onnx"):
    model = YOLO(weights)
    model.export(format=fmt)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="Path to trained .pt (e.g., runs/train/train-jamo/weights/best.pt)")
    ap.add_argument("--format", default="onnx",
                    choices=["onnx","torchscript","openvino","engine","coreml","tflite"])
    args = ap.parse_args()
    export(args.weights, args.format)
