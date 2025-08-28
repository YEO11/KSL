"""
Realtime YOLOv8 webcam demo using custom weights and RealtimeConfig.
Press 'q' to quit. Press 'f' to toggle FPS overlay.
"""
import cv2
import time
import torch
from ultralytics import YOLO
from src.config.config import RealtimeConfig
from src.utils.jamo import JAMO_CLASSES  # or load from config if needed
import numpy as np

print(f"🔷 Is Cuda Available? == {torch.cuda.is_available()}")
print(f"🔷 {torch.cuda.get_device_name(0)}")

# ---------- COLOR ASSIGNMENT ----------
def generate_colors(num_classes):
    np.random.seed(42)  # 항상 같은 색이 나오도록 고정
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_classes)]
    return colors

class_names = JAMO_CLASSES
colors = generate_colors(len(class_names))

# ---------- LOAD CONFIG ----------
cfg = RealtimeConfig()

# Load YOLO model
model = YOLO(cfg.weights)

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("realtime-jamo", cv2.WINDOW_NORMAL)

if not cap.isOpened():
    raise RuntimeError("Cannot access webcam")

show_fps = cfg.show_fps
last_time = time.time()

frame_count = 0
skip_frames = 2   # YOLO는 2프레임마다 한 번만 실행 (즉, FPS 절반 수준)

last_results = []  # 마지막 detection 결과 저장

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam")
        break

    frame_count += 1

    # ---------- YOLO INFERENCE (skip 적용) ----------
    if frame_count % skip_frames == 0:
        results = model(frame, device=cfg.device, conf=cfg.conf, iou=cfg.iou)
        last_results = results  # 최신 결과 저장
    else:
        results = last_results

    annotated = frame.copy()

    for r in results:
        if hasattr(r, "boxes"):
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, ci in zip(boxes, confs, cls_ids):
                if conf < cfg.conf:
                    continue
                name = class_names[ci] if ci < len(class_names) else str(ci)
                color = colors[ci % len(colors)]  # 클래스별 고유 색상 선택

                # 사각형 + 텍스트
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    annotated,
                    f"{name} {conf:.2f}",
                    (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

    # ---------- FPS ----------
    if show_fps:
        now = time.time()
        fps = 1.0 / max(1e-6, now - last_time)
        last_time = now
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    # ---------- DISPLAY ----------
    cv2.imshow("realtime-jamo", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("f"):
        show_fps = not show_fps

cap.release()
cv2.destroyAllWindows()
