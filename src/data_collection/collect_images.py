"""
Webcam collector for building a per-class image dataset quickly.
Press 'c' to capture, 'n' to next class, 'b' to previous, 'q' to quit.
Images saved to data/raw/<class_name>/timestamp.jpg
"""
import os
import time
import cv2
from src.utils.jamo import JAMO_CLASSES
from src.utils.paths import RAW_DIR  # RAW_DIR can still be a Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def put_korean_text(img, text, pos=(10,30), font_size=30, color=(0,255,0)):
    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 한글 폰트 경로 (적절히 변경 필요)
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows 예시, Mac은 AppleGothic.ttf 등
    font = ImageFont.truetype(font_path, font_size)
    
    draw.text(pos, text, font=font, fill=color)
    
    # 다시 OpenCV 이미지로 변환
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)  # works like Path.mkdir(parents=True, exist_ok=True)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    idx = 0
    print("🔷 Press 'c'=capture, 'n'=next, 'b'=prev, 'q'=quit")
    while True:
        ok, frame = cap.read()
        if not ok: break
        name = JAMO_CLASSES[idx]
        view = frame.copy()
        view = put_korean_text(view, f"Class: {name} ({idx+1}/{len(JAMO_CLASSES)})", pos=(10,30))
        cv2.imshow("collector", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            out_dir = os.path.join(RAW_DIR, name)
            os.makedirs(out_dir, exist_ok=True)

            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}.jpg"
            fp = os.path.join(out_dir, filename)

            # Encode first, then write
            success, buffer = cv2.imencode('.jpg', frame)
            if success:
                with open(fp, 'wb') as f:
                    f.write(buffer)
                print("Saved", fp)
            else:
                print("Failed to encode image")

        elif k == ord('n'):
            idx = (idx + 1) % len(JAMO_CLASSES)
        elif k == ord('b'):
            idx = (idx - 1) % len(JAMO_CLASSES)
        elif k == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
