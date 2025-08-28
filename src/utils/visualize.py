import cv2

COLORS = {}

def _get_color(i: int):
    if i not in COLORS:
        COLORS[i] = (0, 255, (i * 37) % 255)
    return COLORS[i]

def draw_boxes(image, boxes, scores=None, class_ids=None, class_names=None):
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = _get_color(class_ids[idx] if class_ids is not None else idx)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if class_ids is not None and class_names is not None:
            name = class_names[class_ids[idx]] if class_ids[idx] < len(class_names) else str(class_ids[idx])
            score = f" {scores[idx]:.2f}" if scores is not None else ""
            label = f"{name}{score}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return image
