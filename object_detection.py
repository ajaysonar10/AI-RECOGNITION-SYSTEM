"""
BAS-AI • Object Detection Module
--------------------------------
Detects ALL 80 COCO classes (person, chair, laptop, bottle, cup,
cell phone, keyboard, book, mouse, ...) — not just persons.

Why "sirf person" dikhta tha:
1. yolo11n-pose.pt = POSE model hai, wo SIRF person detect karta hai.
   Yahan hamesha DETECTION model use karo (yolo11n.pt / yolo11s.pt).
2. conf (confidence) threshold zyada ho to chhote/uncertain objects
   drop ho jaate hain — isliye ye configurable rakha hai.
3. classes=[0] filter lagane par bhi sirf person milta hai.
"""

import cv2
from ultralytics import YOLO


# ============================================================
# MODEL LOAD
# ============================================================

# Full 80-class COCO DETECTION model.
# Better accuracy chahiye to "yolo11s.pt" ya "yolo11m.pt" likho.
MODEL_PATH = "yolo11n.pt"

model = YOLO(MODEL_PATH)


# ============================================================
# DETECTION SETTINGS
# ============================================================

DEFAULT_CONF = 0.30    # confidence threshold (kam = zyada objects milenge)
DEFAULT_IOU = 0.45     # NMS overlap threshold (duplicate boxes hataata hai)
DEFAULT_IMGSZ = 640    # image size (bada rakho to small objects better milte hain)

# classes=None  -> ALL 80 COCO classes detect honge  (correct default)
# classes=[0]   -> sirf person                       (yahi galti hoti hai)
DEFAULT_CLASSES = None


def detect_objects(
    frame,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    imgsz=DEFAULT_IMGSZ,
    classes=DEFAULT_CLASSES,
):
    """
    Frame ke andar ke SAARE objects detect karta hai.

    Args:
        frame:   BGR image (numpy array, jaisa cv2 deta hai)
        conf:    confidence threshold (0-1)
        iou:     NMS IoU threshold (0-1)
        imgsz:   inference image size (px)
        classes: None = sabhi 80 classes, ya [0, 56, ...] specific list

    Returns:
        annotated_frame: boxes + labels + count ke saath frame
        detections: list of dicts with class_name, confidence, box
    """

    results = model.predict(
        frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=classes,
        verbose=False,
    )

    result = results[0]

    # Detection result ko frame par draw karo (sabhi classes)
    annotated_frame = result.plot()

    # ------------------------------------------------------------
    # Detection details nikaalo (app mein dikhane ke liye)
    # ------------------------------------------------------------

    detections = []

    if result.boxes is not None:

        names = result.names  # {class_id: class_name}

        for box in result.boxes:

            class_id = int(box.cls[0])

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": float(box.conf[0]),
                    "box": [float(v) for v in box.xyxy[0]],
                }
            )

    # ------------------------------------------------------------
    # Frame par summary text (kitne objects mile)
    # ------------------------------------------------------------

    summary = f"Objects: {len(detections)}"

    cv2.putText(
        annotated_frame,
        summary,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    return annotated_frame, detections


# ============================================================
# STANDALONE TEST — is file ko direct run karo:
#     python object_detection.py
# Webcam khulegi aur SAARE objects boxes ke saath dikhenge.
# ============================================================

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("X Camera could not be opened.")
        raise SystemExit(1)

    print("Object detection started - press 'q' to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("X Frame not received from camera.")
            break

        annotated_frame, detections = detect_objects(frame)

        for det in detections:
            print(
                f"  {det['class_name']:<15} "
                f"{det['confidence'] * 100:5.1f}%  "
                f"box={det['box']}"
            )

        cv2.imshow(
            "BAS-AI Object Detection (q = quit)",
            annotated_frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
