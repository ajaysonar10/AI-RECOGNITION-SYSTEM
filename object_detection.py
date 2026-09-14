"""
BAS-AI • Object Detection Module
--------------------------------
Detects ALL 80 COCO classes (person, chair, laptop, bottle, cup,
cell phone, keyboard, book, mouse, ...) — not just persons.

Why "only persons" appeared earlier:
1. yolo11n-pose.pt is a POSE model — it ONLY detects persons.
   Always use a DETECTION model here (yolo11n.pt / yolo11s.pt).
2. A higher conf (confidence) threshold drops small/uncertain
   objects — that is why it is kept configurable.
3. Applying a classes=[0] filter also yields only persons.
"""

import cv2
from ultralytics import YOLO


# ============================================================
# MODEL LOAD
# ============================================================

# Full 80-class COCO DETECTION model.
# For better accuracy, use "yolo11s.pt" or "yolo11m.pt".
MODEL_PATH = "yolo11n.pt"

model = YOLO(MODEL_PATH)


# ============================================================
# DETECTION SETTINGS
# ============================================================

DEFAULT_CONF = 0.30    # confidence threshold (lower = more objects found)
DEFAULT_IOU = 0.45     # NMS overlap threshold (removes duplicate boxes)
DEFAULT_IMGSZ = 640    # image size (higher values find small objects better)

# classes=None  -> ALL 80 COCO classes are detected  (correct default)
# classes=[0]   -> only person                       (the common mistake)
DEFAULT_CLASSES = None


def detect_objects(
    frame,
    conf=DEFAULT_CONF,
    iou=DEFAULT_IOU,
    imgsz=DEFAULT_IMGSZ,
    classes=DEFAULT_CLASSES,
):
    """
    Detects ALL objects inside the frame.

    Args:
        frame:   BGR image (numpy array, as given by cv2)
        conf:    confidence threshold (0-1)
        iou:     NMS IoU threshold (0-1)
        imgsz:   inference image size (px)
        classes: None = all 80 classes, or [0, 56, ...] specific list

    Returns:
        annotated_frame: frame with boxes + labels + count
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

    # Draw the detection result on the frame (all classes)
    annotated_frame = result.plot()

    # ------------------------------------------------------------
    # Extract detection details (to show in the app)
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
    # Summary text on the frame (how many objects were found)
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
# STANDALONE TEST — run this file directly:
#     python object_detection.py
# The webcam opens and ALL objects are shown with boxes.
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
