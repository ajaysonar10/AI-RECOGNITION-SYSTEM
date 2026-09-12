from ultralytics import YOLO
import cv2


# YOLO model load
model = YOLO("yolo11n.pt")


def detect_objects(frame):

    # Object detection
    results = model(frame, verbose=False)

    # Detection result ko frame par draw karo
    annotated_frame = results[0].plot()

    return annotated_frame