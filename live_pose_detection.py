"""
Live Pose Detection using webcam video.

Uses YOLO11n-pose (same model as the main app).
Press 'q' to quit.
"""

import cv2
from ultralytics import YOLO

# Load the pose estimation model
model = YOLO("yolo11n-pose.pt")

# COCO keypoint order used by YOLO pose models
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# 0 = laptop webcam. Replace with "http://<mobile_ip>:8080/video"
# to use a mobile camera (e.g. via the DroidCam app).
VIDEO_SOURCE = 0

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

prev_time = cv2.getTickCount()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Run pose detection on the current frame
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    frame_count += 1

    # --- Console output: person count + keypoint coordinates ---
    person_count = (
        len(results[0].boxes)
        if results[0].boxes is not None
        else 0
    )

    print(f"\n--- Frame {frame_count} | Persons detected: {person_count} ---")

    if (
        results[0].keypoints is not None
        and len(results[0].keypoints.xy) > 0
    ):
        all_keypoints = results[0].keypoints.xy.cpu().numpy()

        for person_id, keypoints in enumerate(all_keypoints, start=1):
            print(f"  Person {person_id}:")
            for name, (x, y) in zip(KEYPOINT_NAMES, keypoints):
                print(f"    {name:>15}: ({x:7.1f}, {y:7.1f})")
    else:
        print("  No keypoints detected.")

    # Calculate and show FPS
    current_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f} | Persons: {person_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Live Pose Detection (press q to quit)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
