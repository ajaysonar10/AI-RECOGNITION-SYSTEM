from ultralytics import YOLO


model = YOLO("yolo11n-pose.pt")


def detect_pose(frame):

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    keypoints = None

    if results[0].keypoints is not None:

        if len(results[0].keypoints.xy) > 0:

            keypoints = (
                results[0]
                .keypoints
                .xy[0]
                .cpu()
                .numpy()
            )

    return annotated_frame, keypoints