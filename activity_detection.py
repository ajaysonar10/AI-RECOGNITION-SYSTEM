import math
from collections import deque


# Store previous body positions for Walking detection
motion_history = deque(maxlen=15)


def calculate_angle(a, b, c):
    """
    Calculates the angle between points A-B-C.
    B = middle/joint point
    """

    try:
        ba = (
            a[0] - b[0],
            a[1] - b[1]
        )

        bc = (
            c[0] - b[0],
            c[1] - b[1]
        )

        magnitude_ba = math.sqrt(
            ba[0] ** 2 + ba[1] ** 2
        )

        magnitude_bc = math.sqrt(
            bc[0] ** 2 + bc[1] ** 2
        )

        if magnitude_ba == 0 or magnitude_bc == 0:
            return None

        cosine_angle = (
            (ba[0] * bc[0]) +
            (ba[1] * bc[1])
        ) / (
            magnitude_ba * magnitude_bc
        )

        # Avoid floating-point error
        cosine_angle = max(
            -1.0,
            min(1.0, cosine_angle)
        )

        angle = math.degrees(
            math.acos(cosine_angle)
        )

        return angle

    except Exception:
        return None


def point_valid(point):
    """
    Checks whether the keypoint is valid or not.
    """

    if point is None:
        return False

    if len(point) < 2:
        return False

    x = float(point[0])
    y = float(point[1])

    # Handle the case of a missing point in YOLO
    if x <= 0 and y <= 0:
        return False

    return True


def average_point(a, b):
    """
    Calculates the center of two points.
    """

    if not point_valid(a) or not point_valid(b):
        return None

    return (
        (float(a[0]) + float(b[0])) / 2,
        (float(a[1]) + float(b[1])) / 2
    )


def detect_activity(keypoints):
    """
    From YOLO Pose keypoints, detects:
        Standing
        Sitting
        Walking

    YOLO COCO keypoints:

    5  = Left Shoulder
    6  = Right Shoulder

    11 = Left Hip
    12 = Right Hip

    13 = Left Knee
    14 = Right Knee

    15 = Left Ankle
    16 = Right Ankle
    """

    # -------------------------------------------------
    # KEYPOINT CHECK
    # -------------------------------------------------

    if keypoints is None:
        return "Unknown", 0.0

    if len(keypoints) < 17:
        return "Unknown", 0.0


    # -------------------------------------------------
    # KEYPOINTS
    # -------------------------------------------------

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    left_hip = keypoints[11]
    right_hip = keypoints[12]

    left_knee = keypoints[13]
    right_knee = keypoints[14]

    left_ankle = keypoints[15]
    right_ankle = keypoints[16]


    # -------------------------------------------------
    # CHECK BODY PARTS
    # -------------------------------------------------

    hip_center = average_point(
        left_hip,
        right_hip
    )

    shoulder_center = average_point(
        left_shoulder,
        right_shoulder
    )


    # -------------------------------------------------
    # KNEE ANGLES
    # -------------------------------------------------

    left_knee_angle = None
    right_knee_angle = None

    if (
        point_valid(left_hip)
        and point_valid(left_knee)
        and point_valid(left_ankle)
    ):
        left_knee_angle = calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )


    if (
        point_valid(right_hip)
        and point_valid(right_knee)
        and point_valid(right_ankle)
    ):
        right_knee_angle = calculate_angle(
            right_hip,
            right_knee,
            right_ankle
        )


    # -------------------------------------------------
    # AVERAGE KNEE ANGLE
    # -------------------------------------------------

    valid_angles = []

    if left_knee_angle is not None:
        valid_angles.append(left_knee_angle)

    if right_knee_angle is not None:
        valid_angles.append(right_knee_angle)


    average_knee_angle = None

    if len(valid_angles) > 0:
        average_knee_angle = (
            sum(valid_angles)
            / len(valid_angles)
        )


    # -------------------------------------------------
    # BODY MOVEMENT
    # -------------------------------------------------

    movement = 0.0

    if hip_center is not None:

        current_position = hip_center

        if len(motion_history) > 0:

            previous_position = motion_history[-1]

            movement = math.sqrt(
                (
                    current_position[0]
                    - previous_position[0]
                ) ** 2
                +
                (
                    current_position[1]
                    - previous_position[1]
                ) ** 2
            )

        motion_history.append(current_position)


    # -------------------------------------------------
    # WALKING DETECTION
    # -------------------------------------------------

    # We only consider Walking when there is
    # enough body movement.

    if len(motion_history) >= 8:

        recent_positions = list(motion_history)

        total_movement = 0.0

        for i in range(1, len(recent_positions)):

            p1 = recent_positions[i - 1]
            p2 = recent_positions[i]

            distance = math.sqrt(
                (p2[0] - p1[0]) ** 2
                +
                (p2[1] - p1[1]) ** 2
            )

            total_movement += distance


        average_movement = (
            total_movement
            / (len(recent_positions) - 1)
        )

        # Threshold
        if average_movement > 3.0:

            confidence = min(
                95.0,
                70.0 + average_movement * 3
            )

            return "Walking", confidence


    # -------------------------------------------------
    # SITTING DETECTION
    # -------------------------------------------------

    if average_knee_angle is not None:

        # Bent knees
        if average_knee_angle < 125:

            confidence = 75.0 + (
                (125 - average_knee_angle)
                * 0.5
            )

            confidence = min(
                95.0,
                confidence
            )

            return "Sitting", confidence


    # -------------------------------------------------
    # STANDING DETECTION
    # -------------------------------------------------

    if average_knee_angle is not None:

        # Legs relatively straight
        if average_knee_angle >= 155:

            confidence = 80.0 + (
                (average_knee_angle - 155)
                * 0.3
            )

            confidence = min(
                98.0,
                confidence
            )

            return "Standing", confidence


    # -------------------------------------------------
    # PARTIAL BODY / UNKNOWN
    # -------------------------------------------------

    # If the legs are not properly visible,
    # we will not make a false Standing/Walking claim.

    if shoulder_center is not None and hip_center is not None:

        return "Unknown", 50.0


    return "Unknown", 0.0