"""
pose_detection.py
=================
Person detection + position + activity (with justification).

Two functions are provided:

1) detect_pose(frame)
   -> (annotated_frame, keypoints)
   [Legacy-compatible interface — for camera.py]

2) analyze_frame(frame)
   -> (annotated_frame, persons)
   For every person:
      - Position  (bbox, center, frame region)
      - Activity  (Standing / Sitting / Walking / Unknown)
      - Justification (why the activity was detected)
"""

import math
from collections import deque

import cv2
from ultralytics import YOLO


# ============================================================
# MODEL
# ============================================================

model = YOLO("yolo11n-pose.pt")

# COCO keypoint order (YOLO pose model)
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


# ============================================================
# ACTIVITY THRESHOLDS
# ============================================================

KNEE_STRAIGHT_ANGLE = 155.0   # above this = legs straight (Standing)
KNEE_BENT_ANGLE = 125.0       # below this = legs bent (Sitting)
MOTION_HISTORY_LEN = 12       # remember this many frames of movement

# Walking detection (jitter-proof):
WALK_MIN_MOVEMENT = 4.0        # avg movement (px/frame) must be at least this
WALK_MIN_DISPLACEMENT = 25.0   # net displacement (px) — jitter is smaller than this
WALK_MIN_CONSISTENCY = 0.55    # net/path ratio — ~1.0 when walking, ~0 for random jitter
WALK_CONFIRM_FRAMES = 4        # Walking only after this many consecutive passing frames
WALK_EXIT_FRAMES = 6           # Walking exits after this many consecutive failing frames


# ============================================================
# SIMPLE TRACKER
# (To detect Walking, each person's movement must be tracked —
#  so every person gets a track ID via nearest-center matching)
# ============================================================

class _PersonTrack:

    def __init__(self, track_id, center):
        self.track_id = track_id
        self.history = deque(maxlen=MOTION_HISTORY_LEN)
        self.history.append(center)
        self.missed = 0

        # Walking hysteresis state (to block jitter false-positives)
        self.is_walking = False
        self.walk_streak = 0
        self.still_streak = 0

    def update(self, center):
        self.history.append(center)
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

    def average_movement(self):
        """Average movement of the body center in recent frames (px/frame)."""
        if len(self.history) < 3:
            return 0.0

        pts = list(self.history)
        total = 0.0

        for i in range(1, len(pts)):
            total += math.dist(pts[i - 1], pts[i])

        return total / (len(pts) - 1)

    def net_displacement(self):
        """
        Straight-line distance (px) between the first and last
        history points.
        Very small for Standing/jitter,
        large for Walking (the body actually moves forward).
        """
        if len(self.history) < 2:
            return 0.0

        pts = list(self.history)
        return math.dist(pts[0], pts[-1])

    def total_path(self):
        """Total distance of all steps in the history (px)."""
        if len(self.history) < 2:
            return 0.0

        pts = list(self.history)
        total = 0.0

        for i in range(1, len(pts)):
            total += math.dist(pts[i - 1], pts[i])

        return total

    def passes_walking_test(self):
        """
        Jitter-proof walking test — 3 conditions:
          1. Average movement above the threshold
          2. Large net displacement (not just wobble)
          3. Consistent direction (net / total path high)
        """
        movement = self.average_movement()

        if movement < WALK_MIN_MOVEMENT:
            return False

        net = self.net_displacement()

        if net < WALK_MIN_DISPLACEMENT:
            return False

        path = self.total_path()

        if path <= 0:
            return False

        consistency = net / path

        if consistency < WALK_MIN_CONSISTENCY:
            return False

        return True

    def update_walking_state(self):
        """
        Hysteresis: to ENTER Walking, the test must pass for
        WALK_CONFIRM_FRAMES consecutive frames; to EXIT, it must
        fail for WALK_EXIT_FRAMES consecutive frames.
        Returns: (is_walking, streak_count)
        """
        passed = self.passes_walking_test()

        if self.is_walking:

            if passed:
                self.still_streak = 0
            else:
                self.still_streak += 1

            if self.still_streak >= WALK_EXIT_FRAMES:
                self.is_walking = False
                self.still_streak = 0
                self.walk_streak = 0

            return self.is_walking, self.walk_streak

        else:

            if passed:
                self.walk_streak += 1
                self.still_streak = 0
            else:
                self.walk_streak = 0

            if self.walk_streak >= WALK_CONFIRM_FRAMES:
                self.is_walking = True

            return self.is_walking, self.walk_streak


_tracks = []
_next_track_id = 1
_TRACK_MATCH_MAX_DIST = 120   # px — matches farther than this are rejected
_TRACK_MAX_MISSED = 15        # remember a missing person for this many frames


def _update_tracks(centers):
    """
    Matches every person to the nearest track from the previous frame.
    Returns: list of _PersonTrack (same order as centers)
    """
    global _tracks, _next_track_id

    matched_tracks = []
    used_track_ids = set()

    for center in centers:

        best_track = None
        best_dist = _TRACK_MATCH_MAX_DIST

        for track in _tracks:
            if track.track_id in used_track_ids:
                continue

            dist = math.dist(track.history[-1], center)

            if dist < best_dist:
                best_dist = dist
                best_track = track

        if best_track is not None:
            best_track.update(center)
            used_track_ids.add(best_track.track_id)
            matched_tracks.append(best_track)

        else:
            # A new person has entered the frame
            track = _PersonTrack(_next_track_id, center)
            _next_track_id += 1
            matched_tracks.append(track)

    # Mark tracks that did not match this frame as "missed"
    matched_ids = {t.track_id for t in matched_tracks}

    for track in _tracks:
        if track.track_id not in matched_ids:
            track.mark_missed()

    # Surviving tracks: matched + unmatched (currently missing)
    _tracks = matched_tracks + [
        t for t in _tracks
        if t.track_id not in matched_ids
    ]

    # Drop very old missing tracks
    _tracks = [
        t for t in _tracks
        if t.missed <= _TRACK_MAX_MISSED
    ]

    return matched_tracks


# ============================================================
# TRACKER RESET (for a new video / task evaluation)
# ============================================================

def reset_tracker():
    """
    Resets all person tracks.
    Call before starting a new video or task evaluation,
    so old movement history does not affect the new analysis.
    """
    global _tracks, _next_track_id

    _tracks = []
    _next_track_id = 1


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def _point_valid(point):
    """Checks whether the keypoint is valid or not."""

    if point is None:
        return False

    if len(point) < 2:
        return False

    x = float(point[0])
    y = float(point[1])

    # A missing point in YOLO is (0, 0)
    if x <= 0 and y <= 0:
        return False

    return True


def _angle(a, b, c):
    """
    Angle between points A-B-C (degrees).
    B = middle/joint point.
    """
    if not (_point_valid(a) and _point_valid(b) and _point_valid(c)):
        return None

    ba = (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    bc = (float(c[0]) - float(b[0]), float(c[1]) - float(b[1]))

    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])

    if mag_ba == 0 or mag_bc == 0:
        return None

    cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))


def _mid(a, b):
    """Center of two points (None-safe)."""
    if not (_point_valid(a) and _point_valid(b)):
        return None

    return (
        (float(a[0]) + float(b[0])) / 2,
        (float(a[1]) + float(b[1])) / 2
    )


# ============================================================
# POSITION ANALYSIS
# ============================================================

def _describe_position(bbox, frame_w, frame_h):
    """
    Describes a person's position:
      - bbox in pixels
      - body center
      - normalized (0-1) coordinates
      - frame region (top/middle/bottom + left/center/right)
    """

    x1, y1, x2, y2 = [float(v) for v in bbox]

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    norm_x = center_x / max(1, frame_w)
    norm_y = center_y / max(1, frame_h)

    # Horizontal region
    if norm_x < 0.33:
        horizontal = "left"
    elif norm_x < 0.66:
        horizontal = "center"
    else:
        horizontal = "right"

    # Vertical region
    if norm_y < 0.33:
        vertical = "top"
    elif norm_y < 0.66:
        vertical = "middle"
    else:
        vertical = "bottom"

    return {
        "bbox_pixels": [
            round(x1, 1), round(y1, 1),
            round(x2, 1), round(y2, 1)
        ],
        "center_pixel": (
            round(center_x, 1), round(center_y, 1)
        ),
        "normalized": (
            round(norm_x, 3), round(norm_y, 3)
        ),
        "description": f"{vertical}-{horizontal} region of frame"
    }


# ============================================================
# ACTIVITY ANALYSIS (with justification)
# ============================================================

def _analyze_activity(keypoints, track):
    """
    Extracts the activity from keypoints + movement history
    and returns a reason (justification) for every decision.

    Returns: (activity, confidence, reasons_list)
    """

    reasons = []

    # ------------------------------------------------
    # KEYPOINT CHECK
    # ------------------------------------------------

    if keypoints is None or len(keypoints) < 17:
        return "Unknown", 0.0, ["No body keypoints detected"]

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    # ------------------------------------------------
    # KNEE ANGLES
    # ------------------------------------------------

    left_knee_angle = _angle(left_hip, left_knee, left_ankle)
    right_knee_angle = _angle(right_hip, right_knee, right_ankle)

    if left_knee_angle is not None:
        reasons.append(f"Left knee angle: {left_knee_angle:.0f} deg")
    else:
        reasons.append("Left leg keypoints not visible")

    if right_knee_angle is not None:
        reasons.append(f"Right knee angle: {right_knee_angle:.0f} deg")
    else:
        reasons.append("Right leg keypoints not visible")

    valid_angles = [
        a for a in (left_knee_angle, right_knee_angle)
        if a is not None
    ]

    avg_knee_angle = None
    if valid_angles:
        avg_knee_angle = sum(valid_angles) / len(valid_angles)

    # ------------------------------------------------
    # TORSO TILT (upright vs leaning)
    # ------------------------------------------------

    shoulder_center = _mid(left_shoulder, right_shoulder)
    hip_center = _mid(left_hip, right_hip)

    if shoulder_center is not None and hip_center is not None:
        dx = abs(shoulder_center[0] - hip_center[0])
        dy = abs(shoulder_center[1] - hip_center[1])

        if dy > 0:
            torso_tilt = math.degrees(math.atan2(dx, dy))
            reasons.append(f"Torso tilt: {torso_tilt:.0f} deg")

    # ------------------------------------------------
    # BODY MOVEMENT (from tracker history)
    # ------------------------------------------------

    movement = track.average_movement() if track is not None else 0.0

    net_disp = track.net_displacement() if track is not None else 0.0

    track_path = track.total_path() if track is not None else 0.0

    consistency = (
        (net_disp / track_path)
        if track_path > 0
        else 0.0
    )

    reasons.append(
        f"Average body movement: {movement:.1f} px/frame"
    )

    reasons.append(
        f"Net displacement: {net_disp:.1f} px | "
        f"Direction consistency: {consistency:.2f}"
    )

    # ------------------------------------------------
    # WALKING CHECK (jitter-proof + hysteresis)
    # ------------------------------------------------
    # Walking cannot be claimed from per-frame movement alone,
    # because standing also has body/camera jitter.
    # Therefore all three checks run: movement + net displacement
    # + direction consistency, and the CONFIRM-frame streak
    # must be complete.

    if track is not None:

        is_walking, streak = track.update_walking_state()

        if is_walking:

            confidence = min(95.0, 75.0 + movement * 2)

            reasons.insert(
                0,
                f"Body center is consistently moving: "
                f"{movement:.1f} px/frame movement, "
                f"{net_disp:.0f} px net displacement, "
                f"{consistency:.2f} direction consistency"
            )

            return "Walking", confidence, reasons

        if movement >= WALK_MIN_MOVEMENT:

            reasons.append(
                f"Movement detected but Walking not confirmed — "
                f"displacement or consistency is too low "
                f"(streak: {streak}/{WALK_CONFIRM_FRAMES})"
            )

    # ------------------------------------------------
    # SITTING CHECK (knees bent)
    # ------------------------------------------------

    if avg_knee_angle is not None and avg_knee_angle < KNEE_BENT_ANGLE:

        confidence = min(
            95.0,
            75.0 + (KNEE_BENT_ANGLE - avg_knee_angle) * 0.5
        )

        reasons.insert(
            0,
            f"Both knees are bent "
            f"(avg knee angle {avg_knee_angle:.0f} deg < {KNEE_BENT_ANGLE:.0f} deg)"
        )

        return "Sitting", confidence, reasons

    # ------------------------------------------------
    # STANDING CHECK (legs straight)
    # ------------------------------------------------

    if avg_knee_angle is not None and avg_knee_angle >= KNEE_STRAIGHT_ANGLE:

        confidence = min(
            98.0,
            80.0 + (avg_knee_angle - KNEE_STRAIGHT_ANGLE) * 0.3
        )

        reasons.insert(
            0,
            f"Legs are straight "
            f"(avg knee angle {avg_knee_angle:.0f} deg >= {KNEE_STRAIGHT_ANGLE:.0f} deg)"
        )

        return "Standing", confidence, reasons

    # ------------------------------------------------
    # UNKNOWN (legs are not clearly visible)
    # ------------------------------------------------

    reasons.insert(
        0,
        "Legs are not properly visible, "
        "so Standing/Sitting cannot be stated confidently"
    )

    if shoulder_center is not None and hip_center is not None:
        return "Unknown", 50.0, reasons

    return "Unknown", 0.0, reasons


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def detect_pose(frame):
    """
    Legacy-compatible interface (for camera.py).

    Returns:
        (annotated_frame, keypoints)
        keypoints = numpy array of the first person's 17 keypoints
                    (None if no person was found)
    """
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


def analyze_frame(frame):
    """
    FULL ANALYSIS: position + activity + justification for every person.

    Returns:
        (annotated_frame, persons)

    persons = list of dicts:
        {
            "track_id":          int,
            "position":          {...},
            "activity":          str,
            "confidence":        float,
            "justification":     [str, str, ...],
            "keypoints":         {name: (x, y), ...},
            "person_confidence": float,   # detection confidence
        }
    """

    frame_h, frame_w = frame.shape[:2]

    results = model(frame, verbose=False)
    r = results[0]

    annotated_frame = r.plot()

    persons = []

    boxes = r.boxes
    kps = r.keypoints

    person_count = len(boxes) if boxes is not None else 0

    if person_count == 0:
        return annotated_frame, persons

    # ------------------------------------------------
    # Step 1: compute each person's body center (for tracking)
    # ------------------------------------------------

    centers = []

    for i in range(person_count):

        center = None

        if kps is not None and i < len(kps.xy):
            xy = kps.xy[i].cpu().numpy()

            if len(xy) >= 17:
                center = _mid(xy[11], xy[12])  # hip center

        if center is None:
            # Fall back to bbox center when keypoints are missing
            bbox = boxes.xyxy[i].cpu().numpy()
            center = (
                (float(bbox[0]) + float(bbox[2])) / 2,
                (float(bbox[1]) + float(bbox[3])) / 2
            )

        centers.append(center)

    # Step 2: tracker update (movement history for Walking)
    tracks = _update_tracks(centers)

    # ------------------------------------------------
    # Step 3: full analysis for every person
    # ------------------------------------------------

    for i in range(person_count):

        bbox = boxes.xyxy[i].cpu().numpy()
        track = tracks[i] if i < len(tracks) else None

        xy = None
        if kps is not None and i < len(kps.xy):
            xy = kps.xy[i].cpu().numpy()

        # Position
        position = _describe_position(bbox, frame_w, frame_h)

        # Activity + justification
        activity, confidence, reasons = _analyze_activity(xy, track)

        # Keypoints dict (named)
        keypoints_dict = None
        if xy is not None and len(xy) >= 17:
            keypoints_dict = {
                name: (round(float(p[0]), 1), round(float(p[1]), 1))
                for name, p in zip(KEYPOINT_NAMES, xy)
            }

        # Detection confidence (YOLO person confidence)
        person_conf = 0.0
        if boxes.conf is not None:
            person_conf = float(boxes.conf[i]) * 100

        persons.append({
            "track_id": track.track_id if track else i + 1,
            "position": position,
            "activity": activity,
            "confidence": round(float(confidence), 1),
            "justification": reasons,
            "keypoints": keypoints_dict,
            "person_confidence": round(person_conf, 1),
        })

        # ------------------------------------------------
        # Draw the activity label on the frame
        # ------------------------------------------------

        x1, y1 = int(bbox[0]), int(bbox[1])

        label = (
            f"ID {persons[-1]['track_id']}: "
            f"{activity} ({persons[-1]['confidence']:.0f}%)"
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    return annotated_frame, persons


# ============================================================
# LIVE DEMO (when run directly)
# ============================================================

if __name__ == "__main__":

    print("Live pose detection + position + activity justification")
    print("Press 'q' to quit.\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        raise SystemExit

    frame_count = 0
    prev_time = cv2.getTickCount()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not receive frame.")
            break

        annotated_frame, persons = analyze_frame(frame)

        frame_count += 1

        # FPS
        current_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_time - prev_time)
        prev_time = current_time

        # ------------------------------------------------
        # CONSOLE OUTPUT
        # ------------------------------------------------

        print(
            f"\n=== Frame {frame_count} | "
            f"FPS: {fps:.1f} | "
            f"Persons: {len(persons)} ==="
        )

        if not persons:
            print("  No person detected.")

        for p in persons:

            print(f"  Person ID {p['track_id']}:")
            print(f"    Position   : {p['position']['description']}")
            print(f"    Center     : {p['position']['center_pixel']}")
            print(f"    BBox       : {p['position']['bbox_pixels']}")
            print(f"    Activity   : {p['activity']} ({p['confidence']}%)")
            print(f"    Justification:")

            for reason in p["justification"]:
                print(f"      - {reason}")

        cv2.imshow(
            "Pose + Position + Activity (q = quit)",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
