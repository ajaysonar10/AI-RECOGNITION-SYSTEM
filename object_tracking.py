"""
object_tracking.py
==================

BAS-AI • Object Tracking for Task Verification

Converts frame-by-frame YOLO detections into persistent TRACKS so that
object MOVEMENT (picking up, putting down, carrying around) can be
measured.

Every track has:
  - box / center / confidence / class
  - history (recent centers)      -> displacement + stability
  - hand proximity (left/right)   -> hold detection (near the wrist)
  - hold state                    -> when it was picked up, how much
                                     it was lifted, who grabbed it

Both object_tasks.py (task logic) and camera.py (live loop) use this
module. Pure geometry — no random/dummy values.
"""

import math
from collections import deque

import cv2


# ============================================================
# SETTINGS
# ============================================================

TRACK_HISTORY_LEN = 30      # ~4-6 seconds of recent centers
TRACK_MAX_MISSED = 12       # missing for this many frames -> track deleted
MATCH_MIN_IOU = 0.25        # IoU below this -> no match
MATCH_MAX_CENTER_RATIO = 0.8  # center dist < ratio * box diag -> match

HOLD_ENTER_FRAMES = 6       # wrist near for this many frames -> HOLD
HOLD_RELEASE_FRAMES = 4     # wrist away for this long -> RELEASE
NEAR_HAND_GAIN = 2          # streak gain on a near frame
NEAR_HAND_DECAY = 2         # streak decay on an away frame

# When to reset the lift baseline (object back at REST). The baseline
# is RETAINED even after release so the lift measurement does not
# break during wrist-flicker / re-grip (real-life pick bug).
REST_RESET_FRAMES = 8       # free + stable + no-hand frames -> rest

# Release confirm: release only after this many CONSECUTIVE no-hand
# frames (the wrist keypoint often disappears for a frame or two when
# the object is in hand — prevent false release).
RELEASE_CONFIRM_FRAMES = 2

# A held object moves fast with the hand — relax the match radius
# (otherwise a fast lift re-ids the track and the lift
# measurement is lost).
HELD_MATCH_CENTER_RATIO = 1.6


def _iou(box_a, box_b):
    """Intersection-over-Union between two boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _box_diag(box):
    return math.hypot(box[2] - box[0], box[3] - box[1])


def _center(box):
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def _point_valid(point):
    """Missing/zero keypoint safe check (same rule as the pose pipeline)."""
    if point is None:
        return False
    try:
        if len(point) < 2:
            return False
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError):
        return False
    if x <= 0 and y <= 0:
        return False
    return True


def _torso_length(keypoints):
    """Person body scale (shoulder-hip distance). Fallback: shoulder width."""
    def get(name):
        idx = {
            "left_shoulder": 5, "right_shoulder": 6,
            "left_hip": 11, "right_hip": 12,
        }[name]

        if isinstance(keypoints, dict):
            return keypoints.get(name)
        try:
            return keypoints[idx] if len(keypoints) > idx else None
        except (TypeError, IndexError):
            return None

    ls, rs = get("left_shoulder"), get("right_shoulder")
    lh, rh = get("left_hip"), get("right_hip")

    if all(_point_valid(p) for p in (ls, rs, lh, rh)):
        sm = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hm = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        return max(1.0, math.dist(sm, hm))

    if all(_point_valid(p) for p in (ls, rs)):
        return max(1.0, math.dist(ls, rs))

    return 100.0


def _wrist(keypoints, side):
    idx = 9 if side == "left" else 10

    if isinstance(keypoints, dict):
        point = keypoints.get(f"{side}_wrist")
    else:
        try:
            point = keypoints[idx] if len(keypoints) > idx else None
        except (TypeError, IndexError):
            point = None

    if not _point_valid(point):
        return None

    return (float(point[0]), float(point[1]))


def _elbow(keypoints, side):
    """Elbow keypoint (for hold SUSTAIN — the wrist occludes during a grip)."""
    idx = 7 if side == "left" else 8

    if isinstance(keypoints, dict):
        point = keypoints.get(f"{side}_elbow")
    else:
        try:
            point = keypoints[idx] if len(keypoints) > idx else None
        except (TypeError, IndexError):
            point = None

    if not _point_valid(point):
        return None

    return (float(point[0]), float(point[1]))


# ============================================================
# TRACK
# ============================================================

class ObjectTrack:
    """Persistent identity + movement history of one detected object."""

    def __init__(self, track_id, class_name, box, confidence):
        self.track_id = track_id
        self.class_name = class_name
        self.box = tuple(float(v) for v in box)
        self.center = _center(self.box)
        self.confidence = float(confidence)

        self.history = deque(maxlen=TRACK_HISTORY_LEN)
        self.history.append(self.center)

        self.seen = 1
        self.missed = 0

        # recent confidences (for avg_confidence)
        self.conf_history = deque(maxlen=12)
        self.conf_history.append(self.confidence)

        # ---- hand proximity / hold state ----
        self.near_streak = {"left": 0, "right": 0}
        self.hold_frames = 0            # consecutive near-hand frames
        self.held_by = None             # {"side": ..., "person_id": ...}
        self.ever_held = False
        self.holder_history = deque(maxlen=10)   # person ids jo pakde

        # container membership (updated by object_tasks.evaluate_evidence
        # — which object is inside which container)
        self.members = set()

        # center at hold start (displacement/lift are measured from this)
        self.hold_start_center = None
        self.max_lift = 0.0

        # LIFT BASELINE — rest position (or hold-enter position).
        # NOT cleared on release; reset only after REST_RESET_FRAMES
        # (the object is back at rest).
        # This keeps the lift measurement alive during re-grip /
        # wrist-flicker.
        self.lift_baseline = None
        self._rest_frames = 0
        self._release_streak = 0

    # ------------------------------------------------
    # UPDATE
    # ------------------------------------------------

    def update(self, box, confidence):
        self.box = tuple(float(v) for v in box)
        self.center = _center(self.box)
        self.confidence = float(confidence)
        self.conf_history.append(self.confidence)

        self.history.append(self.center)
        self.seen += 1
        self.missed = 0

    @property
    def avg_confidence(self):
        """
        Average confidence of recent detections (last ~12 frames).
        Robust to weak-but-persistent detections (like a table seen
        from a webcam angle, conf 0.07-0.11 flicker) — one-frame noise
        cannot lift the average.
        """
        if not self.conf_history:
            return self.confidence
        return sum(self.conf_history) / len(self.conf_history)

    def mark_missed(self):
        self.missed += 1

    # ------------------------------------------------
    # MOVEMENT MEASUREMENTS (real geometry)
    # ------------------------------------------------

    def net_displacement(self, window=None):
        """Straight-line distance (px) between the oldest and newest center."""
        pts = list(self.history)
        if window:
            pts = pts[-window:]

        if len(pts) < 2:
            return 0.0
        return math.dist(pts[0], pts[-1])

    def total_path(self, window=None):
        pts = list(self.history)
        if window:
            pts = pts[-window:]

        total = 0.0
        for i in range(1, len(pts)):
            total += math.dist(pts[i - 1], pts[i])
        return total

    def is_stable(self, frames=6, max_px=6.0):
        """Has the object been roughly stationary in recent frames?"""
        pts = list(self.history)[-frames:]
        if len(pts) < frames:
            return False

        moved = sum(
            math.dist(pts[i - 1], pts[i])
            for i in range(1, len(pts))
        )
        return (moved / (len(pts) - 1)) <= max_px

    def displacement_while_held(self):
        """How far the object has moved since the hold started (px)."""
        if self.hold_start_center is None:
            return 0.0
        return math.dist(self.hold_start_center, self.center)

    def lift_px(self):
        """
        How far UP the object was lifted during the hold
        (px, positive = up).
        The baseline comes from the last REST position (or hold-enter)
        and is NOT cleared on release — the lift measurement stays
        valid across re-grip / wrist-flicker until the object returns
        to rest (REST_RESET_FRAMES).
        """
        if self.lift_baseline is None:
            return 0.0
        return self.lift_baseline[1] - self.center[1]

    def effective_lift_px(self):
        """
        Robust lift for pick tasks: the current lift OR the max lift
        achieved during the hold — whichever is larger. Remains valid
        after release/re-grip flicker until the object returns to rest
        (after REST_RESET_FRAMES both baseline and max have reset).
        """
        lift = self.lift_px()

        if self._rest_frames >= REST_RESET_FRAMES:
            return lift

        return max(lift, self.max_lift)

    def carry_displacement(self):
        """
        For carry/move tasks: the straight-line distance from the
        baseline (last rest position or hold-enter) until now. The
        baseline is retained across release — so even if the hold
        flickers during a carry, the full carry distance is measured
        (it used to reset to 0 on re-grip — the "needed 168, moved 61"
        live-test bug).
        """
        if self.lift_baseline is None:
            return 0.0
        return math.dist(self.lift_baseline, self.center)

    # ------------------------------------------------
    # REGION HELPERS
    # ------------------------------------------------

    def inside_box(self, region_box, expand=0.0):
        """Is the object center inside the region box? (expand = fraction)"""
        if region_box is None:
            return False

        x1, y1, x2, y2 = region_box

        if expand > 0:
            ex = (x2 - x1) * expand
            ey = (y2 - y1) * expand
            x1, y1, x2, y2 = x1 - ex, y1 - ey, x2 + ex, y2 + ey

        cx, cy = self.center
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def was_inside_box(self, region_box, expand=0.0):
        """Was it ever inside the region in recent history?"""
        if region_box is None:
            return False

        x1, y1, x2, y2 = region_box
        if expand > 0:
            ex = (x2 - x1) * expand
            ey = (y2 - y1) * expand
            x1, y1, x2, y2 = x1 - ex, y1 - ey, x2 + ex, y2 + ey

        for cx, cy in self.history:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    def horizontal_overlap(self, region_box):
        """Horizontal overlap fraction between the object box and the region box."""
        if region_box is None:
            return 0.0

        ox1, oy1, ox2, oy2 = self.box
        rx1, ry1, rx2, ry2 = region_box

        overlap = max(0.0, min(ox2, rx2) - max(ox1, rx1))
        obj_width = max(1.0, ox2 - ox1)

        return overlap / obj_width

    def near_vertical(self, region_box, tolerance_frac=0.25):
        """
        Is the object's bottom near the region's top? (to measure
        something resting on a surface — table/bowl rim)
        """
        if region_box is None:
            return False

        surface_y = region_box[1]
        obj_bottom = self.box[3]
        tolerance = (region_box[3] - region_box[1]) * tolerance_frac

        return abs(obj_bottom - surface_y) <= tolerance

    # ------------------------------------------------
    # HAND PROXIMITY (hold detection)
    # ------------------------------------------------

    def update_hand_proximity(self, persons):
        """
        Called every frame (inside tracker.update).
        Maintains the hold state from wrist-object proximity.
        Reach = max(0.7 * torso, 0.5 * box diagonal) — a scale
        independent of camera distance.
        """
        diag = _box_diag(self.box)

        best_near = None      # (dist, side, person_id, reach) — WRIST only
        best_elbow = None     # (dist, side, person_id) — sustain fallback

        for person in persons or []:

            keypoints = person.get("keypoints")
            if not keypoints:
                continue

            torso = _torso_length(keypoints)
            reach = max(0.7 * torso, 0.5 * diag)

            for side in ("left", "right"):
                wrist = _wrist(keypoints, side)

                if wrist is not None:
                    dist = math.dist(wrist, self.center)

                    if dist < reach:
                        if best_near is None or dist < best_near[0]:
                            best_near = (
                                dist, side, person.get("track_id"), reach
                            )

                # ELBOW FALLBACK: when gripping an object, the wrist
                # keypoint is often occluded/dropped (missing for 12s
                # in a live test). The elbow near the object = hold
                # SUSTAIN evidence.
                # NOTE: hold INITIATION happens only via the wrist.
                elbow = _elbow(keypoints, side)

                if elbow is not None:
                    e_radius = min(0.55 * diag, 2.0 * torso)
                    edist = math.dist(elbow, self.center)

                    if edist < e_radius:
                        if best_elbow is None or edist < best_elbow[0]:
                            best_elbow = (
                                edist, side, person.get("track_id")
                            )

        # streaks update
        near_sides = set()

        if best_near is not None:
            _, side, person_id, _ = best_near
            near_sides.add(side)
            self.near_streak[side] = min(
                HOLD_ENTER_FRAMES + 6, self.near_streak[side] + NEAR_HAND_GAIN
            )
            self._pending_holder = (side, person_id)
        else:
            self._pending_holder = None

        for side in ("left", "right"):
            if side not in near_sides:
                self.near_streak[side] = max(
                    0, self.near_streak[side] - NEAR_HAND_DECAY
                )

        best_streak = max(self.near_streak.values())

        # --------------------------------------------
        # REST BASELINE (reference for lift measurement)
        # Free + stable + no hand near — the object is genuinely at
        # rest -> reset the baseline. During small re-grip/spray gaps
        # the baseline survives (the lift measurement does not break).
        # --------------------------------------------

        if self.held_by is None and best_near is None and best_elbow is None:
            if self.is_stable(frames=4, max_px=10.0):
                self._rest_frames += 1
            else:
                self._rest_frames = 0

            if self._rest_frames >= REST_RESET_FRAMES:
                self.lift_baseline = self.center
                self.max_lift = 0.0
        else:
            self._rest_frames = 0

        # --------------------------------------------
        # HOLD ENTER (hysteresis)
        # --------------------------------------------

        if best_near is not None and self.held_by is None:
            if best_streak >= HOLD_ENTER_FRAMES:
                side, person_id = self._pending_holder
                self.held_by = {"side": side, "person_id": person_id}
                self.ever_held = True
                self.hold_frames = best_streak
                self.hold_start_center = self.center

                # Lift baseline: on the first grip (or when lifting off
                # from rest) use the current center; on a mid-air
                # RE-GRIP retain the old baseline (otherwise the lift
                # resets to 0 — the bug that stopped pick from ever
                # completing).
                if self.lift_baseline is None:
                    self.lift_baseline = self.center
                    self.max_lift = 0.0

                if person_id is not None:
                    self.holder_history.append(person_id)

        elif self.held_by is not None:
            # Sustain: wrist OR elbow near (the wrist occludes during a
            # grip — live-verified). Both gone -> 2-frame release confirm.
            sustained = best_near is not None or best_elbow is not None

            if sustained:
                self.hold_frames += 1
                self.max_lift = max(self.max_lift, self.lift_px())
                self._release_streak = 0
            else:
                self._release_streak += 1
                self.hold_frames = max(0, self.hold_frames - 1)

                if (
                    self._release_streak >= RELEASE_CONFIRM_FRAMES
                    and best_streak < HOLD_ENTER_FRAMES * 0.5
                ):
                    # RELEASE (with hysteresis)
                    # NOTE: lift_baseline is deliberately RETAINED —
                    # after a wrist flicker, a re-grip measures the lift
                    # from the same baseline.
                    self.held_by = None
                    self.hold_start_center = None
                    self._release_streak = 0

    def is_held(self):
        return self.held_by is not None


# ============================================================
# TRACKER
# ============================================================

class ObjectTracker:
    """
    Frame-by-frame YOLO detections -> persistent tracks.

    Usage (in the camera loop, every frame):
        tracker.update(detections, persons)
        tracks = tracker.tracks
    """

    def __init__(self):
        self.tracks = []
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    # ------------------------------------------------
    # MATCHING
    # ------------------------------------------------

    def _best_match(self, det):
        """Match the best track of the same class (IoU or center)."""
        det_box = det["box"]
        det_center = _center(det_box)
        det_diag = _box_diag(det_box)

        best = None
        best_score = 0.0

        for track in self.tracks:

            if track.class_name != det["class_name"]:
                continue

            iou = _iou(track.box, det_box)
            center_dist = math.dist(track.center, det_center)

            # A held / about-to-be-gripped object moves fast with the
            # hand — relax its match radius (so the track does not re-id
            # on a fast lift; otherwise the lift measurement is lost).
            ratio = MATCH_MAX_CENTER_RATIO
            if track.is_held() or max(track.near_streak.values()) >= 3:
                ratio = HELD_MATCH_CENTER_RATIO

            center_ok = center_dist < ratio * max(
                det_diag, _box_diag(track.box)
            )

            if (
                iou >= MATCH_MIN_IOU
                or (iou > 0 and center_ok)
                or (track.is_held() and center_ok)
            ):
                # A held object moves fast in the hand — overlap can
                # sometimes be zero; center-distance matching suffices.
                score = iou + (0.2 if center_ok else 0.0)

                if score > best_score:
                    best_score = score
                    best = track

        return best

    # ------------------------------------------------
    # MAIN UPDATE
    # ------------------------------------------------

    def update(self, detections, persons=None):
        """
        detections: list from object_detection.detect_objects()
                    (person class is filtered out)
        persons:    persons list from pose_detection.analyze_frame()

        Returns: list of active tracks
        """
        matched_tracks = set()

        for det in detections or []:

            # Do not keep persons in the object tracker
            # (the pose pipeline tracks them)
            if det.get("class_name") == "person":
                continue

            track = self._best_match(det)

            if track is not None:
                track.update(det["box"], det["confidence"])
                matched_tracks.add(track.track_id)
            else:
                track = ObjectTrack(
                    self._next_id,
                    det["class_name"],
                    det["box"],
                    det["confidence"],
                )
                self._next_id += 1
                self.tracks.append(track)
                matched_tracks.add(track.track_id)

        # unmatched tracks -> missed
        for track in self.tracks:
            if track.track_id not in matched_tracks:
                track.mark_missed()

        # hand proximity + hold state (visible tracks only)
        for track in self.tracks:
            if track.missed == 0:
                track.update_hand_proximity(persons)

        # drop old missing tracks
        self.tracks = [
            t for t in self.tracks if t.missed <= TRACK_MAX_MISSED
        ]

        return [t for t in self.tracks if t.missed == 0]

    # ------------------------------------------------
    # QUERIES
    # ------------------------------------------------

    def of_class(self, class_names, min_conf=0.35):
        """Visible tracks of the given classes (sorted by confidence)."""
        if isinstance(class_names, str):
            class_names = [class_names]

        found = [
            t for t in self.tracks
            if t.missed == 0
            and t.class_name in class_names
            and t.confidence >= min_conf
        ]

        return sorted(found, key=lambda t: -t.confidence)

    def any_object(self, min_conf=0.40, min_diag=40.0):
        """Any non-person object track (for "pick up an object")."""
        found = [
            t for t in self.tracks
            if t.missed == 0
            and t.class_name != "person"
            and t.confidence >= min_conf
            and _box_diag(t.box) >= min_diag
        ]

        return sorted(found, key=lambda t: -t.confidence)

    def containers(self, container_classes, min_conf=0.35):
        return self.of_class(container_classes, min_conf)


# ============================================================
# DRAW HELPERS (for camera.py)
# ============================================================

TRACK_COLORS = {
    True: (0, 220, 255),     # held -> yellow-ish (BGR)
    False: (255, 160, 60),   # free -> blue-ish
}


def draw_tracks(frame, tracks, region_box=None):
    """
    Draws the tracks + optional target region(s) on the frame.
    region_box: one box [x1,y1,x2,y2] OR a list of boxes
    (2 boxes = PLACE 1 / PLACE 2 labels, move_bottle_places demo).
    """

    for track in tracks:

        x1, y1, x2, y2 = [int(v) for v in track.box]
        color = TRACK_COLORS[track.is_held()]

        cv_label = f"#{track.track_id} {track.class_name}"

        if track.is_held():
            cv_label += " [HELD]"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            frame,
            cv_label,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if region_box is not None:

        regions = region_box
        if region_box and not isinstance(region_box[0], (list, tuple)):
            regions = [region_box]

        for i, box in enumerate(regions):
            rx1, ry1, rx2, ry2 = [int(v) for v in box]

            label = "TARGET"
            if len(regions) == 2:
                label = f"PLACE {i + 1}"

            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (90, 255, 90), 2)

            cv2.putText(
                frame,
                label,
                (rx1 + 4, max(18, ry1 + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (90, 255, 90),
                1,
                cv2.LINE_AA,
            )

    return frame
