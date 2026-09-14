"""
object_tracking.py
==================

BAS-AI • Object Tracking for Task Verification

YOLO detections (frame-by-frame) ko persistent TRACKS me convert karta hai
taaki object MOVEMENT (uthana, rakhna, idhar-udhar lena) measure ho sake.

Har track ke paas hota hai:
  - box / center / confidence / class
  - history (recent centers)      -> displacement + stability
  - hand proximity (left/right)   -> hold detection (wrist ke paas)
  - hold state                    -> kab uthaya, kitna uthaya (lift),
                                     kis person ne pakda

Ye module object_tasks.py (task logic) aur camera.py (live loop) dono
use karte hain. Pure geometry — koi random/dummy value nahi.
"""

import math
from collections import deque

import cv2


# ============================================================
# SETTINGS
# ============================================================

TRACK_HISTORY_LEN = 30      # ~4-6 seconds of recent centers
TRACK_MAX_MISSED = 12       # itne frames gayab -> track delete
MATCH_MIN_IOU = 0.25        # IoU isse kam -> match nahi
MATCH_MAX_CENTER_RATIO = 0.8  # center dist < ratio * box diag -> match

HOLD_ENTER_FRAMES = 6       # wrist itne frames paas rahe -> HOLD
HOLD_RELEASE_FRAMES = 4     # wrist itni der door -> RELEASE
NEAR_HAND_GAIN = 2          # paas frame par streak gain
NEAR_HAND_DECAY = 2         # door frame par streak decay

# Lift baseline kab reset ho (object wapas REST par). Release ke
# baad bhi baseline RETAIN hota hai taaki wrist-flicker / re-grip
# ke dauran lift measurement na toote (real-life pick bug).
REST_RESET_FRAMES = 8       # free + stable + no-hand frames -> rest

# Release confirm: itne LAGATAR no-hand frames ke baad hi release
# (wrist keypoint ek-do frame ke liye often gayab ho jaata hai jab
# object haath me ho — false release roko).
RELEASE_CONFIRM_FRAMES = 2

# Held object haath ke saath tez chalta hai — iske liye match radius
# relax karo (warna fast lift par track re-id hota hai aur lift
# measurement kho jaata hai).
HELD_MATCH_CENTER_RATIO = 1.6


def _iou(box_a, box_b):
    """Do boxes ke beech Intersection-over-Union."""
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
    """Missing/zero keypoint safe check (pose pipeline jaisa rule)."""
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
    """Elbow keypoint (hold SUSTAIN ke liye — grip me wrist occlude hota hai)."""
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
    """Ek detected object ki persistent identity + movement history."""

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

        # recent confidences (avg_confidence ke liye)
        self.conf_history = deque(maxlen=12)
        self.conf_history.append(self.confidence)

        # ---- hand proximity / hold state ----
        self.near_streak = {"left": 0, "right": 0}
        self.hold_frames = 0            # consecutive near-hand frames
        self.held_by = None             # {"side": ..., "person_id": ...}
        self.ever_held = False
        self.holder_history = deque(maxlen=10)   # person ids jo pakde

        # container membership (object_tasks.evaluate_evidence update
        # karta hai — kaunsa object kis container ke andar hai)
        self.members = set()

        # hold shuru hote waqt ka center (displacement/lift isse)
        self.hold_start_center = None
        self.max_lift = 0.0

        # LIFT BASELINE — rest position (ya hold-enter position).
        # Release par CLEAR nahi hota; sirf REST_RESET_FRAMES ke
        # baad reset hota hai (object wapas rest par aa chuka hai).
        # Isse re-grip / wrist-flicker ke dauran lift measure hoti rehti hai.
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
        Recent detections ki average confidence (last ~12 frames).
        Weak-but-persistent detections (jaise webcam angle se table,
        conf 0.07-0.11 flicker) ke liye robust — ek-frame noise
        average ko utha nahi sakta.
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
        """Purane aur naye center ke beech seedha distance (px)."""
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
        """Recent frames me object lagbhag stationary hai?"""
        pts = list(self.history)[-frames:]
        if len(pts) < frames:
            return False

        moved = sum(
            math.dist(pts[i - 1], pts[i])
            for i in range(1, len(pts))
        )
        return (moved / (len(pts) - 1)) <= max_px

    def displacement_while_held(self):
        """Hold shuru hone ke baad object kitna door gaya (px)."""
        if self.hold_start_center is None:
            return 0.0
        return math.dist(self.hold_start_center, self.center)

    def lift_px(self):
        """
        Hold ke dauran object kitna UPAR utha (px, positive = upar).
        Baseline last REST position (ya hold-enter) se aata hai aur
        release par clear NahI hota — re-grip / wrist-flicker par bhi
        lift measurement valid rehti hai jab tak object rest par na
        laut aaya ho (REST_RESET_FRAMES).
        """
        if self.lift_baseline is None:
            return 0.0
        return self.lift_baseline[1] - self.center[1]

    def effective_lift_px(self):
        """
        Pick tasks ke liye robust lift: current lift YA hold ke dauran
        ki max lift — jo bhi badi ho. Release/re-grip flicker ke baad
        bhi valid rehti hai jab tak object rest par na laut aaye
        (REST_RESET_FRAMES ke baad baseline+max dono reset ho chuke
        hote hain).
        """
        lift = self.lift_px()

        if self._rest_frames >= REST_RESET_FRAMES:
            return lift

        return max(lift, self.max_lift)

    def carry_displacement(self):
        """
        Carry/move tasks ke liye: baseline (last rest position ya
        hold-enter) se ab tak ki seedhi doori. Baseline release par
        retain hota hai — isliye carry ke dauran hold flicker hone
        par bhi poori carry distance measure hoti hai (re-grip par
        0 ho jaati thi — live test me "needed 168, moved 61" bug).
        """
        if self.lift_baseline is None:
            return 0.0
        return math.dist(self.lift_baseline, self.center)

    # ------------------------------------------------
    # REGION HELPERS
    # ------------------------------------------------

    def inside_box(self, region_box, expand=0.0):
        """Object center region box ke andar hai? (expand = fraction)"""
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
        """Recent history me kabhi region ke andar tha?"""
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
        """Object box aur region box ka horizontal overlap fraction."""
        if region_box is None:
            return 0.0

        ox1, oy1, ox2, oy2 = self.box
        rx1, ry1, rx2, ry2 = region_box

        overlap = max(0.0, min(ox2, rx2) - max(ox1, rx1))
        obj_width = max(1.0, ox2 - ox1)

        return overlap / obj_width

    def near_vertical(self, region_box, tolerance_frac=0.25):
        """
        Object ka bottom, region ke top ke paas hai? (surface par
        rakha hua maapne ke liye — table/bowl rim)
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
        Har frame call hota hai (tracker.update ke andar).
        Wrist-object proximity se hold state maintain karta hai.
        Reach = max(0.7 * torso, 0.5 * box diagonal) — camera
        distance se independent scale.
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

                # ELBOW FALLBACK: object grip karne par wrist keypoint
                # aksar occlude/drop ho jaata hai (live test me 12s tak
                # gayab). Elbow object ke paas = hold SUSTAIN evidence.
                # NOTE: hold INITIATE sirf wrist se hota hai.
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
        # REST BASELINE (lift measurement ka reference)
        # Free + stable + koi haath paas nahi — object sach me rest
        # par hai -> baseline reset. Re-grip/chhaunte ke chhote gaps
        # me baseline bana rehta hai (lift measurement toot-ti nahi).
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

                # Lift baseline: pehli baar (ya rest se uthate waqt)
                # to abhi ka center; RE-GRIP hawa me ho to purana
                # baseline RETAIN (warna lift dobara 0 ho jaati hai —
                # wahi bug jisse pick kabhi complete nahi hota tha).
                if self.lift_baseline is None:
                    self.lift_baseline = self.center
                    self.max_lift = 0.0

                if person_id is not None:
                    self.holder_history.append(person_id)

        elif self.held_by is not None:
            # Sustain: wrist YA elbow paas ho (grip me wrist occlude hota
            # hai — live-verified). Dono gayab -> release 2-frame confirm.
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
                    # RELEASE (hysteresis ke saath)
                    # NOTE: lift_baseline jaan-boojh kar RETAIN —
                    # wrist flicker ke baad re-grip par lift wahi
                    # baseline se measure hogi.
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

    Usage (camera loop me har frame par):
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
        """Same class ke sabse achhe track se match karo (IoU ya center)."""
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

            # Held / grip-hone-wala object haath ke saath tez chalta
            # hai — iska match radius relax (fast lift par track
            # re-id na ho; warna lift measurement kho jaati hai).
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
                # Held object haath me tez chalta hai — overlap kabhi
                # kabhi zero ho jaata hai; center-distance match kaafi hai.
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
        detections: object_detection.detect_objects() ki list
                    (person class filter ho jaata hai)
        persons:    pose_detection.analyze_frame() ki persons list

        Returns: active tracks list
        """
        matched_tracks = set()

        for det in detections or []:

            # Person ko object tracker me nahi rakhte
            # (wo pose pipeline track karta hai)
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

        # hand proximity + hold state (sirf visible tracks)
        for track in self.tracks:
            if track.missed == 0:
                track.update_hand_proximity(persons)

        # purane gayab tracks hatao
        self.tracks = [
            t for t in self.tracks if t.missed <= TRACK_MAX_MISSED
        ]

        return [t for t in self.tracks if t.missed == 0]

    # ------------------------------------------------
    # QUERIES
    # ------------------------------------------------

    def of_class(self, class_names, min_conf=0.35):
        """Specific classes ke visible tracks (confidence sorted)."""
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
        """Koi bhi non-person object track (pick up an object ke liye)."""
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
# DRAW HELPERS (camera.py ke liye)
# ============================================================

TRACK_COLORS = {
    True: (0, 220, 255),     # held -> yellow-ish (BGR)
    False: (255, 160, 60),   # free -> blue-ish
}


def draw_tracks(frame, tracks, region_box=None):
    """
    Tracks + optional target region(s) frame par draw karta hai.
    region_box: ek box [x1,y1,x2,y2] YA boxes ki list
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
