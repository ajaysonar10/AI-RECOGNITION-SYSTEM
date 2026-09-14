"""
task_detection.py
=================

BAS-AI • Task Verification / Task Completion System

Ye module REAL YOLO-Pose keypoints se user tasks verify karta hai.
Koi dummy/random/static values nahi — ek task tabhi COMPLETE hota hai jab:

  1. pose_detection.analyze_frame() se mile keypoints par task ka
     pose GEOMETRY match ho (shoulder/elbow/wrist/hip/knee/ankle angles),
  2. wo match TASK_CONFIRM_FRAMES consecutive (near-consecutive) frames
     tak bana rahe (temporal confirmation — false positives rokne ke liye),
  3. aur multi-step tasks me steps apni TURN par hi complete hon
     (sequential enforcement).

Pipeline (existing architecture ke andar):

    camera frame
      ↓
    pose_detection.analyze_frame()   -> persons (keypoints, activity, conf)
      ↓
    task_detection.evaluate_action() -> keypoint geometry check
      ↓
    TaskVerificationSession          -> temporal + sequential state machine
      ↓
    task status (UI) + task history

Public API:
    parse_task_text(task_text)        -> action key ya None
    detect_task(task_text, persons)   -> one-frame verification dict
    verify_single_task(...)           -> single-task session stepping
    verify_task_step(...)             -> multi-step session stepping
    TaskVerificationSession           -> sequential + temporal engine
    evaluate_action(action, persons)  -> (detected, confidence, reasons)

OBJECT TASKS (modular extension):
    Object-interaction tasks (pick up bottle, place in box, handoff,
    ...) object_tasks.py ke HANDLERS me hain. TaskVerificationSession
    unhe transparently dispatch karta hai — parse_task_text pehle
    pose actions try karta hai, phir object task library. Temporal
    confirmation + sequential lock object tasks par bhi LAGU hota hai
    (same sliding window, same LOCKED/IN PROGRESS/COMPLETED states).
    camera.py ObjectTaskContext (tracker + objects + options) banata
    hai aur process_persons(persons, object_context=...) pass karta hai.

NOTE: Ye module jaan-boojh kar sirf math/re/datetime use karta hai —
koi cv2/ultralytics import nahi, taaki ye bina camera/model ke bhi
test ho sake. (object_tasks import bhi lazy hai — pose-only use me
object module load hi nahi hota.)
"""

import math
import re
from collections import deque
from datetime import datetime


# ============================================================
# SETTINGS
# ============================================================

# Temporal confirmation: required pose itne (near-)consecutive valid
# frames tak match ho tabhi step COMPLETED ho. (Spec: 8-15 frames)
TASK_CONFIRM_FRAMES = 12

# Sliding-window temporal confirmation: required pose ko last
# (TASK_CONFIRM_FRAMES + WINDOW_EXTRA_FRAMES) frames me kam se kam
# TASK_CONFIRM_FRAMES dafa detect hona chahiye. YOLO keypoints
# threshold ke paas ek-do frame flicker karte hain — window unhe
# forgive karti hai, par pose sach me na ho to completion kabhi
# nahi hoti (false-positive safe).
WINDOW_EXTRA_FRAMES = 6

# Stand: avg knee angle iske upar + hip<knee<ankle vertical order
STAND_KNEE_ANGLE = 150.0
# Sit: avg knee angle iske neeche + hip knee-level ke paas
SIT_KNEE_ANGLE = 125.0
SIT_HIP_KNEE_RATIO = 0.6   # |hip.y - knee.y| < 0.6 * thigh length => seated

# Raise hand: wrist shoulder se itna (torso length ka fraction) upar ho
RAISE_MARGIN_RATIO = 0.2


# ============================================================
# TASK CATALOG (modular — naye tasks yahan add karo)
# ============================================================

ACTION_LABELS = {
    "stand":            "Stand",
    "sit":              "Sit Down",
    "raise_right_hand": "Raise Right Hand",
    "raise_left_hand":  "Raise Left Hand",
    "raise_both_hands": "Raise Both Hands",
    "raise_hand":       "Raise Hand",
    "walk":             "Walk",
}

# (phrase, action) — longest-phrase-first matching parse me hota hai
TASK_PHRASES = [
    ("raise your both hands", "raise_both_hands"),
    ("raise both hands",      "raise_both_hands"),
    ("both hands up",         "raise_both_hands"),
    ("lift both hands",       "raise_both_hands"),
    ("hands up",              "raise_both_hands"),
    ("raise your right hand", "raise_right_hand"),
    ("raise right hand",      "raise_right_hand"),
    ("lift right hand",       "raise_right_hand"),
    ("raise your right arm",  "raise_right_hand"),
    ("right hand up",         "raise_right_hand"),
    ("right arm up",          "raise_right_hand"),
    ("raise your left hand",  "raise_left_hand"),
    ("raise left hand",       "raise_left_hand"),
    ("lift left hand",        "raise_left_hand"),
    ("raise your left arm",   "raise_left_hand"),
    ("left hand up",          "raise_left_hand"),
    ("left arm up",           "raise_left_hand"),
    ("raise your hand",       "raise_hand"),
    ("raise one hand",        "raise_hand"),
    ("raise hand",            "raise_hand"),
    ("lift your hand",        "raise_hand"),
    ("wave your hand",        "raise_hand"),
    ("hand up",               "raise_hand"),
    ("stand up",              "stand"),
    ("be standing",           "stand"),
    ("standing",              "stand"),
    ("stand",                 "stand"),
    ("sit down",              "sit"),
    ("be seated",             "sit"),
    ("take a seat",           "sit"),
    ("sitting",               "sit"),
    ("sit",                   "sit"),
    ("move around",           "walk"),
    ("walking",               "walk"),
    ("walk",                  "walk"),
]

_TASK_PHRASES_SORTED = sorted(
    TASK_PHRASES, key=lambda item: -len(item[0])
)


def parse_task_text(task_text):
    """
    Free-text task ko canonical action key me convert karta hai.

    Returns:
        action key (str) ya None agar task samajh nahi aaya.
    """
    text = (task_text or "").lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)     # punctuation hatao
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return None

    for phrase, action in _TASK_PHRASES_SORTED:
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            return action

    return None


# ============================================================
# KEYPOINT HELPERS (pose_detection.py ke conventions ke mutabik)
# ============================================================

# Agar keypoints raw 17-array ke roop me aaye to ye index mapping
_ARRAY_INDEX = {
    "nose": 0,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


def _point_valid(point):
    """Missing/zero keypoint safe check (pose_detection jaisa hi rule)."""
    if point is None:
        return False
    try:
        if len(point) < 2:
            return False
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return False
    # YOLO missing point (0, 0) deta hai
    if x <= 0 and y <= 0:
        return False
    return True


def _get_kp(keypoints, name):
    """
    Named keypoint nikalta hai.
    keypoints: dict {name: (x, y)} (analyze_frame se) YA raw 17-list.
    Returns (x, y) tuple ya None (invalid/missing).
    """
    if keypoints is None:
        return None

    point = None

    if isinstance(keypoints, dict):
        point = keypoints.get(name)
    else:
        try:
            idx = _ARRAY_INDEX[name]
            if len(keypoints) > idx:
                point = keypoints[idx]
        except (TypeError, KeyError, IndexError):
            point = None

    if not _point_valid(point):
        return None

    return (float(point[0]), float(point[1]))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle(a, b, c):
    """A-B-C ke beech angle (degrees), B joint point."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])

    if mag_ba == 0 or mag_bc == 0:
        return None

    cos_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (mag_ba * mag_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))


def _torso_length(keypoints):
    """
    Body scale nikalta hai (shoulder-hip distance), taaki thresholds
    camera distance se independent rahen. Fallback: shoulder width.
    """
    left_shoulder = _get_kp(keypoints, "left_shoulder")
    right_shoulder = _get_kp(keypoints, "right_shoulder")
    left_hip = _get_kp(keypoints, "left_hip")
    right_hip = _get_kp(keypoints, "right_hip")

    shoulder_mid = None
    hip_mid = None

    if left_shoulder and right_shoulder:
        shoulder_mid = _mid(left_shoulder, right_shoulder)

    if left_hip and right_hip:
        hip_mid = _mid(left_hip, right_hip)

    if shoulder_mid and hip_mid:
        return max(1.0, _dist(shoulder_mid, hip_mid))

    if left_shoulder and right_shoulder:
        return max(1.0, _dist(left_shoulder, right_shoulder))

    return 100.0   # last-resort fallback (px)


def _person_confidence(person):
    """YOLO person-detection confidence (0-100), safe access."""
    value = person.get("person_confidence", 0.0) if person else 0.0
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _geometry_confidence(person, geometry_score):
    """
    Task confidence (0-100): YOLO person confidence + pose-geometry
    score (0-1) ka blend. Random/dummy value nahi — dono real
    measurements se aate hain.
    """
    person_conf = _person_confidence(person)
    base = 70.0 + 28.0 * max(0.0, min(1.0, geometry_score))
    conf = base * (person_conf / 100.0)
    return max(50.0, min(98.0, conf))


# ============================================================
# PER-FRAME DETECTORS
# Har detector: (person) -> (detected: bool, geometry_score: 0-1,
#                            reasons: [str])
# ============================================================

def _leg_data(keypoints, side):
    """Ek side ke hip/knee/ankle points + validity."""
    hip = _get_kp(keypoints, f"{side}_hip")
    knee = _get_kp(keypoints, f"{side}_knee")
    ankle = _get_kp(keypoints, f"{side}_ankle")

    valid = hip is not None and knee is not None and ankle is not None

    return hip, knee, ankle, valid


def _detect_stand(person):
    """
    Stand: hip-knee-ankle angle seedhi taang (>= 150 deg) +
    vertical order (hip upar, knee beech, ankle neeche).
    """
    keypoints = person.get("keypoints")

    if not keypoints:
        return False, 0.0, ["Person keypoints not available"]

    angles = []
    upright_legs = 0
    reasons = []

    for side in ("left", "right"):
        hip, knee, ankle, valid = _leg_data(keypoints, side)

        if not valid:
            reasons.append(f"{side.capitalize()} leg keypoints not visible")
            continue

        angle = _angle(hip, knee, ankle)

        if angle is None:
            continue

        angles.append(angle)
        reasons.append(
            f"{side.capitalize()} knee angle: {angle:.0f} deg"
        )

        if hip[1] < knee[1] < ankle[1]:
            upright_legs += 1

    if not angles:
        reasons.append(
            "Stand cannot be verified — leg keypoints not visible"
        )
        return False, 0.0, reasons

    avg_angle = sum(angles) / len(angles)

    if avg_angle >= STAND_KNEE_ANGLE and upright_legs > 0:
        score = min(1.0, 0.5 + (avg_angle - STAND_KNEE_ANGLE) / 50.0)
        reasons.insert(
            0,
            f"Legs straight (avg knee angle {avg_angle:.0f} deg "
            f">= {STAND_KNEE_ANGLE:.0f} deg), body upright"
        )
        return True, score, reasons

    reasons.append(
        f"Knee angle {avg_angle:.0f} deg below "
        f"{STAND_KNEE_ANGLE:.0f} deg threshold"
    )
    return False, 0.0, reasons


def _detect_sit(person):
    """
    Sit: knees bent (<= 125 deg) + hips knee-level ke paas
    (thigh horizontal — chair/stool par baithne ka geometry).
    """
    keypoints = person.get("keypoints")

    if not keypoints:
        return False, 0.0, ["Person keypoints not available"]

    angles = []
    seated_legs = 0
    reasons = []

    for side in ("left", "right"):
        hip, knee, ankle, valid = _leg_data(keypoints, side)

        if not valid:
            reasons.append(f"{side.capitalize()} leg keypoints not visible")
            continue

        angle = _angle(hip, knee, ankle)

        if angle is None:
            continue

        angles.append(angle)
        reasons.append(
            f"{side.capitalize()} knee angle: {angle:.0f} deg"
        )

        thigh = _dist(hip, knee)

        if thigh > 0 and abs(hip[1] - knee[1]) < SIT_HIP_KNEE_RATIO * thigh:
            seated_legs += 1
            reasons.append(
                f"{side.capitalize()} hip is at knee level (seated posture)"
            )

    if not angles:
        reasons.append(
            "Sit cannot be verified — leg keypoints not visible"
        )
        return False, 0.0, reasons

    avg_angle = sum(angles) / len(angles)

    if avg_angle <= SIT_KNEE_ANGLE and seated_legs > 0:
        score = min(1.0, 0.5 + (SIT_KNEE_ANGLE - avg_angle) / 60.0)
        reasons.insert(
            0,
            f"Knees bent (avg knee angle {avg_angle:.0f} deg "
            f"<= {SIT_KNEE_ANGLE:.0f} deg), hips at knee level"
        )
        return True, score, reasons

    if avg_angle > SIT_KNEE_ANGLE:
        reasons.append(
            f"Knee angle {avg_angle:.0f} deg above "
            f"{SIT_KNEE_ANGLE:.0f} deg — legs not bent"
        )
    if seated_legs == 0:
        reasons.append(
            "Hips are not at knee level — posture is not seated"
        )

    return False, 0.0, reasons


def _raise_check(keypoints, side, torso):
    """
    Ek side ka raise check.
    Returns (raised: bool, excess_ratio: 0-1+).
    Wrist shoulder se RAISE_MARGIN_RATIO * torso upar ho.
    """
    shoulder = _get_kp(keypoints, f"{side}_shoulder")
    wrist = _get_kp(keypoints, f"{side}_wrist")

    if shoulder is None or wrist is None:
        return False, 0.0

    margin = RAISE_MARGIN_RATIO * torso
    excess = (shoulder[1] - margin) - wrist[1]   # >0 => wrist upar hai

    if excess <= 0:
        return False, 0.0

    return True, excess / max(1.0, torso)


def _detect_raise_hand(person, which="either"):
    """
    Raise Right/Left/Both/Any hand:
    wrist ko shoulder/head se compare karta hai (torso-length margin ke
    saath, camera distance independent).
    """
    keypoints = person.get("keypoints")

    if not keypoints:
        return False, 0.0, ["Person keypoints not available"]

    torso = _torso_length(keypoints)

    right_raised, right_ratio = _raise_check(keypoints, "right", torso)
    left_raised, left_ratio = _raise_check(keypoints, "left", torso)

    sides = {
        "right":  (right_raised, right_ratio,
                   "Right wrist is above right shoulder"),
        "left":   (left_raised, left_ratio,
                   "Left wrist is above left shoulder"),
        "both":   (right_raised and left_raised,
                   min(right_ratio, left_ratio),
                   "Both wrists are above their shoulders"),
        "either": (right_raised or left_raised,
                   max(right_ratio, left_ratio),
                   "A wrist is above the shoulder"),
    }

    detected, ratio, reason = sides[which]

    if detected:
        score = min(1.0, 0.5 + ratio / 0.7)
        return True, score, [
            reason,
            f"Wrist height margin: {ratio * 100:.0f}% of torso length"
        ]

    which_label = {
        "right": "Right", "left": "Left",
        "both": "Both", "either": "Either",
    }[which]

    return False, 0.0, [
        f"{which_label} hand is not raised "
        f"(wrist not above shoulder + {RAISE_MARGIN_RATIO * 100:.0f}% "
        f"torso margin)"
    ]


def _detect_raise_right_hand(person):
    return _detect_raise_hand(person, "right")


def _detect_raise_left_hand(person):
    return _detect_raise_hand(person, "left")


def _detect_raise_both_hands(person):
    return _detect_raise_hand(person, "both")


def _detect_raise_either_hand(person):
    return _detect_raise_hand(person, "either")


def _detect_walk(person):
    """
    Walk: pose_detection ke motion tracker (hip/ankle movement across
    frames + displacement + direction consistency) ka confirmed
    'Walking' activity label use karta hai — real pipeline reuse.
    """
    activity = person.get("activity")

    if activity == "Walking":
        activity_conf = 0.0
        try:
            activity_conf = float(person.get("confidence", 0.0))
        except (TypeError, ValueError):
            activity_conf = 0.0

        score = min(1.0, max(0.0, (activity_conf - 75.0) / 20.0))

        reasons = [
            "Body center is consistently moving — "
            "Walking confirmed by motion tracker"
        ]
        reasons.extend(list(person.get("justification", []))[:2])

        return True, score, reasons

    return False, 0.0, [
        f"Detected activity is '{activity or 'None'}' — "
        f"Walking not confirmed by motion tracker"
    ]


# action -> detector (naya task = naya detector + catalog entry)
_ACTION_DETECTORS = {
    "stand":            _detect_stand,
    "sit":              _detect_sit,
    "raise_right_hand": _detect_raise_right_hand,
    "raise_left_hand":  _detect_raise_left_hand,
    "raise_both_hands": _detect_raise_both_hands,
    "raise_hand":       _detect_raise_either_hand,
    "walk":             _detect_walk,
}


# ============================================================
# ONE-FRAME EVALUATION
# ============================================================

def evaluate_action(action_key, persons):
    """
    Ek frame (persons list) par action verify karta hai.

    Multi-person: sabse strong matching person jeetta hai.
    Missing/low-confidence keypoints safely handle hote hain
    (detect = False + honest reason).

    Returns:
        (detected: bool, confidence: float 0-100, reasons: [str])
    """
    if action_key not in _ACTION_DETECTORS:
        return False, 0.0, [f"Unsupported task action: {action_key}"]

    if not persons:
        return False, 0.0, ["No person detected in frame"]

    # Defensive: agar caller galti se single person dict bhej de
    if isinstance(persons, dict):
        persons = [persons]

    best = (False, 0.0, ["Person detected, task pose not matched"])

    for person in persons:
        try:
            detected, score, reasons = _ACTION_DETECTORS[action_key](person)
        except Exception as exc:              # kisi bhi keypoint edge case par crash nahi
            detected, score, reasons = False, 0.0, [
                f"Detection error handled safely: {exc}"
            ]

        if not detected:
            continue

        confidence = _geometry_confidence(person, score)

        if confidence > best[1]:
            best = (True, confidence, reasons)

    return best


def detect_task(task_text, persons):
    """
    Single-frame task verification (task_eval-style API).

    Returns:
        {
            "task": original text,
            "action": action key ya None,
            "label": human label ya None,
            "detected": bool,
            "confidence": float 0-100,
            "reasons": [str],
        }
    """
    action = parse_task_text(task_text)

    if action is None:
        supported = ", ".join(ACTION_LABELS.values())
        return {
            "task": task_text,
            "action": None,
            "label": None,
            "detected": False,
            "confidence": 0.0,
            "reasons": [
                "Task not recognized. Supported tasks: " + supported
            ],
        }

    detected, confidence, reasons = evaluate_action(action, persons)

    return {
        "task": task_text,
        "action": action,
        "label": ACTION_LABELS[action],
        "detected": detected,
        "confidence": round(confidence, 1),
        "reasons": reasons,
    }


# ============================================================
# SEQUENTIAL + TEMPORAL TASK SESSION
# ============================================================

STATE_LOCKED = "LOCKED"
STATE_IN_PROGRESS = "IN PROGRESS"
STATE_COMPLETED = "COMPLETED"
STATE_NOT_COMPLETED = "NOT COMPLETED"


class TaskVerificationSession:
    """
    Ek task run ka poori state (Streamlit session_state me store hoti hai,
    isliye har frame par reset nahi hoti).

    - Single task  -> steps list me sirf ek step (mode = "single")
    - Multi-step   -> sequential enforcement: sirf current step verify
                      hota hai; future steps LOCKED rehte hain.
    - Temporal     -> current step ka pose sliding window me
                      TASK_CONFIRM_FRAMES dafa match ho tabhi COMPLETE
                      (ek-do flicker frame completion nahi rokte).
    """

    def __init__(self, task_texts, required_frames=TASK_CONFIRM_FRAMES):
        self.required_frames = max(4, int(required_frames))

        self.steps = []

        for raw in task_texts:
            text = (raw or "").strip()
            action = parse_task_text(text)
            unsupported_reason = None

            if action is None:
                # Object-interaction task library (lazy import — pose-only
                # installations me ye module load hi nahi hota).
                try:
                    from object_tasks import (
                        parse_object_task,
                        UNSUPPORTED_TASKS,
                    )
                    obj_key, detail = parse_object_task(text)
                except ImportError:
                    obj_key, detail = None, {}

                if obj_key is not None and detail.get("unsupported"):
                    # Honest rejection — unreliable rule nahi banayenge
                    unsupported_reason = UNSUPPORTED_TASKS[obj_key]["reason"]
                elif obj_key is not None:
                    action = obj_key

            self.steps.append({
                "text": text,
                "action": action,
                "unsupported_reason": unsupported_reason,
                "state": STATE_NOT_COMPLETED,
                "confidence": 0.0,
                "completed_at": None,
            })

        self.mode = "single" if len(self.steps) == 1 else "multi"
        self.current_index = 0
        self.streak = 0
        self.best_streak_confidence = 0.0

        # Sliding window (temporal confirmation) — flicker-tolerant
        self.window_frames = max(
            self.required_frames + WINDOW_EXTRA_FRAMES, 18
        )
        self.recent = deque(maxlen=self.window_frames)
        self.all_completed = False
        self.started_at = datetime.now()

        # Fresh start: step 1 IN PROGRESS, baaki LOCKED
        if self.steps:
            self.steps[0]["state"] = STATE_IN_PROGRESS
            for step in self.steps[1:]:
                step["state"] = STATE_LOCKED

        self.last_result = self._empty_result()

    # ------------------------------------------------
    # STATE HELPERS
    # ------------------------------------------------

    def _empty_result(self):
        return {
            "active": False,
            "all_completed": self.all_completed,
            "step_completed": False,
            "completed_step_number": None,
            "current_step": None,
            "current_index": self.current_index,
            "state": STATE_NOT_COMPLETED,
            "detected": False,
            "confidence": 0.0,
            "streak": self.streak,
            "required_frames": self.required_frames,
            "reasons": [],
            "events": [],
            "progress": self.progress(),
        }

    def current_step(self):
        if 0 <= self.current_index < len(self.steps):
            return self.steps[self.current_index]
        return None

    def progress(self):
        """0.0 - 1.0 (completed steps / total steps)."""
        if not self.steps:
            return 0.0
        done = sum(
            1 for s in self.steps if s["state"] == STATE_COMPLETED
        )
        return done / len(self.steps)

    def completed_count(self):
        return sum(
            1 for s in self.steps if s["state"] == STATE_COMPLETED
        )

    def _step_label(self, step):
        """Step ka human label (pose ya object task library se)."""
        action = step["action"]

        if action is None:
            return "Unrecognized"

        label = ACTION_LABELS.get(action)

        if label is None:
            try:
                from object_tasks import TASK_LABELS
                label = TASK_LABELS.get(action)
            except ImportError:
                label = None

        return label or "Unrecognized"

    def steps_view(self):
        """UI ke liye steps ki snapshot (icon ke saath)."""
        view = []

        for i, step in enumerate(self.steps):
            state = step["state"]

            if state == STATE_COMPLETED:
                icon = "✓"
            elif state == STATE_IN_PROGRESS:
                icon = "→"
            else:
                icon = "○"

            view.append({
                "number": i + 1,
                "text": step["text"],
                "label": self._step_label(step),
                "state": state,
                "icon": icon,
                "confidence": step["confidence"],
            })

        return view

    # ------------------------------------------------
    # RESET (spec: sab steps NOT COMPLETED, step 1 IN PROGRESS)
    # ------------------------------------------------

    def reset(self):
        self.current_index = 0
        self.streak = 0
        self.recent.clear()
        self.best_streak_confidence = 0.0
        self.all_completed = False
        self.last_result = self._empty_result()

        for i, step in enumerate(self.steps):
            step["state"] = (
                STATE_IN_PROGRESS if i == 0 else STATE_NOT_COMPLETED
            )
            step["confidence"] = 0.0
            step["completed_at"] = None

    # ------------------------------------------------
    # MAIN: har camera frame par call karo
    # ------------------------------------------------

    def process_persons(self, persons, object_context=None):
        """
        Ek camera frame process karta hai (real YOLO-Pose persons list).

        Returns: result dict (UI + history ke liye) —
            step_completed, all_completed, events (history records),
            streak, detected, confidence, reasons, progress, ...
        """
        # Run poora ho chuka hai ya steps hain hi nahi
        if self.all_completed or not self.steps:
            self.last_result = self._empty_result()
            return self.last_result

        step = self.current_step()

        if step is None or step["action"] is None:
            result = self._empty_result()

            if step is not None and step.get("unsupported_reason"):
                result["reasons"] = [
                    "This task is marked UNSUPPORTED: "
                    + step["unsupported_reason"]
                ]
            else:
                result["reasons"] = [
                    "Current task text is not recognized — "
                    "use New Task to re-enter it."
                ]

            self.last_result = result
            return result

        # ------------------------------------------------
        # DISPATCH: object task ya pose task?
        # Object bridge (object_tasks.check_object_task_step) sirf
        # tab non-None deta hai jab current step OBJECT task ho.
        # Temporal window/completion ka owner neeche wala shared code
        # hi hai — dono paths ke liye identical.
        # ------------------------------------------------

        object_frame = None

        if object_context is not None:
            try:
                from object_tasks import check_object_task_step
                object_frame = check_object_task_step(
                    session=self, context=object_context
                )
            except Exception as exc:
                object_frame = {
                    "detected": False,
                    "confidence": 0.0,
                    "reasons": [
                        f"Object task verification error: {exc}"
                    ],
                }

        if object_frame is not None:
            detected = object_frame["detected"]
            confidence = object_frame["confidence"]
            reasons = object_frame["reasons"]
        else:
            detected, confidence, reasons = evaluate_action(
                step["action"], persons
            )

        # --------------------------------------------
        # TEMPORAL CONFIRMATION (sliding window)
        # Required pose ko window ke andar required_frames dafa
        # detect hona chahiye. Ek-do flicker frame (YOLO threshold
        # ke paas wiggle karta hai) confirm hona nahi rokta, lekin
        # pose sach me na ho to match-count badhta hi nahi —
        # false positive safe.
        # --------------------------------------------

        if persons:
            self.recent.append(bool(detected))

            if detected:
                self.best_streak_confidence = max(
                    self.best_streak_confidence, confidence
                )
        # Person absent frame: window me count NAHI hota (hold)

        self.streak = sum(self.recent)

        events = []
        completed_step_number = None

        # --------------------------------------------
        # STEP COMPLETE?
        # --------------------------------------------

        if self.streak >= self.required_frames:

            step["state"] = STATE_COMPLETED
            step["confidence"] = round(self.best_streak_confidence, 1)
            step["completed_at"] = datetime.now()

            completed_step_number = self.current_index + 1
            completed_at_time = step["completed_at"]

            events.append({
                "date": completed_at_time.strftime("%d %b %Y"),
                "time": completed_at_time.strftime("%H:%M:%S"),
                "task": step["text"],
                "step": (
                    f"Step {completed_step_number}"
                    if self.mode == "multi" else "Single Task"
                ),
                "status": "Completed",
                "confidence": step["confidence"],
            })

            self.streak = 0
            self.recent.clear()
            self.best_streak_confidence = 0.0
            self.current_index += 1

            if self.current_index >= len(self.steps):
                # ------------------------------------
                # ALL TASKS COMPLETED
                # ------------------------------------
                self.all_completed = True

                avg_conf = (
                    sum(s["confidence"] for s in self.steps)
                    / len(self.steps)
                )

                events.append({
                    "date": completed_at_time.strftime("%d %b %Y"),
                    "time": completed_at_time.strftime("%H:%M:%S"),
                    "task": (
                        "All steps" if self.mode == "multi"
                        else step["text"]
                    ),
                    "step": "-",
                    "status": "All Tasks Completed",
                    "confidence": round(avg_conf, 1),
                })
            else:
                # Next step ab IN PROGRESS (sequential unlock)
                self.steps[self.current_index]["state"] = (
                    STATE_IN_PROGRESS
                )

        # --------------------------------------------
        # RESULT
        # --------------------------------------------

        active_step = self.current_step()

        result = {
            "active": active_step is not None,
            "all_completed": self.all_completed,
            "step_completed": completed_step_number is not None,
            "completed_step_number": completed_step_number,
            "current_step": (
                active_step["text"] if active_step else None
            ),
            "current_index": self.current_index,
            "state": (
                active_step["state"] if active_step
                else STATE_COMPLETED
            ),
            "detected": detected,
            "confidence": round(confidence, 1),
            "streak": self.streak,
            "required_frames": self.required_frames,
            "reasons": reasons,
            "events": events,
            "progress": self.progress(),
        }

        self.last_result = result
        return result


# ============================================================
# CONVENIENCE WRAPPERS (spec ke naam ke mutabik)
# ============================================================

def verify_task_step(session, persons, object_context=None):
    """Multi-step session ka ek frame process karo (pose ya object task)."""
    return session.process_persons(
        persons, object_context=object_context
    )


def verify_single_task(task_text, persons, session=None):
    """
    Single task verification.
    session Streamlit session_state me rakha hota hai — agar None hai
    to naya ban jaata hai (temporal confirmation ke saath).
    """
    if session is None:
        session = TaskVerificationSession([task_text])

    result = session.process_persons(persons)

    return session, result
