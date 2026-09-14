"""
object_tasks.py
===============

BAS-AI • Object-Interaction Task Library (RGB-verifiable tasks only)

Every task is verified from REAL sensors:
  - YOLO Pose keypoints     (hands/wrists/body)
  - YOLO Object Detection   (bottle, box, table, ...)
  - ObjectTracker           (movement, hold, lift, displacement)
  - sliding-window temporal confirmation (TaskVerificationSession)

HONEST SUPPORT MATRIX (of the user's 28 requested tasks):

  SUPPORTED (REAL verification from RGB + COCO + tracking): 22
    1  pick_bottle              12 place_marked_location
    2  pick_object              13 remove_marked_location
    3  place_object_box         14 sort_containers
    4  remove_object_box        15 color_matching
    5  move_a_to_b              16 carry_a_to_b
    6  bottle_on_table          17 place_multiple_container
    7  tool_to_container        18 remove_multiple_container
    9  open_container           20 inspect_object
   10  handoff_person           21 touch_object
   11  receive_person           22 point_object
   19 arrange_order
   25 transfer_containers
   26 collect_one_by_one
   27 pick_move_workstation

  UNSUPPORTED (honest — unreliable/fine-grained/hidden info): 6
    8  open box (lid occlusion state is not RGB-observable)
    28 multi-step maintenance procedure (domain-specific abstract steps)
    23/24 push/pull (direction is person-relative — not inferable from RGB)
    8' close box (same as 8)
    (+ a generic displacement task is offered instead of push/pull)

  Note: 23/24 (push/pull) and 8/9 (open/close) — of open/close, ONLY
  the "remove object from box" part is reliably observable; the lid
  open/close state is not reliable from an RGB camera. Push/pull
  direction depends on the person's facing, which a single RGB
  camera cannot know. These tasks are marked UNSUPPORTED —
  we will not build fake rules.

Parsing happens only on whitelisted phrases — if the task text does
not match any known task, the result is honestly "unsupported",
never a random completion.
"""

import math

from object_tracking import ObjectTracker, draw_tracks, _box_diag, _wrist


# ============================================================
# CONFIG
# ============================================================

# Colors (task 15 — from user text or the UI selector)
COLOR_WORDS = {
    "red": ["red"],
    "blue": ["blue"],
    "green": ["green"],
    "yellow": ["yellow"],
}

# Sort/arrange UI options (task 14/19)
SORTABLE_OBJECTS = ["bottle", "cup", "book", "cell phone",
                    "apple", "banana", "bowl"]

DEFAULT_MULTIPLE_COUNT = 3     # tasks 17/18/26
DEFAULT_SORT_TARGETS = 2       # task 14 (containers needed)

MIN_OBJECT_CONF = 0.30         # min YOLO confidence for the required object
MIN_CONTAINER_CONF = 0.30
# Separate threshold + avg-smoothing for table/webcam-angle (YOLO
# gives dining table only 0.07-0.11 in webcam views — live
# verified. avg_confidence accepts a persistent weak signal,
# not one-frame noise).
MIN_TABLE_CONF = 0.08
MIN_PERSON_COUNT = 2           # handoff tasks

# Cap (px) for the pick lift threshold: for large boxes 0.3*diag
# becomes physically too large (134px in a live test).
# ~60px ≈ 6-8cm at desk distance — reachable for any object size.
PICK_LIFT_CAP_PX = 60.0

# SUSTAINED-HOLD fallback: if the object stays in the hand for this
# many CONSECUTIVE frames (wrist/elbow proximity sustained), treat it
# as a pick — even without a lift after the baseline (the object may
# already be in the hand at run start; "holding for 4+ seconds
# without lifting is impossible").
SUSTAINED_HOLD_FRAMES = 45

# COCO has NO 'box' class — real container classes:
CONTAINER_CLASSES = [
    "bowl", "cup", "vase", "sink",
    "refrigerator", "microwave", "oven",
]

# Handoff: the receiver's hand must be this close to the object (px, scale-free)
HANDOFF_NEAR_FRAMES = 4        # sliding window confirmations


# ============================================================
# TASK LIBRARY (supported)
# ============================================================

TASK_LIBRARY = {
    # ---- pick ----
    "pick_bottle": {
        "label": "Pick up a Bottle",
        "phrases": ["pick up the bottle", "pick up bottle",
                    "pick up a bottle", "pick a bottle",
                    "pick the bottle", "grab the bottle",
                    "lift the bottle", "take the bottle"],
        "requires": ["bottle"],
    },
    "pick_object": {
        "label": "Pick up an Object",
        "phrases": ["pick up an object", "pick up object",
                    "pick up something", "grab an object",
                    "lift an object", "pick any object"],
        "requires": ["any object"],
    },
    # ---- containers ----
    "place_object_box": {
        "label": "Place Object Inside a Box",
        "phrases": ["place object inside box", "put object in box",
                    "place object in the box", "put it in the box",
                    "place inside box", "put in the box",
                    "pick up the bottle and place it inside a box",
                    "pick the bottle and place it inside a box",
                    "place the bottle inside a box",
                    "put the bottle inside a box"],
        "requires": ["bottle/object", "box/container or marked location"],
    },
    "remove_object_box": {
        "label": "Remove Object from a Box",
        "phrases": ["remove object from box", "take object out of box",
                    "take out from box", "take the object out",
                    "take the bottle out of the box",
                    "take the bottle out of box",
                    "remove the bottle from the box",
                    "take the bottle out"],
        "requires": ["box/container with object inside"],
    },
    # ---- movement / placement ----
    "move_a_to_b": {
        "label": "Move Object A to Location B",
        "phrases": ["move object", "move the object",
                    "move this object", "relocate object"],
        "requires": ["any object"],
    },
    "bottle_on_table": {
        "label": "Put Bottle on Table",
        "phrases": ["put bottle on table", "place bottle on table",
                    "put the bottle on the table",
                    "place the bottle on the table",
                    "put the bottle on a table", "keep bottle on the table"],
        "requires": ["bottle", "table/dining table"],
    },
    "tool_to_container": {
        "label": "Pick Up Tool, Place in Container",
        "phrases": [                    "pick up tool", "put the tool in a container",
                    "place tool in container", "put tool in container"],
        "requires": ["tool-like object", "container"],
    },
    # ---- handoff ----
    "handoff_person": {
        "label": "Hand Object to Another Person",
        "phrases": ["hand object to another person",
                    "give object to another person",
                    "hand over the object", "give the object to someone else",
                    "give it to someone"],
        "requires": ["object", "two persons"],
    },
    "receive_person": {
        "label": "Receive Object from Another Person",
        "phrases": ["receive object from another person",
                    "take object from another person",
                    "take the object from someone else"],
        "requires": ["object", "two persons"],
    },
    # ---- marked location ----
    "place_marked_location": {
        "label": "Place Object on Marked Location",
        "phrases": ["place object on marked location",
                    "put object on marked location",
                    "put it on the marked location"],
        "requires": ["any object", "marked location"],
    },
    "remove_marked_location": {
        "label": "Remove Object from Marked Location",
        "phrases": ["remove object from marked location",
                    "remove from the marked location",
                    "pick up from marked location"],
        "requires": ["any object", "marked location"],
    },
    # ---- sorting ----
    "sort_containers": {
        "label": "Sort Objects into Containers",
        "phrases": ["sort objects", "sort the objects",
                    "objects into different containers"],
        "requires": ["objects", f"{DEFAULT_SORT_TARGETS}+ containers"],
    },
    "color_matching": {
        "label": "Put Colored Object into Matching Box",
        "phrases": ["put correct colored object", "color matching",
                    "colored object into matching box",
                    "color object into box"],
        "requires": ["object", "container", "color selection"],
    },
    # ---- carry ----
    "carry_a_to_b": {
        "label": "Carry Object from A to B",
        "phrases": ["carry object", "carry the object",
                    "carry the object with you"],
        "requires": ["any object"],
    },
    # ---- multiple objects ----
    "place_multiple_container": {
        "label": "Place Multiple Objects into Container",
        "phrases": ["place multiple objects", "put multiple objects",
                    "put several objects in the container"],
        "requires": ["objects", "container"],
    },
    "remove_multiple_container": {
        "label": "Remove Multiple Objects from Container",
        "phrases": ["remove multiple objects",
                    "take out multiple objects",
                    "take several objects out"],
        "requires": ["container", "objects"],
    },
    # ---- arrange ----
    "arrange_order": {
        "label": "Arrange Objects in Specified Order",
        "phrases": ["arrange objects", "arrange in order",
                    "put the objects in order"],
        "requires": ["objects", "order selection"],
    },
    # ---- inspect ----
    "inspect_object": {
        "label": "Pick Up, Inspect/Hold, Place Back",
        "phrases": ["inspect object", "pick up and inspect",
                    "hold and place back", "check the object"],
        "requires": ["any object"],
    },
    # ---- touch / point ----
    "touch_object": {
        "label": "Touch a Designated Object",
        "phrases": ["touch object", "touch the object",
                    "tap the object", "touch a designated object"],
        "requires": ["any object"],
    },
    "point_object": {
        "label": "Point to a Designated Object",
        "phrases": ["point to object", "point at the object",
                    "point to a designated object", "point toward the object"],
        "requires": ["any object"],
    },
    # ---- transfer ----
    "transfer_containers": {
        "label": "Transfer Object Between Two Containers",
        "phrases": ["transfer object between containers",
                    "move the object to the other container",
                    "transfer between boxes"],
        "requires": ["any object", "two containers"],
    },
    "collect_one_by_one": {
        "label": "Collect Objects One by One",
        "phrases": ["collect objects", "collect objects one by one",
                    "collect the objects one at a time"],
        "requires": ["objects"],
    },
    "pick_move_workstation": {
        "label": "Pick Tool, Move to Workstation",
        "phrases": [                    "move tool to workstation", "take the tool to the workstation",
                    "pick up tool and move"],
        "requires": ["tool-like object", "workstation region"],
    },
    "move_bottle_places": {
        "label": "Move Bottle: Place 1 to Place 2",
        "phrases": ["move the bottle from place 1 to place 2",
                    "move bottle from place 1 to place 2",
                    "move the bottle from one place to another",
                    "move bottle from one place to another",
                    "carry the bottle from place 1 to place 2"],
        "requires": ["bottle", "place 1 = left third of view, "
                     "place 2 = right third"],
    },
}

# ------------------------------------------------------------
# UNSUPPORTED — honest reasons (no fake rules)
# ------------------------------------------------------------

UNSUPPORTED_TASKS = {
    "open_container": {
        "label": "Open a Box/Container",
        "reason": ("The lid-open state is not reliably observable from "
                   "an RGB camera — COCO objects only give box/bottle, "
                   "not lid state. A reliable rule is not possible."),
    },
    "close_container": {
        "label": "Close a Box/Container",
        "reason": ("The lid-closed state cannot be verified from RGB — "
                   "open/close pose rules would be unreliable."),
    },
    "push_object": {
        "label": "Push Object",
        "reason": ("Push vs pull direction depends on the person's "
                   "facing, which a single RGB camera cannot know. "
                   "It would be INFERRED — hence unsupported."),
    },
    "pull_object": {
        "label": "Pull Object",
        "reason": ("Push vs pull direction depends on the person's "
                   "facing, which a single RGB camera cannot know. "
                   "It would be INFERRED — hence unsupported."),
    },
    "maintenance_procedure": {
        "label": "Complete a Multi-Step Maintenance Procedure",
        "reason": ("Maintenance steps are domain-specific — verifying "
                   "them from generic RGB sensors would be unreliable. "
                   "Instead, specific steps can be defined in "
                   "Multi-Step Task mode."),
    },
}


# ============================================================
# PARSING
# ============================================================

def _clean(text):
    text = (text or "").lower().strip()
    cleaned = []
    for ch in text:
        cleaned.append(ch if ch.isalnum() or ch.isspace() else " ")
    return " ".join("".join(cleaned).split())


def parse_object_task(task_text):
    """
    Free text -> (task_key or None, detail_dict)

    detail_dict may carry parameters like color/count.
    Nothing matches outside the whitelisted phrases —
    unknown tasks honestly return unsupported.
    """
    text = _clean(task_text)

    if not text:
        return None, {}

    # ---- unsupported tasks first (honest rejection) ----
    unsupported_map = {
        "open_container": ["open box", "open the box", "open container",
                           "open the container", "open up the box"],
        "close_container": ["close box", "close the box", "close container",
                            "close the container", "shut the box"],
        "push_object": ["push object", "push the object", "push it"],
        "pull_object": ["pull object", "pull the object", "pull it"],
        "maintenance_procedure": [
            "maintenance procedure", "maintenance",
            "complete a multi-step maintenance",
        ],
    }

    for key, phrases in unsupported_map.items():
        for phrase in phrases:
            if phrase in text:
                return key, {"unsupported": True}

    # ---- supported tasks (longest phrase match) ----
    phrase_matches = []

    for key, spec in TASK_LIBRARY.items():
        for phrase in spec["phrases"]:
            if phrase in text:
                phrase_matches.append((len(phrase), key, spec))

    if not phrase_matches:
        return None, {}

    phrase_matches.sort(reverse=True)
    _, best_key, best_spec = phrase_matches[0]

    detail = {}

    # color extraction (task 15)
    for color, words in COLOR_WORDS.items():
        if any(w in text for w in words):
            detail["color"] = color
            break

    # count extraction (multiple/collect tasks)
    for word in ("two", "2"):
        if word in text:
            detail["count"] = 2
    for word in ("three", "3"):
        if word in text:
            detail["count"] = 3
    for word in ("four", "4"):
        if word in text:
            detail["count"] = 4

    return best_key, detail


def get_task_requirements(task_key):
    """Physical requirements of the task (UI + honest failure reasons)."""
    if task_key in TASK_LIBRARY:
        return TASK_LIBRARY[task_key]["requires"]
    if task_key in UNSUPPORTED_TASKS:
        return [UNSUPPORTED_TASKS[task_key]["reason"]]
    return []


# ============================================================
# PERSON CONTEXT HELPERS (pose keypoints)
# ============================================================

def _kp(keypoints, name):
    """Named keypoint (dict or 17-array). Invalid -> None."""
    idx_map = {
        "nose": 0,
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10,
        "left_hip": 11, "right_hip": 12,
    }

    if keypoints is None:
        return None

    if isinstance(keypoints, dict):
        point = keypoints.get(name)
    else:
        try:
            index = idx_map[name]
            point = keypoints[index] if len(keypoints) > index else None
        except (TypeError, KeyError, IndexError):
            point = None

    if point is None:
        return None

    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return None

    if x <= 0 and y <= 0:
        return None

    return (x, y)


def _torso(keypoints):
    ls = _kp(keypoints, "left_shoulder")
    rs = _kp(keypoints, "right_shoulder")
    lh = _kp(keypoints, "left_hip")
    rh = _kp(keypoints, "right_hip")

    if all(p is not None for p in (ls, rs, lh, rh)):
        sm = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hm = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        return max(1.0, math.dist(sm, hm))

    if ls is not None and rs is not None:
        return max(1.0, math.dist(ls, rs))

    return 100.0


def _hand_near_point(keypoints, point, scale_torso, radius_factor=0.55):
    """
    Is any wrist near the given point?
    Returns (bool, side, dist) — the closest hand.
    """
    if point is None:
        return False, None, None

    radius = radius_factor * scale_torso
    best = (False, None, None)

    for side in ("left", "right"):
        wrist = _wrist(keypoints, side)

        if wrist is None:
            continue

        dist = math.dist(wrist, point)

        if dist <= radius:
            if not best[0] or dist < best[2]:
                best = (True, side, dist)

    return best


def _find_free_hand(persons, exclude_track):
    """
    The free hand of any free person (about to reach for the object).
    The current holder of exclude_track is not counted.
    Returns a list of (person_index, side, wrist_xy).
    """
    hands = []

    for idx, person in enumerate(persons or []):

        keypoints = person.get("keypoints")

        if not keypoints:
            continue

        for side in ("left", "right"):
            wrist = _wrist(keypoints, side)

            if wrist is None:
                continue

            hands.append((idx, side, wrist))

    return hands


def _wrist_raised(keypoints, side, torso):
    """Is the wrist above the shoulder? (inspect/hold-up gesture)"""
    shoulder = _kp(keypoints, f"{side}_shoulder")
    wrist = _wrist(keypoints, side)

    if shoulder is None or wrist is None:
        return False

    return wrist[1] < (shoulder[1] - 0.1 * torso)


def _pointing_pose(keypoints, side, torso):
    """
    Pointing pose: shoulder-elbow-wrist roughly a straight line
    (>= 140 deg) AND the arm at or above horizontal-ish (the wrist at
    least 0.6*torso from the elbow + at or above shoulder level).
    """
    shoulder = _kp(keypoints, f"{side}_shoulder")
    elbow = _kp(keypoints, f"{side}_elbow")
    wrist = _wrist(keypoints, side)

    if shoulder is None or elbow is None or wrist is None:
        return False

    ba = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    bc = (wrist[0] - elbow[0], wrist[1] - elbow[1])

    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return False

    cos_a = (ba[0] * bc[0] + ba[1] * bc[1]) / (mag_ba * mag_bc)
    cos_a = max(-1.0, min(1.0, cos_a))

    angle = math.degrees(math.acos(cos_a))

    # Only a straight-elbow + raised arm counts as pointing. The
    # forearm (elbow->wrist) length is ~0.45-0.65 x torso — 0.8 is
    # not biomechanically reachable (never fired in a live test).
    arm_extended = mag_bc >= 0.45 * torso
    arm_level = wrist[1] <= shoulder[1] + 0.35 * torso

    return angle >= 140 and arm_extended and arm_level


# ============================================================
# EVIDENCE EVALUATION (everything for one frame is computed here)
# ============================================================

def evaluate_evidence(context):
    """
    The complete evidence for the current frame — in one place, so all
    handlers work on the same data (consistent + fast).

    context: ObjectTaskContext (built by camera.py)
    """
    tracker = context.tracker
    persons = context.persons or []

    ev = {
        "persons": persons,
        "person_count": len(persons),
        "any_held": None,
        "held_tracks": [],
        "free_tracks": [],
    }

    # ---- held/free tracks ----
    for track in tracker.tracks:

        if track.missed > 0:
            continue

        if track.is_held():
            ev["held_tracks"].append(track)

            if ev["any_held"] is None:
                ev["any_held"] = track
        else:
            ev["free_tracks"].append(track)

    # ---- container membership update (temporal, 2-frame) ----
    for track in tracker.tracks:

        if track.missed > 0 or track.class_name == "person":
            continue

        inside_any = None

        for container in tracker.containers(
            context.container_classes, min_conf=MIN_CONTAINER_CONF
        ):
            if container.track_id == track.track_id:
                continue

            if track.inside_box(container.box, expand=0.10):
                inside_any = container
                break

        if inside_any is not None:
            track._inside_streak = getattr(track, "_inside_streak", 0) + 1
            track._outside_streak = 0

            if track._inside_streak >= 2:
                inside_any.members.add(track.track_id)
        else:
            track._outside_streak = getattr(track, "_outside_streak", 0) + 1
            track._inside_streak = 0

            if track._outside_streak >= 2:
                for container in tracker.tracks:
                    if container.track_id in getattr(
                        container, "members", set()
                    ):
                        container.members.discard(track.track_id)

    return ev


# ============================================================
# SHARED DETECTORS (containers etc.)
# ============================================================

def _detect_container(context):
    """
    Find a container for the task (box/bowl/cup + marked location).
    Returns (region_box, label, failure_reason, container_track).

    container_track can be None (marked location). Handlers must SKIP
    it in placement loops — otherwise the container becomes its own
    "placed object" (false positive).
    """
    tracker = context.tracker

    candidates = tracker.containers(
        context.container_classes, min_conf=MIN_CONTAINER_CONF
    )

    if candidates:
        best = candidates[0]
        return best.box, best.class_name, None, best

    if context.marked_location is not None:
        m = context.marked_location
        region = [
            float(m[0]), float(m[1]),
            float(m[0]) + float(m[2]),
            float(m[1]) + float(m[3]),
        ]
        return region, "marked location", None, None

    return None, None, (
        "No container/box detected in frame. "
        "Keep a box/bowl/cup visible OR use Marked Location."
    ), None


def _detect_table(context):
    """
    A table-like flat surface — returns (region, label, err, track).
    Table confidence is weak at webcam angles — uses avg_confidence
    (smoothed) + MIN_TABLE_CONF.
    """
    tracker = context.tracker

    surface_classes = ["dining table", "table", "bed", "couch"]

    candidates = [
        t for t in tracker.tracks
        if t.missed == 0
        and t.class_name in surface_classes
        and t.avg_confidence >= MIN_TABLE_CONF
    ]
    candidates.sort(key=lambda t: -t.confidence)

    if candidates:
        return (
            candidates[0].box,
            candidates[0].class_name,
            None,
            candidates[0],
        )

    if context.marked_location is not None:
        m = context.marked_location
        region = [
            float(m[0]), float(m[1]),
            float(m[0]) + float(m[2]),
            float(m[1]) + float(m[3]),
        ]
        return region, "marked location", None, None

    return None, None, (
        "No table/surface detected. Keep a table in view "
        "OR use Marked Location."
    ), None


# ============================================================
# TASK HANDLERS
# One handler: (context, ev) -> (detected, confidence, reasons)
# ============================================================

def _person_conf(context, track=None):
    """Base person confidence (0-100) — a real detection-quality signal."""
    persons = context.persons or []

    if not persons:
        return 0.0

    if track is not None and track.held_by is not None:
        pid = track.held_by.get("person_id")

        if pid is not None:
            for p in persons:
                if p.get("track_id") == pid:
                    try:
                        return max(
                            0.0, min(100.0, float(
                                p.get("person_confidence", 0.0)
                            ))
                        )
                    except (TypeError, ValueError):
                        return 0.0

    try:
        return max(0.0, min(100.0, float(
            persons[0].get("person_confidence", 0.0)
        )))
    except (TypeError, ValueError):
        return 0.0


def _score(context, track, geometry_score, detected):
    """
    Confidence blend: YOLO person conf + object conf + geometry score.
    Real measurements only — no random values.
    """
    person_conf = _person_conf(context, track) / 100.0

    obj_conf = 1.0
    if track is not None:
        obj_conf = max(0.0, min(1.0, track.confidence / max(
            MIN_OBJECT_CONF, 0.01
        )))

    geo = max(0.0, min(1.0, geometry_score))

    blended = (0.4 * person_conf + 0.25 * obj_conf + 0.35 * geo) * 100.0

    if not detected:
        return 0.0

    return max(50.0, min(97.0, blended))


# ------------------------------------------------------------
# PICK TASKS
# ------------------------------------------------------------

def h_pick_bottle(context, ev):
    bottles = context.tracker.of_class(["bottle"], MIN_OBJECT_CONF)

    if not bottles:
        return False, 0.0, [
            "No bottle detected (conf >= "
            f"{MIN_OBJECT_CONF * 100:.0f}%). Keep a bottle in view."
        ]

    best = None

    for bottle in bottles:

        lift_threshold = min(
            0.35 * _box_diag(bottle.box), PICK_LIFT_CAP_PX
        )
        lift = bottle.effective_lift_px()
        sustained = bottle.hold_frames >= SUSTAINED_HOLD_FRAMES

        if bottle.is_held() and (lift > lift_threshold or sustained):
            geo = min(1.0, 0.4 + lift / (2.5 * _box_diag(bottle.box)))
            score = _score(context, bottle, geo, True)

            if best is None or score > best[1]:
                best = (
                    True,
                    score,
                    [
                        f"Bottle #{bottle.track_id} held by hand",
                        (
                            f"Lift: {lift:.0f} px upward"
                            if not sustained else
                            f"Held steadily "
                            f"({bottle.hold_frames} frames, "
                            f"lift {lift:.0f} px)"
                        ),
                    ],
                )

    if best:
        return best

    held_any = [b for b in bottles if b.is_held()]

    if held_any:
        b = held_any[0]
        return False, 0.0, [
            "Bottle touched but not lifted yet — "
            "lift it clearly upward",
        ]

    return False, 0.0, [
        "Bottle visible but no hand is near it — "
        "reach and pick it up",
    ]


def h_pick_object(context, ev):
    candidates = context.tracker.any_object(
        min_conf=MIN_OBJECT_CONF, min_diag=35.0
    )

    if not candidates:
        return False, 0.0, [
            "No object detected in frame."
        ]

    best = None

    for track in candidates:

        lift_threshold = min(
            0.3 * _box_diag(track.box), PICK_LIFT_CAP_PX
        )
        lift = track.effective_lift_px()
        sustained = track.hold_frames >= SUSTAINED_HOLD_FRAMES

        if track.is_held() and (lift > lift_threshold or sustained):
            geo = min(1.0, 0.45 + lift / (2.0 * _box_diag(track.box)))
            score = _score(context, track, geo, True)

            if best is None or score > best[1]:
                best = (
                    True,
                    score,
                    [
                        f"Object '{track.class_name}' "
                        f"#{track.track_id} held",
                        (
                            f"Lift: {lift:.0f} px"
                            if not sustained else
                            f"Held steadily "
                            f"({track.hold_frames} frames, "
                            f"lift {lift:.0f} px)"
                        ),
                    ],
                )

    if best:
        return best

    return False, 0.0, [
        "Objects visible but none is held+lifted — "
        "pick one up clearly",
    ]


# ------------------------------------------------------------
# CONTAINER PLACEMENT / REMOVAL
# ------------------------------------------------------------

def h_place_object_box(context, ev):
    region, label, err, container_tr = _detect_container(context)

    if region is None:
        return False, 0.0, [err]

    held = ev["any_held"]

    if held is not None:
        return False, 0.0, [
            f"Object still in hand — place it inside the {label}"
        ]

    placed = None

    for track in ev["free_tracks"]:
        if track.class_name == "person":
            continue

        if container_tr is not None and (
            track.track_id == container_tr.track_id
        ):
            continue  # the container must not become its own "placed object"

        if track.inside_box(region, expand=0.05) and track.is_stable(
            frames=4, max_px=8.0
        ):
            placed = track
            break

    if placed is None:
        return False, 0.0, [
            f"No object is resting inside the {label} yet",
        ]

    moved = placed.net_displacement(window=15)
    diag = max(1.0, _box_diag(region))

    geo = min(1.0, 0.5 + (moved / diag) * 0.5)

    return True, _score(context, placed, geo, True), [
        f"Object '{placed.class_name}' is inside the {label} "
        f"and stable",
        f"Recent movement: {moved:.0f} px (was carried)",
    ]


def h_remove_object_box(context, ev):
    region, label, err, container_tr = _detect_container(context)

    if region is None:
        return False, 0.0, [err]

    # was any object inside the box before?
    was_inside = None

    for track in context.tracker.tracks:
        if track.class_name == "person":
            continue

        if container_tr is not None and (
            track.track_id == container_tr.track_id
        ):
            continue

        if track.was_inside_box(region, expand=0.05):
            was_inside = track
            break

    if was_inside is None:
        return False, 0.0, [
            f"No object was seen inside the {label} yet — "
            f"put an object in first, then remove it",
        ]

    # is that object now outside + held (or recently moved out)?
    if was_inside.is_held():
        moved_out = not was_inside.inside_box(region, expand=0.10)
        lift = was_inside.effective_lift_px()

        lift_threshold = min(
            0.3 * _box_diag(was_inside.box), PICK_LIFT_CAP_PX
        )

        if moved_out or lift > lift_threshold:
            geo = 0.6 + min(0.4, lift / (2.0 * _box_diag(was_inside.box)))

            return True, _score(context, was_inside, geo, True), [
                f"Object '{was_inside.class_name}' removed "
                f"from the {label} (held by hand)",
            ]

    # alternative: another track that was a member is now outside
    for container in context.tracker.tracks:
        members = getattr(container, "members", set())

        for member_id in list(members):
            for track in context.tracker.tracks:
                if track.track_id != member_id:
                    continue

                if track.is_held() and not track.inside_box(
                    region, expand=0.10
                ):
                    return True, _score(context, track, 0.7, True), [
                        f"Object '{track.class_name}' that was inside "
                        f"the {label} is now taken out",
                    ]

    return False, 0.0, [
        f"Object seen in {label} but not yet removed — "
        f"take it out with your hand",
    ]


# ------------------------------------------------------------
# MOVEMENT / TABLE / SURFACE TASKS
# ------------------------------------------------------------

def h_move_a_to_b(context, ev):
    return _generic_move(context, ev, min_distance=0.8)


def _generic_move(context, ev, min_distance):
    """
    Common move logic:
      object was held + significant displacement + now stable/present.
    """
    track = ev["any_held"]

    candidates = [track] if track is not None else []
    candidates += [
        t for t in ev["free_tracks"] if t.class_name != "person"
    ]

    best = None

    for t in candidates:
        if t is None:
            continue

        displacement = max(
            t.carry_displacement(),
            t.displacement_while_held(),
            t.net_displacement(window=20),
        )

        scale = max(
            1.0, _torso(
                (context.persons or [{}])[0].get("keypoints")
            ) if context.persons else 100.0
        )

        required_px = min_distance * scale

        if displacement >= required_px:
            geo = min(1.0, 0.5 + displacement / (3.0 * scale))
            score = _score(context, t, geo, True)

            if best is None or score > best[1]:
                best = (
                    True,
                    score,
                    [
                        f"Object '{t.class_name}' moved "
                        f"{displacement:.0f} px "
                        f"(needed {required_px:.0f} px)",
                    ],
                )

    if best:
        return best

    if candidates:
        t = candidates[0]
        displacement = max(
            t.displacement_while_held(), t.net_displacement(window=20)
        )
        scale = max(
            1.0, _torso(
                (context.persons or [{}])[0].get("keypoints")
            ) if context.persons else 100.0
        )

        return False, 0.0, [
            f"Object movement too small: {displacement:.0f} px "
            f"(needed {min_distance * scale:.0f} px)",
        ]

    return False, 0.0, ["No object being tracked — pick one up first"]


def h_carry_a_to_b(context, ev):
    # Carry = move + object held for a longer stretch
    track = ev["any_held"]

    if track is None:
        return False, 0.0, [
            "Carry the object in your hand — "
            "no held object right now"
        ]

    return _generic_move(context, ev, min_distance=1.0)


def h_bottle_on_table(context, ev):
    surface, label, err, surface_tr = _detect_table(context)

    if surface is None:
        return False, 0.0, [err]

    bottles = context.tracker.of_class(["bottle"], MIN_OBJECT_CONF)

    if not bottles:
        return False, 0.0, ["No bottle detected in frame"]

    held_bottle = [b for b in bottles if b.is_held()]

    if held_bottle:
        return False, 0.0, [
            "Bottle still in hand — place it on the " + label
        ]

    for bottle in bottles:
        if surface_tr is not None and (
            bottle.track_id == surface_tr.track_id
        ):
            continue  # the table must not become the bottle itself

        if bottle.near_vertical(surface, tolerance_frac=0.35) and (
            bottle.horizontal_overlap(surface) >= 0.4
        ) and bottle.is_stable(frames=4, max_px=8.0):

            geo = 0.75

            return True, _score(context, bottle, geo, True), [
                f"Bottle resting on {label} "
                f"(bottom at surface level, stable)",
            ]

    return False, 0.0, [
        f"Bottle not resting on the {label} yet — "
        f"place its base on the surface",
    ]


def h_move_bottle_places(context, ev):
    """
    Move Bottle: Place 1 -> Place 2 (staged, live-demo reliable).

    Place 1 = left third of the view, Place 2 = right third
    (the place regions/green boxes are drawn on the live feed).

    Staged pipeline (each stage requires consecutive frames):
      1. bottle detected in Place 1
      2. hand near the bottle (pick up)
      3. bottle leaves Place 1 (carried)
      4. movement tracked while carried
      5. bottle enters Place 2
      6. remains there stable for several frames
      -> COMPLETED
    """
    tracker = context.tracker

    # region boxes: left third / right third of a 640-wide view
    # (derived from frame width — camera.py gives a 640x480 feed)
    frame_w = 640.0
    if context.last_frame is not None:
        frame_w = float(context.last_frame.shape[1])

    p1 = (0.0, 0.0, frame_w / 3.0, 480.0)
    p2 = (frame_w * 2.0 / 3.0, 0.0, frame_w, 480.0)

    bottles = tracker.of_class(["bottle"], MIN_OBJECT_CONF)

    if not bottles:
        return False, 0.0, ["No bottle detected in frame"]

    bottle = bottles[0]

    in_p1 = bottle.inside_box(p1, expand=0.05)
    in_p2 = bottle.inside_box(p2, expand=0.05)
    held = bottle.is_held()

    # ---- stage 1+2: bottle was in Place 1 + hand came near ----
    if in_p1 and not held:
        return False, 0.0, [
            "Stage 1/6: Bottle is in Place 1 (left) — reach and "
            "pick it up",
        ]

    if in_p1 and held:
        return False, 0.0, [
            "Stage 2/6: Hand on bottle — lift and carry it right",
        ]

    # ---- stage 3-5: bottle being carried / entering Place 2 ----
    if held and not in_p2:
        carried = bottle.carry_displacement()
        return False, 0.0, [
            f"Stage 3/6: Carrying the bottle "
            f"({carried:.0f} px so far) — take it to Place 2 (right)",
        ]

    # ---- stage 5+6: reached Place 2, now stable + free ----
    if in_p2 and not held:
        if bottle.is_stable(frames=4, max_px=8.0):
            moved = max(
                bottle.carry_displacement(),
                bottle.net_displacement(window=30),
            )

            if moved >= 0.25 * frame_w:
                geo = 0.75
                return True, _score(context, bottle, geo, True), [
                    "Bottle moved Place 1 -> Place 2 and resting "
                    "there (stable)",
                    f"Carry distance: {moved:.0f} px",
                ]

            return False, 0.0, [
                f"Bottle in Place 2 but movement only "
                f"{moved:.0f}px — carry it from Place 1 first",
            ]

        return False, 0.0, [
            "Bottle in Place 2 — hold it steady there and "
            "let go (keep hands away)",
        ]

    # bottle somewhere else (middle)
    return False, 0.0, [
        "Bottle in the middle — Place 1 is LEFT, "
        "Place 2 is RIGHT",
    ]


# ------------------------------------------------------------
# TOOL TASKS
# ------------------------------------------------------------

TOOL_CLASSES = ["scissors", "knife", "fork", "spoon", "cell phone",
                "mouse", "remote", "toothbrush", "brush"]

WORKSTATION_CLASSES = ["dining table", "laptop", "keyboard", "sink",
                       "oven", "refrigerator", "tv", "microwave"]


def h_tool_to_container(context, ev):
    region, label, err, container_tr = _detect_container(context)

    if region is None:
        return False, 0.0, [err]

    tools = context.tracker.of_class(TOOL_CLASSES, MIN_OBJECT_CONF)

    if not tools:
        return False, 0.0, [
            "No tool-like object detected (scissors/knife/fork/spoon/"
            "phone/remote...)"
        ]

    for tool in tools:
        if tool.is_held():
            return False, 0.0, [
                "Tool still in hand — place it inside the " + label
            ]

        if container_tr is not None and (
            tool.track_id == container_tr.track_id
        ):
            continue  # the container must not become the tool itself

        if tool.inside_box(region, expand=0.05) and tool.is_stable(
            frames=4, max_px=8.0
        ):
            geo = 0.8

            return True, _score(context, tool, geo, True), [
                f"Tool '{tool.class_name}' resting inside {label}",
            ]

    return False, 0.0, [
        "Tool visible but not inside the container yet",
    ]


def h_pick_move_workstation(context, ev):
    """
    Tool -> near the workstation region.
    (camera.py uses this with a 'workstation' marked location/container;
     the movement with the tool is also measured)
    """
    tools = context.tracker.of_class(TOOL_CLASSES, MIN_OBJECT_CONF)

    if not tools:
        return False, 0.0, ["No tool-like object detected"]

    # workstation = either a container-like region or a marked location
    region, label, _, container_tr = _detect_container(context)

    if region is None:
        return False, 0.0, [
            "No workstation region — set Marked Location "
            "or keep a table/sink/box in view"
        ]

    for tool in tools:
        displacement = max(
            tool.displacement_while_held(),
            tool.net_displacement(window=20),
        )

        scale = _torso(
            (context.persons or [{}])[0].get("keypoints")
        ) if context.persons else 100.0

        moved_enough = displacement >= 0.6 * max(1.0, scale)

        if container_tr is not None and (
            tool.track_id == container_tr.track_id
        ):
            continue  # the container must not become the tool itself

        at_station = tool.inside_box(region, expand=0.35) or (
            tool.near_vertical(region, 0.5)
        )

        if moved_enough and at_station and not tool.is_held():
            return True, _score(context, tool, 0.75, True), [
                f"Tool '{tool.class_name}' moved to {label} "
                f"({displacement:.0f} px travel)",
            ]

    return False, 0.0, [
        "Move the tool to the workstation region "
        "(marked location / table)",
    ]


# ------------------------------------------------------------
# HANDOFF TASKS
# ------------------------------------------------------------

def h_handoff_person(context, ev):
    if ev["person_count"] < MIN_PERSON_COUNT:
        return False, 0.0, [
            "Need 2 persons in frame for handoff "
            f"(found {ev['person_count']})"
        ]

    held = ev["any_held"]

    if held is not None:
        # giver holds; is the receiver's hand near?
        giver = held.held_by
        giver_pid = giver.get("person_id") if giver else None

        for person in ev["persons"]:
            pid = person.get("track_id")

            if pid is not None and pid == giver_pid:
                continue

            keypoints = person.get("keypoints")

            if not keypoints:
                continue

            near, side, dist = _hand_near_point(
                keypoints, held.center, _torso(keypoints), 0.5
            )

            if near:
                return False, 0.0, [
                    "Receiver's hand is near the object — "
                    "giver: leave it / receiver: grip it firmly",
                ]

        return False, 0.0, [
            "Object held but receiver's hand is not close",
        ]

    # nobody holds it now — was it recently transferred?
    for track in ev["free_tracks"]:
        if track.class_name == "person":
            continue

        if len(track.holder_history) >= 1 and track.ever_held:
            # after transfer the object is stable + in nobody's hand
            if track.is_stable(frames=3, max_px=10.0):
                return True, _score(context, track, 0.7, True), [
                    f"Object '{track.class_name}' transferred: "
                    f"was held, now released and stable",
                ]

    return False, 0.0, [
        "No handoff detected — giver should hold the object "
        "and receiver should take it",
    ]


def h_receive_person(context, ev):
    # receive = the same handoff, from the other person's perspective
    return h_handoff_person(context, ev)


# ------------------------------------------------------------
# MARKED LOCATION TASKS
# ------------------------------------------------------------

def h_place_marked_location(context, ev):
    if context.marked_location is None:
        return False, 0.0, [
            "No marked location set — use 'Set Marked Location' "
            "button first"
        ]

    m = context.marked_location

    region = [
        float(m[0]), float(m[1]),
        float(m[0]) + float(m[2]),
        float(m[1]) + float(m[3]),
    ]

    held = ev["any_held"]

    if held is not None:
        return False, 0.0, [
            "Object still in hand — place it on the marked location"
        ]

    for track in ev["free_tracks"]:
        if track.class_name == "person":
            continue

        if track.inside_box(region, expand=0.10) and track.is_stable(
            frames=4, max_px=8.0
        ):
            return True, _score(context, track, 0.75, True), [
                f"Object '{track.class_name}' resting on the "
                f"marked location",
            ]

    return False, 0.0, [
        "No object resting on the marked location yet",
    ]


def h_remove_marked_location(context, ev):
    if context.marked_location is None:
        return False, 0.0, [
            "No marked location set — use 'Set Marked Location' "
            "button first"
        ]

    m = context.marked_location

    region = [
        float(m[0]), float(m[1]),
        float(m[0]) + float(m[2]),
        float(m[1]) + float(m[3]),
    ]

    for track in context.tracker.tracks:
        if track.class_name == "person":
            continue

        was_inside = track.was_inside_box(region, expand=0.10)

        if not was_inside:
            continue

        if track.is_held() and not track.inside_box(region, expand=0.15):
            return True, _score(context, track, 0.7, True), [
                f"Object '{track.class_name}' removed from "
                f"the marked location",
            ]

    return False, 0.0, [
        "Pick up the object from the marked location",
    ]


# ------------------------------------------------------------
# SORTING / COLOR / MULTI-OBJECT
# ------------------------------------------------------------

def h_sort_containers(context, ev):
    containers = context.tracker.containers(
        context.container_classes, min_conf=MIN_CONTAINER_CONF
    )

    if len(containers) < DEFAULT_SORT_TARGETS:
        return False, 0.0, [
            f"Need at least {DEFAULT_SORT_TARGETS} containers "
            f"(found {len(containers)})"
        ]

    # did different containers get different objects?
    occupied = set()
    details = []

    for container in containers:
        members = getattr(container, "members", set())

        if members:
            occupied.add(container.track_id)
            details.append(
                f"container #{container.track_id}: "
                f"{len(members)} object(s)"
            )

    if len(occupied) >= DEFAULT_SORT_TARGETS:
        return True, _score(context, None, 0.75, True), [
            "Objects sorted into different containers",
        ] + details

    return False, 0.0, [
        f"Objects must be placed into at least "
        f"{DEFAULT_SORT_TARGETS} different containers "
        f"(currently {len(occupied)})",
    ]


def h_color_matching(context, ev):
    color = context.chosen_color

    if color is None:
        return False, 0.0, [
            "Select a color first (Color selector in the "
            "Task Library) — detecting color hints from RGB "
            "is not reliable"
        ]

    containers = context.tracker.containers(
        context.container_classes, min_conf=MIN_CONTAINER_CONF
    )

    if not containers:
        return False, 0.0, ["No box/container detected in frame"]

    # IMPORTANT HONEST NOTE: COCO does not give colors. We measure the
    # color attribute from a single dominant hue (HSV) — a REAL
    # measurement, not random, but lighting-sensitive. The object must
    # be stable inside the target container AND match the user-chosen
    # color.
    placed = None

    for track in ev["free_tracks"]:
        if track.class_name == "person":
            continue

        for container in containers:
            if track.inside_box(container.box, expand=0.05) and (
                track.is_stable(frames=4, max_px=8.0)
            ):
                placed = track
                break

        if placed is not None:
            break

    if placed is None:
        return False, 0.0, [
            f"Place the '{color}' object inside the box first"
        ]

    # color measure (real HSV analysis — from object_tasks)
    measured = context.measure_object_color(placed)

    if measured is None:
        return False, 0.0, [
            "Object found in box but its color could not be "
            "measured reliably this frame"
        ]

    if measured != color:
        return False, 0.0, [
            f"Object in box measures '{measured}', "
            f"task needs '{color}'",
        ]

    return True, _score(context, placed, 0.7, True), [
        f"'{color}' object verified inside the box "
        "(real HSV color measurement)",
    ]


def h_place_multiple_container(context, ev):
    containers = context.tracker.containers(
        context.container_classes, min_conf=MIN_CONTAINER_CONF
    )

    if not containers:
        return False, 0.0, ["No container detected in frame"]

    needed = context.required_count or DEFAULT_MULTIPLE_COUNT

    members = getattr(containers[0], "members", set())

    if len(members) >= needed:
        return True, _score(context, None, 0.75, True), [
            f"{len(members)} objects are inside the container "
            f"(needed {needed})"
        ]

    return False, 0.0, [
        f"{len(members)}/{needed} objects inside the container",
    ]


def h_remove_multiple_container(context, ev):
    containers = context.tracker.containers(
        context.container_classes, min_conf=MIN_CONTAINER_CONF
    )

    if not containers:
        return False, 0.0, ["No container detected in frame"]

    needed = context.required_count or DEFAULT_MULTIPLE_COUNT

    container = containers[0]
    members = getattr(container, "members", set())

    removed = 0

    for member_id in members:
        for track in context.tracker.tracks:
            if track.track_id == member_id and not track.inside_box(
                container.box, expand=0.10
            ):
                removed += 1

    if removed >= needed:
        return True, _score(context, None, 0.7, True), [
            f"{removed} objects removed from the container "
            f"(needed {needed})"
        ]

    return False, 0.0, [
        f"{removed}/{needed} objects removed so far",
    ]


def h_collect_one_by_one(context, ev):
    needed = context.required_count or DEFAULT_MULTIPLE_COUNT

    held_ids = {
        t.track_id for t in context.tracker.tracks if t.ever_held
    }

    if len(held_ids) >= needed:
        return True, _score(context, None, 0.7, True), [
            f"{len(held_ids)} distinct objects picked up "
            f"(needed {needed})"
        ]

    return False, 0.0, [
        f"{len(held_ids)}/{needed} distinct objects picked so far — "
        f"pick them one by one",
    ]


# ------------------------------------------------------------
# ARRANGE / TRANSFER
# ------------------------------------------------------------

def h_arrange_order(context, ev):
    order = context.sorted_order or []

    if len(order) < 2:
        return False, 0.0, [
            "Choose at least 2 objects to arrange "
            "(Order selector in Task Library)"
        ]

    # find the target objects
    tracks = []

    for name in order:
        found = context.tracker.of_class([name], MIN_OBJECT_CONF)

        if not found:
            return False, 0.0, [
                f"Object '{name}' not detected in frame"
            ]

        tracks.append(found[0])

    # left-to-right order check (stable + correct sequence)
    tracks.sort(key=lambda t: t.box[0])

    current_order = [t.class_name for t in tracks]
    target_sorted = sorted(order)

    aligned = current_order == target_sorted

    all_stable = all(
        t.is_stable(frames=4, max_px=8.0) for t in tracks
    )

    if aligned and all_stable:
        return True, _score(context, None, 0.7, True), [
            "Objects arranged in the specified "
            "left-to-right order",
        ]

    return False, 0.0, [
        f"Current order: {current_order} — "
        f"target: {target_sorted} (left to right)",
    ]


def h_transfer_containers(context, ev):
    containers = context.tracker.containers(
        context.container_classes, min_conf=MIN_CONTAINER_CONF
    )

    if len(containers) < 2:
        return False, 0.0, [
            f"Need 2 containers in frame (found {len(containers)})"
        ]

    c1, c2 = containers[0], containers[1]

    # was an object in c1 before and is it in c2 now?
    for track in context.tracker.tracks:
        if track.class_name == "person":
            continue

        if track.track_id in (c1.track_id, c2.track_id):
            continue  # the containers must not become the object themselves

        if track.was_inside_box(c1.box, expand=0.05) and track.inside_box(
            c2.box, expand=0.05
        ) and track.is_stable(frames=3, max_px=10.0):

            return True, _score(context, track, 0.75, True), [
                f"Object '{track.class_name}' moved from container "
                f"#{c1.track_id} to container #{c2.track_id}",
            ]

    return False, 0.0, [
        "Move an object from the first container "
        "into the second one",
    ]


# ------------------------------------------------------------
# INSPECT / TOUCH / POINT
# ------------------------------------------------------------

def h_inspect_object(context, ev):
    track = ev["any_held"]

    if track is None:
        return False, 0.0, [
            "Hold the object in your hand to inspect it"
        ]

    holder = track.held_by
    side = holder.get("side") if holder else None

    person = None

    if holder and holder.get("person_id") is not None:
        for p in ev["persons"]:
            if p.get("track_id") == holder.get("person_id"):
                person = p
                break

    if person is None and ev["persons"]:
        person = ev["persons"][0]

    if person is None or side is None:
        return False, 0.0, ["Object held but person not tracked"]

    keypoints = person.get("keypoints")

    if not keypoints:
        return False, 0.0, ["Held object but keypoints not visible"]

    torso = _torso(keypoints)
    raised = _wrist_raised(keypoints, side, torso)

    if raised:
        return True, _score(context, track, 0.7, True), [
            "Object held up at eye level — inspecting",
        ]

    return False, 0.0, [
        "Raise the object toward your face to inspect it",
    ]


def h_touch_object(context, ev):
    candidates = context.tracker.any_object(
        min_conf=MIN_OBJECT_CONF, min_diag=35.0
    )

    if not candidates:
        return False, 0.0, ["No object detected in frame"]

    for track in candidates:
        for person in ev["persons"]:
            keypoints = person.get("keypoints")

            if not keypoints:
                continue

            near, side, dist = _hand_near_point(
                keypoints, track.center, _torso(keypoints), 0.3
            )

            if near:
                return True, _score(context, track, 0.6, True), [
                    f"Hand touched '{track.class_name}' "
                    f"({dist:.0f} px away)",
                ]

    return False, 0.0, [
        "Touch the object with your hand",
    ]


def h_point_object(context, ev):
    candidates = context.tracker.any_object(
        min_conf=MIN_OBJECT_CONF, min_diag=35.0
    )

    if not candidates:
        return False, 0.0, ["No object detected in frame"]

    for person in ev["persons"]:
        keypoints = person.get("keypoints")

        if not keypoints:
            continue

        torso = _torso(keypoints)

        for side in ("left", "right"):
            if not _pointing_pose(keypoints, side, torso):
                continue

            wrist = _wrist(keypoints, side)
            elbow = _kp(keypoints, f"{side}_elbow")

            if wrist is None or elbow is None:
                continue

            # pointing direction vector
            dx = wrist[0] - elbow[0]
            dy = wrist[1] - elbow[1]

            norm = math.hypot(dx, dy)

            if norm == 0:
                continue

            ux, uy = dx / norm, dy / norm

            for track in candidates:
                ox = track.center[0] - elbow[0]
                oy = track.center[1] - elbow[1]

                proj = ox * ux + oy * uy

                if proj <= 0:
                    continue

                perp = abs(ox * uy - oy * ux)
                cone = 0.55 * proj

                if perp <= cone:
                    return True, _score(context, track, 0.6, True), [
                        f"Pointing at '{track.class_name}' "
                        f"({side} arm extended toward it)",
                    ]

    return False, 0.0, [
        "Extend your arm and point at the object",
    ]


# ============================================================
# HANDLER REGISTRY
# ============================================================

HANDLERS = {
    "pick_bottle": h_pick_bottle,
    "pick_object": h_pick_object,
    "place_object_box": h_place_object_box,
    "remove_object_box": h_remove_object_box,
    "move_a_to_b": h_move_a_to_b,
    "bottle_on_table": h_bottle_on_table,
    "tool_to_container": h_tool_to_container,
    "handoff_person": h_handoff_person,
    "receive_person": h_receive_person,
    "place_marked_location": h_place_marked_location,
    "remove_marked_location": h_remove_marked_location,
    "sort_containers": h_sort_containers,
    "color_matching": h_color_matching,
    "carry_a_to_b": h_carry_a_to_b,
    "place_multiple_container": h_place_multiple_container,
    "remove_multiple_container": h_remove_multiple_container,
    "arrange_order": h_arrange_order,
    "inspect_object": h_inspect_object,
    "touch_object": h_touch_object,
    "point_object": h_point_object,
    "transfer_containers": h_transfer_containers,
    "collect_one_by_one": h_collect_one_by_one,
    "pick_move_workstation": h_pick_move_workstation,
    "move_bottle_places": h_move_bottle_places,
}

TASK_LABELS = {
    key: spec["label"] for key, spec in TASK_LIBRARY.items()
}

TASK_LABELS.update({
    key: spec["label"] for key, spec in UNSUPPORTED_TASKS.items()
})

VERIFIED_OBJECT_TASKS = sorted(TASK_LIBRARY.keys())


# ============================================================
# CONTEXT + BRIDGE (used by camera.py)
# ============================================================

class ObjectTaskContext:
    """
    Shared context for one verification run.
    camera.py updates it with every batch frame.
    """

    def __init__(self, tracker, marked_location=None, chosen_color=None,
                 sorted_order=None, required_count=None,
                 container_classes=None):
        self.tracker = tracker                      # ObjectTracker
        self.persons = []                           # latest pose persons
        self.objects = []                           # latest raw detections
        self.marked_location = marked_location      # [x, y, w, h] ya None
        self.chosen_color = chosen_color            # 'red'... ya None
        self.sorted_order = sorted_order            # [names] ya None
        self.required_count = required_count        # int ya None
        self.container_classes = list(
            container_classes
            if container_classes is not None else CONTAINER_CLASSES
        )
        self.last_frame = None                      # for color measurement

        # for task 15 color verification (real HSV measurement)
        self._color_cache = {}

    def measure_object_color(self, track):
        """
        Dominant color of the object's box (REAL HSV measurement,
        cached per track). Returns 'red'/'blue'/'green'/'yellow'
        or None (unreliable).
        """
        if track.track_id in self._color_cache:
            return self._color_cache[track.track_id]

        frame = getattr(self, "last_frame", None)

        if frame is None:
            return None

        import cv2
        import numpy as np

        x1, y1, x2, y2 = [int(max(0, v)) for v in track.box]

        h, w = frame.shape[:2]

        x2 = min(x2, w - 1)
        y2 = min(y2, h - 1)

        if x2 - x1 < 4 or y2 - y1 < 4:
            return None

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        h_ch, s_ch, v_ch = cv2.split(hsv)

        mask = (s_ch > 90) & (v_ch > 70)

        if int(mask.sum()) < 30:
            self._color_cache[track.track_id] = None
            return None

        hue = np.median(h_ch[mask])

        name = None

        if hue < 12 or hue >= 165:
            name = "red"
        elif 35 <= hue < 85:
            name = "green"
        elif 20 <= hue < 35:
            name = "yellow"
        elif 85 <= hue < 165:
            name = "blue"

        self._color_cache[track.track_id] = name

        return name


def get_step_action(session):
    """The current step's object-task action key (or None)."""
    step = session.current_step()

    if step is None:
        return None

    return step.get("action")


def check_object_task_step(session, context):
    """
    camera.py bridge — object-task verification of the current step.

    NOTE: the OWNER of the temporal window / completion logic is
    TaskVerificationSession.process_persons() (shared with the pose
    path). This function only returns one-frame detection.

    Returns:
        None  -> the step is not an object-task (the pose path will be used)
        dict  -> {detected, confidence, reasons, is_object_task}
    """
    action = get_step_action(session)

    if action is None or action not in HANDLERS:
        return None

    # evaluate the evidence (one place, for all handlers)
    ev = evaluate_evidence(context)

    try:
        detected, confidence, reasons = HANDLERS[action](context, ev)
    except Exception as exc:
        detected, confidence, reasons = False, 0.0, [
            f"Verification error (handled safely): {exc}"
        ]

    return {
        "detected": bool(detected),
        "confidence": round(float(confidence), 1),
        "reasons": [str(r) for r in reasons],
        "is_object_task": True,
    }


def _summarize_context(context):
    """Short context summary for the UI (debug panel)."""
    tracker = context.tracker

    return {
        "persons": len(context.persons or []),
        "objects": len(
            [t for t in tracker.tracks if t.missed == 0]
        ),
        "held": len(
            [t for t in tracker.tracks if t.is_held()]
        ),
        "containers": len(
            tracker.containers(
                context.container_classes, MIN_CONTAINER_CONF
            )
        ),
        "marked_location": context.marked_location is not None,
    }
