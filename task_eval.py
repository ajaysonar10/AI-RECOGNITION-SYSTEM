"""
task_eval.py
============
BAS-AI • Live Task Evaluation Engine

The user gives a task (natural language), the LIVE CAMERA feed is
analyzed, and the AI decides whether the task was COMPLETED or not.

How it works:
  1) parse_task(task_text)
     -> extracts the required ACTIVITIES (walking/sitting/standing) and
        required OBJECTS (bottle, chair, ...).

  2) LiveTaskEvaluator
     -> ON EVERY FRAME:
          - pose_detection.analyze_frame()   -> person activity + justification
          - object_detection.detect_objects() -> objects
          - person-object PROXIMITY          -> pick/hold evidence
     -> evidence accumulates FRAME BY FRAME
     -> when all requirements are CONFIRMED -> TASK COMPLETED
     -> the user presses "Check Task" -> final verdict + report

Usage (from camera.py / app.py):
    ev = LiveTaskEvaluator(task_text)
    ev.step(frame)           # with every camera frame
    ev.progress()            # 0-1 overall completion
    verdict = ev.evaluate()  # final verdict dict
"""

import re
import time

import cv2

from pose_detection import analyze_frame, reset_tracker
from object_detection import detect_objects


# ============================================================
# SETTINGS
# ============================================================

ACTIVITY_CONFIRM_CONFIDENCE = 60.0   # confidence above this = activity CONFIRMED
PERSON_PROXIMITY_MARGIN = 0.6        # extra "reach" area around the person bbox
                                     # (fraction of person height)

# Scoring
SCORE_CONFIRMED = 100
SCORE_SEEN = 55
SCORE_SEEN_OBJECT = 60

STATUS_COMPLETED = "TASK COMPLETED"
STATUS_PARTIAL = "TASK PARTIALLY COMPLETED"
STATUS_NOT = "TASK NOT COMPLETED"
STATUS_REVIEW = "NEEDS REVIEW"


# ============================================================
# TASK PARSING
# ============================================================

ACTIVITY_KEYWORDS = {
    "Walking": ["walk", "walking", "move around", "roam", "stroll"],
    "Sitting": ["sit", "sitting", "sit down", "be seated", "take a seat"],
    "Standing": ["stand", "standing", "stand up", "stand still",
                 "stay standing"],
}

TASK_OBJECT_WORDS = {
    "bottle":       ["bottle", "water bottle"],
    "cup":          ["cup", "mug", "glass"],
    "wine glass":   ["wine glass"],
    "chair":        ["chair", "seat"],
    "dining table": ["table", "dining table", "desk"],
    "book":         ["book", "notebook"],
    "laptop":       ["laptop"],
    "cell phone":   ["phone", "cell phone", "mobile", "smartphone"],
    "keyboard":     ["keyboard"],
    "mouse":        ["mouse"],
    "tv":           ["tv", "television", "monitor", "screen"],
    "remote":       ["remote", "remote control"],
    "backpack":     ["backpack", "bag", "school bag"],
    "handbag":      ["handbag", "purse"],
    "scissors":     ["scissors"],
    "sports ball":  ["ball", "sports ball"],
    "apple":        ["apple"],
    "banana":       ["banana"],
    "knife":        ["knife"],
    "fork":         ["fork"],
    "spoon":        ["spoon"],
    "bowl":         ["bowl"],
    "clock":        ["clock", "watch"],
    "umbrella":     ["umbrella"],
    "teddy bear":   ["teddy bear", "soft toy"],
    "toothbrush":   ["toothbrush"],
}

INTERACTION_VERBS = [
    "pick up", "pick", "grab", "hold", "holding", "carry",
    "place", "put", "bring", "give", "hand over", "take",
    "lift", "drop", "throw", "show", "use",
]


def _contains_phrase(text, phrase):
    """Word-boundary match (so the word 'sit' never matches inside 'visit')."""
    return re.search(
        r"\b" + re.escape(phrase) + r"\b",
        text
    ) is not None


def parse_task(task_text):
    """
    Parses a natural language task and extracts the requirements.

    Returns:
        {
            "activities":          ["Walking", ...],
            "objects":             ["bottle", ...],
            "interaction_required": bool,
            "mentions_person":     bool,
        }
    """
    text = (task_text or "").lower().strip()

    activities = []

    activity_phrases = []

    for activity, phrases in ACTIVITY_KEYWORDS.items():
        for phrase in phrases:
            activity_phrases.append((phrase, activity))

    # Longest phrases first — "move around" matches before
    # small tokens like "move" (which no longer exist in the already-replaced text)
    activity_phrases.sort(key=lambda item: -len(item[0]))

    for phrase, activity in activity_phrases:

        if activity in activities:
            continue

        if _contains_phrase(text, phrase):
            activities.append(activity)
            text = text.replace(phrase, " ")

    objects = []
    object_phrases = []

    for coco_name, words in TASK_OBJECT_WORDS.items():
        for word in words:
            object_phrases.append((word, coco_name))

    # Longest words first — "cell phone" before "phone"
    object_phrases.sort(key=lambda item: -len(item[0]))

    for word, coco_name in object_phrases:

        if coco_name in objects:
            continue

        if _contains_phrase(text, word):
            objects.append(coco_name)
            text = text.replace(word, " ")

    interaction_required = False

    for verb in INTERACTION_VERBS:
        if _contains_phrase(text, verb):
            interaction_required = True
            break

    mentions_person = _contains_phrase(
        (task_text or "").lower(), "person"
    ) or _contains_phrase(
        (task_text or "").lower(), "people"
    )

    return {
        "activities": activities,
        "objects": objects,
        "interaction_required": interaction_required,
        "mentions_person": mentions_person,
    }


# ============================================================
# PERSON-OBJECT INTERACTION (proximity check)
# ============================================================

def _object_near_person(person_bbox, object_box):
    """
    Checks whether the object's center lies inside the person's
    "reach zone" (bbox + margin) or not.

    Returns:
        (inside: bool, distance_px: float)
    """
    x1, y1, x2, y2 = [float(v) for v in person_bbox]

    person_h = max(1.0, y2 - y1)
    margin = person_h * PERSON_PROXIMITY_MARGIN

    ex1, ey1 = x1 - margin, y1 - margin
    ex2, ey2 = x2 + margin, y2 + margin

    ox1, oy1, ox2, oy2 = [float(v) for v in object_box]

    obj_cx = (ox1 + ox2) / 2
    obj_cy = (oy1 + oy2) / 2

    inside = (ex1 <= obj_cx <= ex2) and (ey1 <= obj_cy <= ey2)

    pcx = (x1 + x2) / 2
    pcy = (y1 + y2) / 2

    distance = ((obj_cx - pcx) ** 2 + (obj_cy - pcy) ** 2) ** 0.5

    return inside, distance


# ============================================================
# LIVE TASK EVALUATOR
# ============================================================

class LiveTaskEvaluator:
    """
    Checks task completion against the live camera feed.

    Call step(frame) on every frame. Evidence accumulates
    internally. When all requirements are confirmed, the task is
    automatically marked COMPLETED.

    evaluate() gives the final verdict (when the user presses
    "Check Task", or on auto-complete).
    """

    def __init__(self, task_text):

        self.task_text = (task_text or "").strip()
        self.requirements = parse_task(self.task_text)

        # --------------------------------------------
        # Evidence accumulators
        # --------------------------------------------

        # activity -> best confidence so far
        self.activity_best = {}

        # activity -> justification of the best frame
        self.activity_reasons = {}

        # object class -> best detection confidence (0-100)
        self.objects_seen = {}

        # object class -> min distance to person (px)
        self.interactions = {}

        self.frames_processed = 0
        self.frames_with_person = 0
        self.start_time = time.time()
        self.last_annotated = None

        self.finished = False
        self.verdict = None

        # 10-second wrong activity tracking
        self.latest_activity = None
        self.wrong_activity_start = None
        self.voice_alert_pending = False

    # ------------------------------------------------
    # MAIN: call with every frame
    # ------------------------------------------------

    def step(self, frame):
        """
        Processes one camera frame.

        Returns:
            (annotated_frame, live_status_text)
        """

        if self.finished:
            # Task already decided — return the last annotated frame only
            return self.last_annotated, self._status_line()

        annotated, persons = analyze_frame(frame)

        # Objects from the fresh raw frame (avoid confusion with pose-plot boxes)
        _, objects = detect_objects(frame)

        self.frames_processed += 1

        # --------------------------------------------
        # Activity evidence
        # --------------------------------------------

        if persons:
            self.frames_with_person += 1

        for person in persons:

            activity = person["activity"]
            conf = float(person["confidence"])

            if conf > self.activity_best.get(activity, 0.0):
                self.activity_best[activity] = conf
                self.activity_reasons[activity] = list(
                    person["justification"]
                )
        # ============================================================
        # 10-SECOND WRONG ACTIVITY DETECTION
        # ============================================================

        if persons:

            current_activity = persons[0]["activity"]
            self.latest_activity = current_activity

            required_activities = self.requirements["activities"]

            if required_activities:

                # If the detected activity is not the required activity
                if current_activity not in required_activities:

                    if self.wrong_activity_start is None:
                        self.wrong_activity_start = time.time()
                        self.voice_alert_pending = False

                    wrong_duration = (
                        time.time() - self.wrong_activity_start
                    )

                    # 10 seconds elapsed
                    if wrong_duration >= 10:
                        self.voice_alert_pending = True

                else:
                    # Correct activity -> reset timer
                    self.wrong_activity_start = None
                    self.voice_alert_pending = False

        else:
            self.latest_activity = None
            self.wrong_activity_start = None
            self.voice_alert_pending = False        

        # --------------------------------------------
        # Object evidence + proximity
        # --------------------------------------------

        for obj in objects:

            name = obj["class_name"]
            conf = float(obj["confidence"]) * 100

            if conf > self.objects_seen.get(name, 0.0):
                self.objects_seen[name] = conf

        # Which objects are near the person
        for obj in objects:

            name = obj["class_name"]

            if name == "person":
                continue

            best = None

            for person in persons:

                inside, distance = _object_near_person(
                    person["position"]["bbox_pixels"],
                    obj["box"]
                )

                if inside:
                    if best is None or distance < best:
                        best = distance

            if best is not None:

                current = self.interactions.get(name)

                if current is None or best < current:
                    self.interactions[name] = best

        # --------------------------------------------
        # Auto-complete check (everything confirmed?)
        # --------------------------------------------

        if self._all_confirmed():
            self.evaluate()

        # Show the status line on the frame too
        status = self._status_line()

        cv2.putText(
            annotated,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 120),
            2,
        )

        self.last_annotated = annotated

        return annotated, status

    # ------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------

    def _activity_state(self, activity):
        """'confirmed' / 'seen' / 'missing'"""

        best = self.activity_best.get(activity, 0.0)

        if best >= ACTIVITY_CONFIRM_CONFIDENCE:
            return "confirmed"

        if best > 0:
            return "seen"

        return "missing"

    def _object_state(self, obj):
        """'interacted' / 'seen' / 'missing'"""

        if obj == "person":
            return "interacted" if self.frames_with_person > 0 else "missing"

        seen = obj in self.objects_seen
        near = obj in self.interactions

        if near:
            return "interacted"

        if seen:
            return "seen"

        return "missing"

    def _all_confirmed(self):
        """Have all requirements been confirmed?"""

        reqs = self.requirements

        for activity in reqs["activities"]:
            if self._activity_state(activity) != "confirmed":
                return False

        for obj in reqs["objects"]:
            if self._object_state(obj) != "interacted":
                return False

        # At least a few frames must have been processed
        return self.frames_processed > 0

    def _part_scores(self):
        """(activity_part, object_part) — None if there is no requirement."""

        states_act = [
            self._activity_state(a)
            for a in self.requirements["activities"]
        ]

        activity_part = None

        if states_act:
            activity_part = (
                sum(
                    SCORE_CONFIRMED if s == "confirmed"
                    else (SCORE_SEEN if s == "seen" else 0)
                    for s in states_act
                )
                / len(states_act)
            )

        states_obj = [
            self._object_state(o)
            for o in self.requirements["objects"]
        ]

        object_part = None

        if states_obj:
            object_part = (
                sum(
                    SCORE_CONFIRMED if s == "interacted"
                    else (SCORE_SEEN_OBJECT if s == "seen" else 0)
                    for s in states_obj
                )
                / len(states_obj)
            )

        return activity_part, object_part

    def _status_line(self):
        """Short live status string (for the frame overlay + metrics)."""

        reqs = self.requirements

        if not reqs["activities"] and not reqs["objects"]:
            return "Task unclear — mention activity/object"

        done = 0
        total = len(reqs["activities"]) + len(reqs["objects"])

        for a in reqs["activities"]:
            if self._activity_state(a) == "confirmed":
                done += 1

        for o in reqs["objects"]:
            if self._object_state(o) == "interacted":
                done += 1

        if done == total:
            return f"Task COMPLETED ({done}/{total})"

        return f"Task progress: {done}/{total} confirmed"

    # ------------------------------------------------
    # PUBLIC: progress + final verdict
    # ------------------------------------------------

    def progress(self):
        """Overall completion 0.0 - 1.0 (for the UI progress bar)."""

        reqs = self.requirements
        total = len(reqs["activities"]) + len(reqs["objects"])

        if total == 0:
            return 0.0

        activity_part, object_part = self._part_scores()

        parts = [p for p in (activity_part, object_part)
                 if p is not None]

        if not parts:
            return 0.0

        return max(0.0, min(1.0, sum(parts) / len(parts) / 100.0))

    def stats_elapsed(self):
        """Elapsed time of the live session (seconds)."""
        return time.time() - self.start_time

    def get_voice_alert(self):

        if not self.voice_alert_pending:
            return None

        required = self.requirements["activities"]

        if required:
            message = (
                "Wrong activity. "
                f"Please perform {required[0]}."
            )
        else:
            message = "Wrong activity. Please perform the required task."

        # Do not repeat the alert immediately after one alert
        self.voice_alert_pending = False

        return message

    def verdict_checklist(self):
        """
        Current state of every requirement for the UI checklist:
            [{"activity": ..., "state": ...}, {"object": ..., "state": ...}]
        """

        items = []

        for a in self.requirements["activities"]:
            items.append({
                "activity": a,
                "state": self._activity_state(a),
            })

        for o in self.requirements["objects"]:
            items.append({
                "object": o,
                "state": self._object_state(o),
            })

        return items

    def evaluate(self):
        """
        Builds the final verdict (idempotent — call it repeatedly,
        the same result returns until new evidence arrives).
        """

        activity_part, object_part = self._part_scores()

        parts = [p for p in (activity_part, object_part)
                 if p is not None]

        if not parts:

            score = 50
            status = STATUS_REVIEW
            explanation = (
                "No recognizable activity or object was found for the task. "
                "Write the task more clearly (e.g.: 'walk across the room', "
                "'sit on the chair', 'pick up the bottle')."
            )

        else:

            score = int(round(sum(parts) / len(parts)))

            if score >= 85:
                status = STATUS_COMPLETED
            elif score >= 55:
                status = STATUS_PARTIAL
            else:
                status = STATUS_NOT

            reasons = []

            confirmed_acts = [
                a for a in self.requirements["activities"]
                if self._activity_state(a) == "confirmed"
            ]
            missing_acts = [
                a for a in self.requirements["activities"]
                if self._activity_state(a) == "missing"
            ]

            if confirmed_acts:
                reasons.append(
                    "Activity observed: " + ", ".join(confirmed_acts)
                )

            if missing_acts:
                reasons.append(
                    "Activity never observed: " + ", ".join(missing_acts)
                )

            confirmed_objs = [
                o for o in self.requirements["objects"]
                if self._object_state(o) == "interacted"
            ]
            seen_objs = [
                o for o in self.requirements["objects"]
                if self._object_state(o) == "seen"
            ]
            missing_objs = [
                o for o in self.requirements["objects"]
                if self._object_state(o) == "missing"
            ]

            if confirmed_objs:
                reasons.append(
                    "Object(s) found near the person: "
                    + ", ".join(confirmed_objs)
                )

            if seen_objs:
                if self.requirements["interaction_required"]:
                    reasons.append(
                        "Object(s) seen but never near the person: "
                        + ", ".join(seen_objs)
                    )
                else:
                    reasons.append(
                        "Object(s) seen: " + ", ".join(seen_objs)
                    )

            if missing_objs:
                reasons.append(
                    "Object(s) never detected: " + ", ".join(missing_objs)
                )

            if self.frames_with_person == 0:
                reasons.insert(
                    0,
                    "No person detected in the camera feed yet."
                )

            explanation = " | ".join(reasons)

            if self.requirements["interaction_required"] and self.requirements["objects"]:
                explanation += (
                    " (Interaction approximated by object appearing "
                    "inside the person's reach zone.)"
                )

        self.verdict = {
            "score": score,
            "status": status,
            "explanation": explanation,
            "activity_checks": [
                {
                    "activity": a,
                    "state": self._activity_state(a),
                    "best_confidence": round(
                        self.activity_best.get(a, 0.0), 1
                    ),
                }
                for a in self.requirements["activities"]
            ],
            "object_checks": [
                {
                    "object": o,
                    "state": self._object_state(o),
                    "best_confidence": round(
                        self.objects_seen.get(o, 0.0), 1
                    ),
                    "min_distance": (
                        round(self.interactions[o], 1)
                        if o in self.interactions else None
                    ),
                }
                for o in self.requirements["objects"]
            ],
            "activity_reasons": dict(self.activity_reasons),
            "stats": {
                "elapsed_s": round(time.time() - self.start_time, 1),
                "frames_processed": self.frames_processed,
                "frames_with_person": self.frames_with_person,
                "objects_detected": sorted(self.objects_seen.keys()),
                "activities_detected": sorted(self.activity_best.keys()),
            },
            "progress": self.progress(),
        }

        self.finished = True

        return self.verdict

    def reset(self, task_text=None):
        """Start a new task (or repeat the same task)."""

        if task_text is not None:
            self.task_text = (task_text or "").strip()
            self.requirements = parse_task(self.task_text)

        self.activity_best = {}
        self.activity_reasons = {}
        self.objects_seen = {}
        self.interactions = {}
        self.frames_processed = 0
        self.frames_with_person = 0
        self.start_time = time.time()
        self.last_annotated = None
        self.finished = False
        self.verdict = None

        # 10-second wrong activity tracking
        self.latest_activity = None
        self.wrong_activity_start = None
        self.voice_alert_pending = False

        reset_tracker()


# ============================================================
# REPORT BUILDER
# ============================================================

def build_report(evaluator):
    """Builds a plain-text report from the LiveTaskEvaluator verdict."""

    verdict = evaluator.verdict
    reqs = evaluator.requirements
    stats = verdict["stats"]

    lines = []
    add = lines.append

    add("TASK EVALUATION REPORT (LIVE)")
    add("=============================")
    add("")
    add("Task:")
    add(evaluator.task_text)
    add("")
    add(f"Status : {verdict['status']}")
    add(f"Score  : {verdict['score']}%")
    add(f"Progress: {verdict['progress'] * 100:.0f}%")
    add("")

    add("Required Activities: "
        + (", ".join(reqs["activities"]) if reqs["activities"]
           else "None identified"))

    for check in verdict["activity_checks"]:

        state = check["state"].upper()

        add(f"  - {check['activity']}: {state} "
            f"(best confidence {check['best_confidence']}%)")

        reasons = verdict.get("activity_reasons", {}).get(
            check["activity"]
        )

        if reasons:
            for reason in reasons[:3]:
                add(f"      * {reason}")

    add("")
    add("Required Objects: "
        + (", ".join(reqs["objects"]) if reqs["objects"]
           else "None identified"))

    for check in verdict["object_checks"]:

        state = check["state"].upper()

        add(f"  - {check['object']}: {state} "
            f"(best confidence {check['best_confidence']}%)")

    add("")
    add("Live Session Stats:")
    add(f"  - Elapsed time     : {stats['elapsed_s']} s")
    add(f"  - Frames processed : {stats['frames_processed']}")
    add(f"  - Frames w/ person : {stats['frames_with_person']}")
    add("")
    add("Objects observed: "
        + (", ".join(stats["objects_detected"])
           if stats["objects_detected"] else "None"))
    add("Activities observed: "
        + (", ".join(stats["activities_detected"])
           if stats["activities_detected"] else "None"))
    add("")
    add("Analysis:")
    add(verdict["explanation"])

    return "\n".join(lines)


# ============================================================
# STANDALONE DEMO (without UI)
# ============================================================

if __name__ == "__main__":

    print("BAS-AI Live Task Evaluation — DEMO")
    print("Type a task, then perform it in front of the webcam.")
    print("'v' = check verdict | 'q' = quit\n")

    task_text = input("Enter the task: ")

    ev = LiveTaskEvaluator(task_text)

    print("Parsed activities :", ev.requirements["activities"] or "None")
    print("Parsed objects    :", ev.requirements["objects"] or "None")
    print("")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        raise SystemExit(1)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        annotated, status = ev.step(frame)

        cv2.putText(
            annotated,
            f"{status}  |  {ev.progress() * 100:.0f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 120),
            2,
        )

        cv2.imshow("Live Task Evaluation (v = verdict, q = quit)",
                   annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("v"):

            verdict = ev.evaluate()

            print("\n" + "=" * 50)
            print("STATUS:", verdict["status"])
            print("SCORE :", str(verdict["score"]) + "%")
            print("=" * 50)
            print(build_report(ev))
            print("=" * 50 + "\n")

        elif key == ord("q"):
            break

        if ev.finished:
            print("\n>>> TASK AUTO-COMPLETED:",
                  ev.verdict["status"])
            break

    cap.release()
    cv2.destroyAllWindows()

    if not ev.finished:
        ev.evaluate()

    print("\nFINAL:", ev.verdict["status"],
          f"({ev.verdict['score']}%)")
