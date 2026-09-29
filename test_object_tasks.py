"""
Simulation test for the Object-Interaction Task Library (without a camera).

Builds synthetic detections + synthetic persons and verifies:
  1. Parser: supported / unsupported / unknown task text.
  2. ObjectTracker: the same object every frame -> one stable track.
  3. Hold detection: wrist proximity -> is_held -> lift -> release.
  4. Full pipeline: pick_bottle COMPLETE through TaskVerificationSession.
  5. place_object_box: bottle stable inside a bowl -> COMPLETE
     (regression: an empty bowl must not become its own "placed object").
  6. handoff_person: 2 persons, hold -> release -> transfer COMPLETE.
  7. Unsupported task: honest reason, never COMPLETE.
  8. Sequential multi-step: pick -> place, steps complete only in their TURN.
  9. bottle_on_table: bottle bottom on the surface + stable -> COMPLETE.

Run:  python test_object_tasks.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from object_tracking import ObjectTracker, ObjectTrack
from object_tasks import (
    parse_object_task,
    ObjectTaskContext,
    check_object_task_step,
)
from task_detection import (
    TaskVerificationSession,
    TASK_CONFIRM_FRAMES,
)


# -----------------------------------------------------
# Synthetic builders (same pattern as test_task_verification.py)
# -----------------------------------------------------

def make_person(keypoints, person_conf=90.0, track_id=1):
    return {
        "keypoints": keypoints,
        "person_confidence": person_conf,
        "activity": "Standing",
        "confidence": 90.0,
        "justification": [],
        "track_id": track_id,
    }


def standing_keypoints(right_wrist=None, left_wrist=(240, 300)):
    """
    Standing person — torso ≈ 150px, so wrist reach = max(0.7*150,
    0.5*box_diag). Default wrists at hip level (far away).
    """
    return {
        "nose": (320, 80),
        "left_shoulder": (270, 150), "right_shoulder": (370, 150),
        "left_elbow": (250, 230), "right_elbow": (390, 230),
        "left_wrist": left_wrist, "right_wrist": right_wrist or (400, 300),
        "left_hip": (285, 300), "right_hip": (355, 300),
        "left_knee": (280, 400), "right_knee": (360, 400),
        "left_ankle": (280, 500), "right_ankle": (360, 500),
    }


def make_det(class_name, box, confidence=0.85):
    """A detection dict like object_detection.detect_objects()."""
    return {
        "class_id": 0,
        "class_name": class_name,
        "confidence": confidence,
        "box": [float(v) for v in box],
    }


def make_context(tracker, persons, marked_location=None):
    """Fresh ObjectTaskContext with current frame's persons attached."""
    ctx = ObjectTaskContext(tracker, marked_location=marked_location)
    ctx.persons = persons
    return ctx


def feed_object_frame(session, ctx, detections, persons):
    """One frame: tracker update + session.process_persons (object path)."""
    ctx.persons = persons
    ctx.tracker.update(detections, persons)
    return session.process_persons(persons, object_context=ctx)


def move_box(box, dx, dy):
    """Shift the box (new detection position)."""
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


# -----------------------------------------------------
print("\\n--- TEST 1: Object task parsing ---")

key, detail = parse_object_task("Pick up the bottle")
check("'Pick up the bottle' -> pick_bottle", key == "pick_bottle",
      f"(got {key})")

key, _ = parse_object_task("Put bottle on table")
check("'Put bottle on table' -> bottle_on_table",
      key == "bottle_on_table", f"(got {key})")

key, detail = parse_object_task("open the box")
check("'open the box' -> unsupported",
      key is not None and detail.get("unsupported") is True,
      f"(got {key}, {detail})")

key, detail = parse_object_task("Do a backflip")
check("Unknown text -> (None, {{}})", key is None and detail == {})

key, detail = parse_object_task("Clean the workstation")
check("'Clean the workstation' -> clean_workstation",
      key == "clean_workstation" and not detail.get("unsupported"),
      f"(got {key}, {detail})")

key, detail = parse_object_task(
    "Cleaning a workstation Collect cleaning equipment "
    "clean designated area dispose/store cleaning material"
)
check("Full cleaning-assignment text -> clean_workstation",
      key == "clean_workstation" and not detail.get("unsupported"),
      f"(got {key}, {detail})")

key, detail = parse_object_task(
    "Put 3 red colored object into matching box"
)
check("Color + count extracted",
      key == "color_matching" and detail.get("color") == "red"
      and detail.get("count") == 3,
      f"(got {key}, {detail})")

# -----------------------------------------------------
print("\\n--- TEST 2: Tracker stability ---")

tracker = ObjectTracker()
bottle_box = [300, 400, 340, 480]

for _ in range(6):
    tracker.update([make_det("bottle", bottle_box)], [])

check("Same box 6 frames -> exactly 1 track", len(tracker.tracks) == 1,
      f"(got {len(tracker.tracks)})")

track = tracker.tracks[0]
check("Track id assigned", track.track_id == 1)
check("Stable after 6 static frames", track.is_stable(frames=6, max_px=6.0))
check("Net displacement ~0", track.net_displacement() < 1.0)

tracker.update([make_det("bottle", move_box(bottle_box, 35, 0))], [])
check("Moderate jump keeps identity (IoU>0 + center match)",
      len(tracker.tracks) == 1 and tracker.tracks[0].track_id == 1)
check("Displacement measured", tracker.tracks[0].net_displacement() >= 30,
      f"(got {tracker.tracks[0].net_displacement():.0f})")

# -----------------------------------------------------
print("\\n--- TEST 3: Hold / lift / release (hysteresis) ---")

tracker = ObjectTracker()
ctx = make_context(tracker, [])

# wrist bottle ke center par: 3 frames near -> streak 6 -> HOLD
center = [(bottle_box[0] + bottle_box[2]) / 2,
          (bottle_box[1] + bottle_box[3]) / 2]

for _ in range(3):
    kp = standing_keypoints(right_wrist=tuple(center))
    tracker.update([make_det("bottle", bottle_box)],
                   [make_person(kp, track_id=1)])

bottle = tracker.tracks[0]
check("Wrist at bottle 3 frames -> is_held", bottle.is_held(),
      f"(near_streak={bottle.near_streak}, held={bottle.held_by})")
check("Held by person 1 (right)",
      bottle.held_by == {"side": "right", "person_id": 1},
      f"(got {bottle.held_by})")

# lift the bottle up — 20px/frame x 4 frames = 80px lift
# (left wrist far: otherwise the lifted bottle comes within the left
#  hand's reach and the hold never releases)
y = bottle_box[1]
for _ in range(4):
    y -= 20
    lifted_box = [bottle_box[0], y, bottle_box[2], y + 80]
    kp = standing_keypoints(
        right_wrist=(center[0], lifted_box[1] + 40),
        left_wrist=(180, 500),
    )
    tracker.update([make_det("bottle", lifted_box)],
                   [make_person(kp, track_id=1)])

check("Lift px >= 60 after moving up 80px",
      bottle.lift_px() >= 60, f"(got {bottle.lift_px():.0f})")

# hand away — the streak already reached 12 during the lift,
# decay 2/frame -> release below 3: 12->10->8->6->4->2 (5 frames)
# (wrist (480,300): ~171px from lifted-center (320,360) > reach 105)
for _ in range(5):
    tracker.update(
        [make_det("bottle", lifted_box)],
        [make_person(
            standing_keypoints(right_wrist=(480, 300),
                               left_wrist=(180, 500)),
            track_id=1
        )],
    )

check("Wrist away 5 frames -> released (hysteresis decay)",
      not bottle.is_held())
check("ever_held stays True", bottle.ever_held)

# -----------------------------------------------------
print("\\n--- TEST 4: Full pipeline — pick_bottle completes ---")

tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(["Pick up the bottle"])

bottle_box = [300, 400, 340, 480]   # diag ~89 -> needs lift > 31px
cy = (bottle_box[1] + bottle_box[3]) / 2
cx = (bottle_box[0] + bottle_box[2]) / 2

r = None
# 3 frames near (hold enter) + 14 frames lifting upward
for i in range(17):
    if i < 3:
        box, wrist_y = bottle_box, cy
    else:
        lift = (i - 2) * 12            # total 168px — clearly above the diagonal
        box = [bottle_box[0], bottle_box[1] - lift,
               bottle_box[2], bottle_box[3] - lift]
        wrist_y = box[1] + 40

    kp = standing_keypoints(right_wrist=(cx, wrist_y))
    r = feed_object_frame(session, ctx, [make_det("bottle", box)],
                          [make_person(kp, track_id=1)])

check("pick_bottle step COMPLETED",
      session.steps[0]["state"] == "COMPLETED",
      f"(state={session.steps[0]['state']}, "
      f"reasons={r['reasons'] if r else None})")
check("Completion event generated",
      r is not None and any(
          e["status"] == "Completed" for e in r["events"]))
check("all_completed True (single task)",
      r is not None and r["all_completed"])

# -----------------------------------------------------
print("\\n--- TEST 5: place_object_box (+ empty-bowl regression) ---")

# --- 5a: empty bowl -> NOT detected (container self-match fix) ---
tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(["Place object inside box"])

bowl_box = [400, 300, 600, 420]
for _ in range(10):
    r = feed_object_frame(session, ctx, [make_det("bowl", bowl_box)],
                          [make_person(standing_keypoints(), track_id=1)])

check("Empty bowl alone -> NOT detected",
      session.steps[0]["state"] != "COMPLETED",
      f"(state={session.steps[0]['state']}, reasons={r['reasons']})")

# --- 5b: put the bottle into the bowl -> COMPLETE ---
# the bowl track is still #1; the bottle becomes track #2
held_box = [200, 200, 240, 280]     # away from the bowl, in hand
target_box = [460, 340, 500, 420]   # inside the bowl (center inside)

for i in range(31):
    if i < 3:
        # hold enter: wrist bottle par
        box, wrist = held_box, (220, 240)
        kp = standing_keypoints(right_wrist=wrist)
        dets = [make_det("bowl", bowl_box), make_det("bottle", box)]
    elif i < 11:
        # carried towards the bowl (slowly, must stay the same track)
        t = (i - 2) / 8.0
        x1 = 200 + (460 - 200) * t
        y1 = 200 + (340 - 200) * t
        box = [x1, y1, x1 + 40, y1 + 80]
        kp = standing_keypoints(right_wrist=(x1 + 20, y1 + 40))
        dets = [make_det("bowl", bowl_box), make_det("bottle", box)]
    else:
        # release: stable inside the bowl, hand away
        box = target_box
        kp = standing_keypoints(right_wrist=(420, 250))
        dets = [make_det("bowl", bowl_box), make_det("bottle", box)]

    r = feed_object_frame(session, ctx, dets,
                          [make_person(kp, track_id=1)])

bottle_track = [t for t in tracker.tracks
                if t.class_name == "bottle"][0]
check("Bottle stayed one track (no re-id)",
      bottle_track.track_id == 2,
      f"(ids={[t.track_id for t in tracker.tracks]})")
check("place_object_box COMPLETED",
      session.steps[0]["state"] == "COMPLETED",
      f"(state={session.steps[0]['state']}, reasons={r['reasons']})")

# -----------------------------------------------------
print("\\n--- TEST 6: handoff_person (2 persons) ---")

tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(["Give object to another person"])

obj_box = [300, 400, 340, 470]
p1 = make_person(
    standing_keypoints(right_wrist=(320, 435)), track_id=1
)
p2 = make_person(
    standing_keypoints(right_wrist=(400, 300)), track_id=2
)

# 3 frames: p1 holds
for _ in range(3):
    r = feed_object_frame(session, ctx, [make_det("cup", obj_box)],
                          [p1, p2])

check("Object held by giver", ctx.tracker.tracks[0].is_held())

# 2 frames: p1 pulls their hand away (release), p2 still far
for _ in range(2):
    p1_away = make_person(standing_keypoints(), track_id=1)
    r = feed_object_frame(session, ctx, [make_det("cup", obj_box)],
                          [p1_away, p2])

track = ctx.tracker.tracks[0]
check("Object released after giver lets go", not track.is_held())
check("ever_held recorded", track.ever_held)

# 3+ frames: object stable + 2 persons present -> transfer COMPLETE
for _ in range(16):
    r = feed_object_frame(session, ctx, [make_det("cup", obj_box)],
                          [p1_away, p2])

check("handoff COMPLETED with 2 persons",
      session.steps[0]["state"] == "COMPLETED",
      f"(state={session.steps[0]['state']}, reasons={r['reasons']})")

# --- 6b: only 1 person -> never completes ---
tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(["Give object to another person"])
for _ in range(20):
    r = feed_object_frame(session, ctx, [make_det("cup", obj_box)],
                          [p1])
check("Single person -> handoff NOT completed",
      session.steps[0]["state"] != "COMPLETED",
      f"(state={session.steps[0]['state']})")

# -----------------------------------------------------
print("\\n--- TEST 7: Unsupported task handled honestly ---")

session = TaskVerificationSession(["Open the box"])
check("Unsupported reason attached",
      session.steps[0]["unsupported_reason"] is not None)

for _ in range(TASK_CONFIRM_FRAMES + 6):
    r = feed_object_frame(
        session, ctx, [make_det("bowl", bowl_box)],
        [make_person(standing_keypoints(), track_id=1)]
    )

check("Unsupported NEVER completes",
      session.steps[0]["state"] != "COMPLETED"
      and not session.all_completed)
check("Honest UNSUPPORTED reason in output",
      any("UNSUPPORTED" in x for x in r["reasons"]),
      f"(reasons={r['reasons']})")

# -----------------------------------------------------
print("\\n--- TEST 8: Sequential multi-step (pick -> place) ---")

tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(
    ["Pick up the bottle", "Place object inside box"]
)

bottle_box = [300, 400, 340, 480]
bowl_box = [400, 300, 600, 420]
cx = (bottle_box[0] + bottle_box[2]) / 2
cy = (bottle_box[1] + bottle_box[3]) / 2

# During Step 1: the bottle is ALREADY in the bowl + free
# (place-evidence) -> the pick step must NOT complete
for _ in range(16):
    kp = standing_keypoints(right_wrist=(420, 250))
    r = feed_object_frame(
        session, ctx,
        [make_det("bowl", bowl_box),
         make_det("bottle", [460, 340, 500, 420])],
        [make_person(kp, track_id=1)]
    )

check("Step 1 (pick) NOT completed by place-evidence",
      session.current_index == 0
      and session.steps[0]["state"] != "COMPLETED",
      f"(index={session.current_index}, "
      f"state={session.steps[0]['state']})")

# Now actually lift: wrist on it + lift (14px/frame — pick threshold
# 0.35*diag~31px, detected from ~i=5; 12+ detections in the window)
for i in range(19):
    if i < 3:
        box, wrist_y = bottle_box, cy
        dets = [make_det("bowl", bowl_box),
                make_det("bottle", bottle_box)]
    else:
        lift = (i - 2) * 14
        box = [bottle_box[0], bottle_box[1] - lift,
               bottle_box[2], bottle_box[3] - lift]
        wrist_y = box[1] + 40
        dets = [make_det("bowl", bowl_box),
                make_det("bottle", box)]

    kp = standing_keypoints(right_wrist=(cx, wrist_y))
    r = feed_object_frame(session, ctx, dets,
                          [make_person(kp, track_id=1)])

check("Step 1 (pick) COMPLETED after hold+lift",
      session.steps[0]["state"] == "COMPLETED")
check("Step 2 now IN PROGRESS",
      session.steps[1]["state"] == "IN PROGRESS")

# Step 2: carry the bottle to the bowl + release
bottle_track = [t for t in tracker.tracks
                if t.class_name == "bottle"][0]
cur = list(bottle_track.box)

for i in range(29):
    if i < 9:
        # carried towards the bowl (hand with it)
        t = (i + 1) / 9.0
        x1 = cur[0] + (460 - cur[0]) * t
        y1 = cur[1] + (340 - cur[1]) * t
        box = [x1, y1, x1 + 40, y1 + 80]
        kp = standing_keypoints(right_wrist=(x1 + 20, y1 + 40))
    else:
        # release — hand away, stable in the bowl
        # (streak 12 -> decay 5 frames -> release -> 12 detections)
        box = [460, 340, 500, 420]
        kp = standing_keypoints(right_wrist=(420, 250))

    dets = [make_det("bowl", bowl_box), make_det("bottle", box)]
    r = feed_object_frame(session, ctx, dets,
                          [make_person(kp, track_id=1)])
    if r and r["all_completed"]:
        break   # preserve the completion result (with events)

check("Step 2 (place) COMPLETED",
      session.steps[1]["state"] == "COMPLETED",
      f"(state={session.steps[1]['state']}, reasons={r['reasons']})")
check("ALL TASKS COMPLETED (multi-step)",
      r is not None and r["all_completed"])
check("All-completed event recorded",
      any(e["status"] == "All Tasks Completed"
          for e in r["events"]))

# -----------------------------------------------------
print("\\n--- TEST 9: bottle_on_table ---")

tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(["Put bottle on table"])

table_box = [100, 380, 700, 500]    # dining table
bottle_box = [300, 250, 340, 380]   # bottom (380) table top (380) par

for _ in range(16):
    # hand away from the bottle (free bottle, resting on the table)
    kp = standing_keypoints(right_wrist=(450, 250),
                            left_wrist=(180, 500))
    r = feed_object_frame(
        session, ctx,
        [make_det("dining table", table_box),
         make_det("bottle", bottle_box)],
        [make_person(kp, track_id=1)]
    )

check("bottle_on_table COMPLETED",
      session.steps[0]["state"] == "COMPLETED",
      f"(state={session.steps[0]['state']}, reasons={r['reasons']})")

# --- 9b: bottle NOT on the table -> NOT detected ---
tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(["Put bottle on table"])

for _ in range(16):
    kp = standing_keypoints(right_wrist=(450, 250),
                            left_wrist=(180, 500))
    r = feed_object_frame(
        session, ctx,
        [make_det("dining table", table_box),
         make_det("bottle", [300, 100, 340, 180])],   # up in mid-air
        [make_person(kp, track_id=1)]
    )

check("Bottle in mid-air -> NOT completed",
      session.steps[0]["state"] != "COMPLETED",
      f"(state={session.steps[0]['state']})")

# -----------------------------------------------------
print("\\n--- TEST 10: Mid-air re-grip (wrist flicker) lift retention ---")

# REAL-LIFE BUG regression: object lifted, wrist keypoint flickered
# (hold released), gripped again — the lift measurement does not
# break from the baseline (otherwise the pick task never completed).

tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(["Pick up an object"])

obj_box = [300, 400, 360, 470]     # diag ~92 -> needs lift > 27px
cx = (obj_box[0] + obj_box[2]) / 2
cy = (obj_box[1] + obj_box[3]) / 2

held_lift = 0
for i in range(26):
    phase = i // 6
    if phase == 0:
        # frames 0-5: hold enter (wrist at object)
        box, wrist = obj_box, (cx, cy)
    elif phase == 1:
        # frames 6-11: lift 20px/frame
        held_lift += 20
        box = [obj_box[0], obj_box[1] - held_lift,
               obj_box[2], obj_box[3] - held_lift]
        wrist = (cx, box[1] + 35)
    elif phase == 2:
        # frames 12-17: WRIST FLICKER — hand away (the hold should
        # release, but the lift baseline is RETAINED)
        box = [obj_box[0], obj_box[1] - held_lift,
               obj_box[2], obj_box[3] - held_lift]
        wrist = (480, 200)
    else:
        # frames 18-25: re-grip mid-air (same lifted height)
        box = [obj_box[0], obj_box[1] - held_lift,
               obj_box[2], obj_box[3] - held_lift]
        wrist = (cx, box[1] + 35)

    kp = standing_keypoints(right_wrist=wrist, left_wrist=(180, 500))
    r = feed_object_frame(session, ctx, [make_det("cup", box)],
                          [make_person(kp, track_id=1)])

cup = [t for t in tracker.tracks if t.class_name == "cup"][0]
check("Track survived lift + flicker (no re-id)", cup.track_id == 1)
check("Lift baseline retained through release",
      cup.effective_lift_px() >= 100,
      f"(effective={cup.effective_lift_px():.0f}, raw={cup.lift_px():.0f})")
check("pick_object COMPLETED despite mid-air flicker",
      session.steps[0]["state"] == "COMPLETED",
      f"(state={session.steps[0]['state']}, reasons={r['reasons']})")

# -----------------------------------------------------
print("\\n--- TEST 11: User's 5 demo tasks (parsing) ---")

five = [
    ("Pick up a bottle", "pick_bottle"),
    ("Put the bottle on the table", "bottle_on_table"),
    ("Pick up the bottle and place it inside a box",
     "place_object_box"),
    ("Take the bottle out of the box", "remove_object_box"),
    ("Move the bottle from Place 1 to Place 2",
     "move_bottle_places"),
]
for text, expected in five:
    key, detail = parse_object_task(text)
    check(f"'{text}' -> {expected}",
          key == expected and not detail.get("unsupported"),
          f"(got {key}, {detail})")

# -----------------------------------------------------
print("\\n--- TEST 12: move_bottle_places staged pipeline ---")

tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(
    ["Move the bottle from Place 1 to Place 2"]
)

bottle_box = [80, 300, 120, 420]     # Place 1 = left third

for i in range(38):
    if i < 4:
        # stage 1: bottle rest in Place 1
        box = bottle_box
        wrist = (450, 250)
    elif i < 7:
        # stage 2: grip (hold enter)
        box = bottle_box
        wrist = (100, 360)
    elif i < 21:
        # stage 3-4: continuous carry right (32px/frame)
        x1 = 80 + (i - 6) * 32
        box = [x1, 300, x1 + 40, 420]
        wrist = (x1 + 20, 360)
    else:
        # stage 5-6: release in Place 2, hands away, stable
        box = [528, 300, 568, 420]
        wrist = (450, 250)

    kp = standing_keypoints(right_wrist=wrist, left_wrist=(180, 500))
    r = feed_object_frame(
        session, ctx,
        [make_det("bottle", box)],
        [make_person(kp, track_id=1)]
    )

check("move_bottle_places COMPLETED (staged)",
      session.steps[0]["state"] == "COMPLETED",
      f"(state={session.steps[0]['state']}, reasons={r['reasons']})")

# negative: the bottle only reached the middle — must not complete
tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(
    ["Move the bottle from Place 1 to Place 2"]
)
for i in range(20):
    box = [280, 300, 320, 420]   # middle, rest
    kp = standing_keypoints(right_wrist=(450, 250),
                            left_wrist=(180, 500))
    r = feed_object_frame(
        session, ctx, [make_det("bottle", box)],
        [make_person(kp, track_id=1)]
    )
check("Bottle only in middle -> NOT completed",
      session.steps[0]["state"] != "COMPLETED")

# -----------------------------------------------------
print("\\n--- TEST 13: clean_workstation staged pipeline ---")

tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(["Clean the workstation"])

table_box = [0, 400, 640, 480]      # designated area = table surface

r = None
for i in range(46):
    if i < 4:
        # stage 1: grip the equipment (wrist on the bottle)
        wx, wy = 450, 415
        box = [430, 395, 470, 455]
    elif i < 22:
        # stage 2: wiping — wrist oscillates laterally over the table,
        # bottle follows the hand (held)
        wx = 360 if ((i - 4) % 4) < 2 else 440
        wy = 415
        box = [wx - 20, 395, wx + 20, 455]
    else:
        # stage 3: put the bottle back down (release + stable)
        wx, wy = 450, 250
        box = [420, 400, 460, 460]

    kp = standing_keypoints(right_wrist=(wx, wy),
                            left_wrist=(180, 500))
    r = feed_object_frame(
        session, ctx,
        [make_det("dining table", table_box, 0.15),
         make_det("bottle", box)],
        [make_person(kp, track_id=1)]
    )

check("clean_workstation COMPLETED (staged)",
      session.steps[0]["state"] == "COMPLETED",
      f"(state={session.steps[0]['state']}, reasons={r['reasons']})")

# negative: wiping with nothing stored (equipment never put down)
tracker = ObjectTracker()
ctx = make_context(tracker, [])
session = TaskVerificationSession(["Clean the workstation"])
for i in range(30):
    wx = 360 if (i % 4) < 2 else 440
    kp = standing_keypoints(right_wrist=(wx, 415),
                            left_wrist=(180, 500))
    r = feed_object_frame(
        session, ctx,
        [make_det("dining table", table_box, 0.15),
         make_det("bottle", [wx - 20, 395, wx + 20, 455])],
        [make_person(kp, track_id=1)]
    )
check("Wiping but equipment never stored -> NOT completed",
      session.steps[0]["state"] != "COMPLETED",
      f"(state={session.steps[0]['state']}, reasons={r['reasons']})")

# -----------------------------------------------------
# BONUS: ObjectTrack direct API sanity
print("\\n--- BONUS: ObjectTrack API sanity ---")

t = ObjectTrack(99, "bottle", [0, 0, 40, 80], 0.9)
check("members set initialized (regression)", isinstance(t.members, set))
check("lift_px 0 without hold", t.lift_px() == 0.0)
check("displacement_while_held 0 without hold",
      t.displacement_while_held() == 0.0)

# -----------------------------------------------------
print(f"\\n{'=' * 50}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'=' * 50}")

sys.exit(1 if failed else 0)
