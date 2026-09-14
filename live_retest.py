"""
Live retest utility — real camera frames through the FULL pipeline:
webcam -> pose_detection -> object_detection -> ObjectTracker ->
TaskVerificationSession("Pick up an object"). Live telemetry every second.

Keep for future checks:  python live_retest.py
NOTE: close the browser task session first (it holds the webcam).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2

from pose_detection import analyze_frame
from object_detection import detect_objects
from object_tracking import ObjectTracker
from object_tasks import ObjectTaskContext
from task_detection import TaskVerificationSession

WINDOW_SECONDS = 60

print("Opening camera (index 0, DSHOW)...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("FAIL: camera 0 could not open (busy in browser session?)")
    for idx in (1, 2):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"OK: camera opened on index {idx}")
            break
    else:
        sys.exit(2)

print("Camera OK. Loading models (first inference warms up)...")
frame = None
for _ in range(10):
    ret, frame = cap.read()
    if ret:
        break
    time.sleep(0.2)

if not ret:
    print("FAIL: camera opened but frames do not arrive")
    cap.release()
    sys.exit(2)

_, persons = analyze_frame(frame)
_, objects = detect_objects(frame)
print(f"Warm-up done: persons={len(persons)}, objects={len(objects)}")

tracker = ObjectTracker()
ctx = ObjectTaskContext(tracker)
session = TaskVerificationSession(["Pick up an object"])

print()
print("=" * 46)
print(" GRAB NOW - pick up any object and HOLD IT UP")
print(" (window: %d seconds)" % WINDOW_SECONDS)
print("=" * 46)

start = time.time()
last_report = 0.0
completed = False

while time.time() - start < WINDOW_SECONDS:
    ret, frame = cap.read()
    if not ret:
        continue

    _, persons = analyze_frame(frame)
    _, objects = detect_objects(frame)

    ctx.persons = persons
    tracks = ctx.tracker.update(objects, persons)

    result = session.process_persons(persons, object_context=ctx)

    now = time.time()
    if now - last_report >= 1.0:
        last_report = now
        elapsed = int(now - start)

        objs = [
            f"{t.class_name}(conf={t.confidence:.2f},"
            f"diag={int((t.box[2]-t.box[0])*(t.box[3]-t.box[1])**0.5)})"
            for t in tracks
        ]
        held = [
            f"{t.class_name} lift={t.lift_px():.0f}px"
            f" eff={t.effective_lift_px():.0f}px"
            for t in tracks if t.is_held()
        ]

        print(
            f"[{elapsed:2d}s] persons={len(persons)} "
            f"objects={objs if objs else '-'} "
            f"held={held if held else '-'} "
            f"streak={result['streak']}/{result['required_frames']} "
            f"detected={result['detected']}"
        )
        if result["reasons"]:
            print(f"        reason: {result['reasons'][0][:90]}")

    if result["all_completed"]:
        completed = True
        print()
        print("*** LIVE TEST: TASK COMPLETED *** "
              f"(at {int(time.time()-start)}s, "
              f"confidence={result['confidence']})")
        break

cap.release()

print()
if completed:
    print("RESULT: PASS - pipeline completes a real pick-up.")
    print("Browser app will behave the same - go retry 'Pick up the bottle'.")
else:
    print(f"RESULT: NO COMPLETION in {WINDOW_SECONDS}s window.")
    print("Telemetry above shows which stage lacked evidence "
          "(person/objects/held/lift).")
