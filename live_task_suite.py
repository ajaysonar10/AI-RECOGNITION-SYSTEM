"""
Live multi-task test suite — tests remaining object tasks one by one
with real camera + real user interaction.

For each task: prints what to do, waits TASK_SECONDS, records pass/fail.
Run:  python live_task_suite.py
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

# per-task window (seconds) + confirmation frames (lower = faster confirm)
TASK_SECONDS = 30
CONFIRM_FRAMES = 6

TASKS = [
    (
        "Move object from one place to another",
        "1. Hold an object (phone) in view\n"
        "2. CARRY it at least one arm-length to the side\n"
        "3. Keep holding it there (carry counts even while held)",
    ),
    (
        "Put bottle on table",
        "1. Keep a TABLE/desk surface visible in the LOWER part of frame\n"
        "2. Hold a BOTTLE (or any bottle-like item)\n"
        "3. PLACE the bottle down ON the table and let go",
    ),
    (
        "Touch object",
        "1. Keep an object visible on the desk\n"
        "2. REACH OUT and TOUCH it with any hand (hold the touch)",
    ),
    (
        "Point to a designated object",
        "1. Keep an object visible to the side\n"
        "2. Extend your arm STRAIGHT toward it (pointing line,\n"
        "   arm fully extended at shoulder height or above)",
    ),
    (
        "Pick up and inspect object",
        "1. Pick an object up\n"
        "2. RAISE it toward your face / shoulder height\n"
        "3. Hold it up as if inspecting it",
    ),
    (
        "Place object on marked location",
        "1. Target = CENTER region of frame (240x200 px box\n"
        "   around the middle — roughly where your face is NOT)\n"
        "2. Place the object down so it sits in the middle\n"
        "   of the view, let go, keep hands away",
    ),
    (
        "Give object to another person",
        "1. Need TWO people in frame\n"
        "2. Person A holds the object, Person B takes it\n"
        "3. B holds it steady after taking (transfer completes)",
    ),
]


def run():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("FAIL: camera will not open")
        return 2

    # warm-up
    for _ in range(10):
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
    analyze_frame(frame)
    detect_objects(frame)

    results = []

    for task_text, instructions in TASKS:
        # marked location = frame center (640x480 view ka beech ka
        # region) — suite-run ke dauran ye fixed target hai
        marked = [200, 140, 240, 200]

        print()
        print("=" * 52)
        print(f"TASK: {task_text}")
        print("-" * 52)
        print(instructions)
        print(f"(window: {TASK_SECONDS}s — confirming at "
              f"{CONFIRM_FRAMES} frames)")
        print("=" * 52)

        tracker = ObjectTracker()
        ctx = ObjectTaskContext(tracker, marked_location=marked)
        session = TaskVerificationSession(
            [task_text], required_frames=CONFIRM_FRAMES
        )

        start = time.time()
        last_report = 0.0
        done = False
        result = None

        while time.time() - start < TASK_SECONDS:
            ok, frame = cap.read()
            if not ok:
                continue

            _, persons = analyze_frame(frame)
            _, objects = detect_objects(frame)

            ctx.persons = persons
            ctx.tracker.update(objects, persons)
            result = session.process_persons(persons, object_context=ctx)

            now = time.time()
            if now - last_report >= 2.0:
                last_report = now
                elapsed = int(now - start)
                tracks = [
                    t for t in tracker.tracks if t.missed == 0
                ]
                objs = ", ".join(
                    f"{t.class_name}" +
                    ("[HELD]" if t.is_held() else "")
                    for t in tracks
                ) or "-"
                print(
                    f"  [{elapsed:2d}s] streak="
                    f"{result['streak']}/{result['required_frames']} "
                    f"objects={objs}"
                )
                if not result["detected"] and result["reasons"]:
                    print(f"          reason: "
                          f"{result['reasons'][0][:80]}")

            if result["all_completed"]:
                done = True
                print(f"  >>> COMPLETED at "
                      f"{int(time.time()-start)}s "
                      f"(conf {result['confidence']})")
                break

        results.append((task_text, done, result))
        status = "PASS ✅" if done else "NOT COMPLETED ❌"
        print(f"  RESULT: {status}")
        if not done and result is not None:
            print(f"  last reason: {result['reasons'][:2]}")

    cap.release()

    print()
    print("=" * 52)
    print("SUMMARY")
    print("=" * 52)
    npass = sum(1 for _, d, _ in results if d)
    for task_text, done, _ in results:
        print(f"  {'PASS' if done else 'FAIL':<6} {task_text}")
    print(f"\n{npass}/{len(results)} tasks completed live")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(run())
