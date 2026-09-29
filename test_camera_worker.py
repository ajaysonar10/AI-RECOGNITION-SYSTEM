"""
Test: camera_worker lifecycle + throttling (NO real camera needed).

Verifies the lag-fix guarantees:
  1. The worker publishes latest-frame snapshots with REAL fps counters.
  2. Capture is faster than pose inference (throttle works, nothing faked).
  3. stop() joins the thread and releases the camera.
  4. start_worker() is a process-wide SINGLETON — starting on a running
     worker is a no-op and the extra cap is released (exactly ONE worker).
  5. stop_active_worker() is idempotent.

A fake capture object stands in for cv2.VideoCapture: it returns
synthetic BGR frames at call time, counts .read() calls, and counts
.release() calls. The real YOLO models still load (module import),
but inference runs on tiny synthetic frames.

Run:  python test_camera_worker.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_worker import (
    CameraWorker,
    SharedFrameState,
    start_worker,
    stop_active_worker,
    empty_task_result,
)

stop_active_worker()   # belt-and-braces clean slate for this process


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


class FakeCapture:
    """Stands in for cv2.VideoCapture — synthetic BGR frames."""

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.read_count = 0
        self.release_count = 0
        self.released = False
        self._props = {}

    def read(self):
        self.read_count += 1
        frame = np.zeros(
            (self.height, self.width, 3), dtype=np.uint8
        )
        frame[:, :, 0] = self.read_count % 255   # B channel varies
        return True, frame

    def isOpened(self):
        return not self.released

    def release(self):
        self.release_count += 1
        self.released = True

    def set(self, prop, value):
        self._props[prop] = value
        return True


print("\n--- TEST 1: worker publishes real snapshots ---")

shared = SharedFrameState()
cap = FakeCapture()
worker = CameraWorker(cap, "Fake Camera", shared)
worker.start()
CameraWorker._set_active(worker)   # mirror start_worker() registration

deadline = time.time() + 10.0
snap = None
while time.time() < deadline:
    snap = shared.latest()
    if (
        snap is not None
        and snap["capture_fps"] > 0
        and snap["pose_fps"] >= 0
        and snap["pose_fps"] < snap["capture_fps"]
    ):
        break
    time.sleep(0.1)

check("snapshot published", snap is not None)
if snap is not None:
    check("annotated frame present",
          snap["annotated"] is not None)
    check("capture_fps counted (real, > 0)",
          snap["capture_fps"] > 0,
          f"(got {snap['capture_fps']})")
    check("IDLE task result shape",
          snap["task_result"]["state"] == "IDLE")
    check("capture faster than pose inference (throttle)",
          snap["capture_fps"] > max(snap["pose_fps"], 0.0),
          f"(capture={snap['capture_fps']:.1f}, "
          f"pose={snap['pose_fps']:.1f})")
    check("worker alive after running", worker.is_alive())

# Let the 2 s FPS windows fill so the throttle comparison is stable.
time.sleep(2.0)
snap = shared.latest()
check("pose throttled below capture rate (real counters)",
      0.0 < snap["pose_fps"] < snap["capture_fps"],
      f"(capture={snap['capture_fps']:.1f}, "
      f"pose={snap['pose_fps']:.1f})")

print("\n--- TEST 2: singleton guard (exactly ONE worker) ---")

extra_cap = FakeCapture()
worker2, started = start_worker(extra_cap, "Fake Camera")

check("start on a running worker is a NO-OP", started is False)
check("the second cap was released (not left open)",
      extra_cap.release_count >= 1)
check("same worker object returned", worker2 is worker)
active = CameraWorker.get_active()
check("get_active returns the same live worker", active is worker)

print("\n--- TEST 3: stop joins the thread, releases the camera ---")

reads_before = cap.read_count
time.sleep(0.3)
check("capture loop kept reading while running",
      cap.read_count > reads_before)

ok = worker.stop(timeout=3.0)
check("stop() joined the thread", ok is True)
check("thread really exited", not worker.is_alive())
worker.release()
check("cap released exactly through release()",
      cap.release_count == 1 and cap.released is True)
check("get_active clears the stopped worker",
      CameraWorker.get_active() is None)

print("\n--- TEST 4: stop_active_worker idempotent ---")

check("stop with no worker -> None",
      stop_active_worker() is None)

cap3 = FakeCapture()
shared3 = SharedFrameState()
w3, started3 = start_worker(cap3, "Fake Camera")
check("fresh worker starts", started3 is True and w3 is not None)
stopped = stop_active_worker(timeout=3.0)
check("stop_active_worker stops + releases it",
      stopped is w3 and cap3.release_count == 1)
check("second stop_active_worker is a no-op",
      stop_active_worker() is None)

print("\n--- TEST 5: empty_task_result shape ---")

r = empty_task_result()
check("IDLE result has the keys the UI renders",
      r["state"] == "IDLE" and "streak" in r and "reasons" in r)

stop_active_worker()   # final cleanup

print(f"\n{'=' * 46}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'=' * 46}")
sys.exit(1 if failed else 0)
