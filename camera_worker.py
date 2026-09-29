"""
camera_worker.py
================
Background camera + inference worker for the Task Verification
live monitor.

Why this exists (lag fix):
    The old pipeline ran cap.read() + YOLO pose + YOLO objects
    synchronously inside the Streamlit script thread. Slow CPU
    inference made stale frames queue up (growing display lag) and
    blocked button clicks.

Design (keeps every existing feature, no faked results):
    - ONE background thread per camera: cap.read() paced to
      CAPTURE_FPS, 640x480, latest-frame-only (stale frames are
      never queued -> the picture stays "now").
    - Throttled inference: pose YOLO ~POSE_INFERENCE_FPS, object
      YOLO ~OBJECT_INFERENCE_FPS and ONLY when an object context
      exists. Skipped frames REUSE the last persons/objects/tracks
      for drawing + task logic (real detections, just not re-run
      every capture).
    - verify_task_step cadence is per inference frame — unchanged
      temporal confirmation semantics (streak counts inference
      frames, exactly like the old batch loop did).
    - Thread-safe latest-writer-wins snapshot for the UI + plain
      GIL-protected channels (tracks, history events).
    - Duplicate-worker guard: exactly one worker process-wide;
      starting on a running worker is a no-op.
    - The thread NEVER touches Streamlit; camera-lost is reported
      via the callback / flags and handled by the UI layer.
"""

import threading
import time
from collections import deque

import cv2

from pose_detection import analyze_frame
from object_detection import detect_objects
from task_detection import verify_task_step


# ------------------------------------------------------------
# PACING (honest, on-screen FPS counters come from these loops)
# ------------------------------------------------------------

CAPTURE_FPS = 15.0          # cap.read() cadence (latest-frame buffer)
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

POSE_INFERENCE_FPS = 10.0   # pose YOLO cadence
OBJECT_INFERENCE_FPS = 7.0  # object YOLO cadence (only with a context)

IDLE_SLEEP = 0.005          # poll granularity between captures
READ_FAIL_SLEEP = 0.05      # backoff after a failed read
MAX_READ_FAILURES = 25      # ~1.2 s of dead reads => camera lost

EVENTS_HISTORY_MAX = 200    # same cap camera.py already used


def empty_task_result():
    """IDLE result — same dict shape the UI already renders."""
    return {
        "detected": False,
        "step_completed": False,
        "all_completed": False,
        "current_step": None,
        "state": "IDLE",
        "streak": 0,
        "required_frames": 0,
        "progress": 0.0,
        "confidence": 0.0,
        "reasons": [],
        "events": [],
        "completed_step_number": None,
    }


class SharedFrameState:
    """
    Latest-writer-wins state shared between the worker thread and
    the Streamlit UI thread.

    - snapshot: dict with the newest annotated frame + result +
      real FPS counters (lock-protected single slot).
    - tracks / events: plain lists mutated under the GIL (atomic
      rebind/append) — the UI reads them for its panels.
    - running: False after the camera died (UI shows the banner).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot = None

        self.task_session = None       # synced by the UI thread
        self.object_context = None     # synced by the UI thread
        self.marked_location = None    # synced by the UI thread

        self.tracks = []
        self.events = []
        self.running = False
        self.worker_generation = 0
        self.camera_lost = False

    def publish(self, snapshot):
        with self._lock:
            self._snapshot = snapshot

    def latest(self):
        with self._lock:
            return self._snapshot

    def clear(self):
        with self._lock:
            self._snapshot = None
            self.events = []
            self.tracks = []


class CameraWorker:
    """One background thread: read -> (throttled) inference -> publish."""

    _global_lock = threading.Lock()
    _active_worker = None          # process-wide singleton

    # ------------------------------------------------------------
    # SINGLETON GUARD
    # ------------------------------------------------------------

    @classmethod
    def get_active(cls):
        """Returns the live worker, or None (and clears the stale one)."""
        with cls._global_lock:
            worker = cls._active_worker
            if worker is not None and worker.is_alive():
                return worker
            cls._active_worker = None
            return None

    @classmethod
    def _set_active(cls, worker):
        with cls._global_lock:
            cls._active_worker = worker

    # ------------------------------------------------------------

    def __init__(self, cap, source, shared, on_camera_lost=None):
        self._cap = cap
        self._source = source
        self._shared = shared
        self._on_camera_lost = on_camera_lost
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="basai-camera-worker",
            daemon=True,
        )

    # ------------------------------------------------------------

    def start(self):
        self._shared.worker_generation += 1
        self._shared.running = True
        self._shared.camera_lost = False
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()

    def is_stopped(self):
        return self._stop_event.is_set()

    def stop(self, timeout=3.0):
        """Signals the thread to exit and waits briefly (UI thread)."""
        self._stop_event.set()
        if (
            self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def release(self):
        """Releases the VideoCapture (idempotent)."""
        try:
            self._cap.release()
        except Exception as exc:
            print("Camera release error:", exc)

    # ------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------

    def _run(self):
        # Capture pacing (best effort — some mobile streams ignore it)
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        except Exception:
            pass

        cap_interval = 1.0 / CAPTURE_FPS
        pose_interval = 1.0 / POSE_INFERENCE_FPS
        obj_interval = 1.0 / OBJECT_INFERENCE_FPS

        # Honest FPS accounting (pruned 2 s windows)
        capture_times = deque(maxlen=120)
        pose_times = deque(maxlen=120)
        object_times = deque(maxlen=120)

        # Last-good inference state (reused between throttle gaps)
        last_annotated = None
        last_persons = []
        last_objects = []

        next_capture = time.perf_counter()
        consecutive_failures = 0

        while not self._stop_event.is_set():

            now = time.perf_counter()
            if now < next_capture:
                time.sleep(min(IDLE_SLEEP, next_capture - now))
                continue
            next_capture = now + cap_interval

            # ----------------------------------------------------
            # 1) READ — always fresh; stale frames are never queued
            # ----------------------------------------------------

            ret, frame = self._cap.read()

            if not ret or frame is None:
                if self._stop_event.is_set():
                    # Clean stop (e.g. the web camera stream was
                    # closed by the user) — not a camera loss.
                    break
                consecutive_failures += 1
                if consecutive_failures >= MAX_READ_FAILURES:
                    # Dead camera -> report + exit (never fake frames)
                    self._shared.running = False
                    self._shared.camera_lost = True
                    self._stop_event.set()
                    if self._on_camera_lost is not None:
                        try:
                            self._on_camera_lost()
                        except Exception:
                            pass
                    break
                time.sleep(READ_FAIL_SLEEP)
                continue

            consecutive_failures = 0
            capture_times.append(now)
            while capture_times and now - capture_times[0] > 2.0:
                capture_times.popleft()

            # UI thread may have swapped the task/context (Start,
            # Reset, New Task, Clear) — pick up the current values.
            task_session = self._shared.task_session
            object_context = self._shared.object_context
            marked_location = self._shared.marked_location

            # ----------------------------------------------------
            # 2) POSE — throttled; reuse last persons between runs
            # ----------------------------------------------------

            if (
                last_annotated is None
                or (now - (pose_times[-1] if pose_times else 0.0))
                >= pose_interval
            ):
                try:
                    annotated, persons = analyze_frame(frame)
                except Exception as exc:
                    print("Pose inference error:", exc)
                    annotated, persons = last_annotated, last_persons
                last_annotated = annotated
                last_persons = persons
                pose_times.append(now)
            else:
                annotated = last_annotated
                persons = last_persons

            # ----------------------------------------------------
            # 3) OBJECTS — throttled, ONLY when a context exists
            #    (monitor-only pose runs skip this YOLO entirely)
            # ----------------------------------------------------

            tracks = []

            if object_context is not None:
                if (
                    not object_times
                    or (now - object_times[-1]) >= obj_interval
                ):
                    try:
                        # conf 0.10: weak-but-real detections (a table at
                        # a webcam angle scores 0.07-0.11) must reach the
                        # tracker — task gates filter further downstream.
                        _, objects = detect_objects(frame, conf=0.10)
                    except Exception as exc:
                        print("Object inference error:", exc)
                        objects = last_objects
                    last_objects = objects
                    object_times.append(now)

                object_context.persons = persons
                object_context.objects = last_objects
                object_context.last_frame = frame
                object_context.marked_location = (
                    list(marked_location) if marked_location else None
                )

                tracks = object_context.tracker.update(
                    last_objects, persons
                )
                self._shared.tracks = tracks

                # Move Bottle (Place 1 -> Place 2): left/right third
                # overlays for stage guidance (same as before)
                region = None
                if object_context.marked_location is not None:
                    m = object_context.marked_location
                    region = [
                        m[0], m[1], m[0] + m[2], m[1] + m[3]
                    ]

                current_action = None
                if task_session is not None:
                    current_step = task_session.current_step()
                    if current_step is not None:
                        current_action = current_step.get("action")

                if current_action == "move_bottle_places":
                    fw = float(frame.shape[1])
                    fh = float(frame.shape[0])
                    region = [
                        [0.0, 0.0, fw / 3.0, fh],
                        [fw * 2.0 / 3.0, 0.0, fw, fh],
                    ]

                try:
                    from object_tracking import draw_tracks
                    draw_tracks(annotated, tracks, region_box=region)
                except Exception as exc:
                    print("Track drawing error:", exc)

            # ----------------------------------------------------
            # 4) TASK LOGIC — per inference frame (streak semantics
            #    unchanged from the old batch loop)
            # ----------------------------------------------------

            if task_session is not None:
                try:
                    result = verify_task_step(
                        task_session,
                        persons,
                        object_context=object_context,
                    )
                except Exception as exc:
                    print("Task verification error:", exc)
                    result = empty_task_result()
            else:
                result = empty_task_result()

            if result.get("events"):
                self._shared.events.extend(result["events"])
                del self._shared.events[:-EVENTS_HISTORY_MAX]

            # ----------------------------------------------------
            # 5) TASK LABEL ON FRAME (same overlay as before)
            # ----------------------------------------------------

            label_text = None
            if result.get("all_completed"):
                label_text = "ALL TASKS COMPLETED"
            elif result.get("current_step"):
                label_text = (
                    f"TASK: {result['current_step'].upper()} "
                    f"| {result['streak']}/"
                    f"{result['required_frames']}"
                )

            if label_text:
                try:
                    cv2.putText(
                        annotated,
                        label_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 120),
                        2,
                        cv2.LINE_AA,
                    )
                except Exception:
                    pass

            # ----------------------------------------------------
            # 6) PUBLISH — latest snapshot only (UI renders this)
            # ----------------------------------------------------

            self._shared.publish({
                "annotated": annotated,          # BGR, drawn on
                "persons": persons,
                "tracks": tracks,
                "task_result": result,
                "capture_fps": (
                    len(capture_times) / 2.0
                    if len(capture_times) >= 2 else 0.0
                ),
                "pose_fps": (
                    len(pose_times) / 2.0
                    if len(pose_times) >= 2 else 0.0
                ),
                "object_fps": (
                    len(object_times) / 2.0
                    if len(object_times) >= 2 else 0.0
                ),
                "timestamp": time.time(),
            })

        # Clean exit: nothing is released here — the UI thread owns
        # the cap lifecycle (stop_active_worker does join + release).

    # ------------------------------------------------------------


# ------------------------------------------------------------
# MODULE-LEVEL LIFECYCLE (used by camera.py)
# ------------------------------------------------------------


def start_worker(cap, source, on_camera_lost=None):
    """
    Starts the ONE process-wide worker on `cap`.

    Returns (worker, started):
        started=True  -> a new worker was created + started
        started=False -> an existing live worker was reused (no-op);
                         `cap` is released here because a second
                         VideoCapture on the same device must not
                         stay open.
    """
    with CameraWorker._global_lock:
        existing = CameraWorker._active_worker

        if existing is not None and existing.is_alive():
            # Exactly ONE worker rule: reuse it. The registration
            # STAYS (it must survive — a cleared entry here would
            # make the next get_active() spawn a duplicate).
            try:
                cap.release()
            except Exception:
                pass
            return existing, False

        # Dead/stale entry: unregister it, replace below.
        CameraWorker._active_worker = None

    if existing is not None:
        existing.stop(timeout=1.5)
        existing.release()

    shared = SharedFrameState()
    worker = CameraWorker(cap, source, shared, on_camera_lost=on_camera_lost)
    worker.start()
    CameraWorker._set_active(worker)
    return worker, True


def stop_active_worker(timeout=3.0):
    """
    Stops the live worker (if any), joins it and releases its camera.
    Safe to call repeatedly / when nothing runs (idempotent).
    """
    with CameraWorker._global_lock:
        worker = CameraWorker._active_worker
        CameraWorker._active_worker = None

    if worker is None:
        return None

    worker.stop(timeout=timeout)
    worker.release()
    worker._shared.running = False
    return worker
