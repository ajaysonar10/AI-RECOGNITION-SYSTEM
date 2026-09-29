# Camera Lag Fix — Saved Progress (analysis complete, no code changed yet)

Date saved: 2026-09-29
Status: **IMPLEMENTED + SIMULATION-TESTED. Manual `streamlit run app.py` check by the user still recommended.**
Goal: fix live camera lag WITHOUT rebuilding or removing any feature.

---

## 1. Active code path (what actually runs)

- Entry: `app.py` → nav page "📡 Live Monitor" → `show_task_verification()` in **root `camera.py`** (2089 lines).
- `show_camera()` and `show_task_evaluation()` still exist in `camera.py` but are NOT routed in the nav — leave them alone.
- Pipeline per frame: `pose_detection.analyze_frame()` (YOLO `yolo11n-pose.pt`) → optional `object_detection.detect_objects()` (YOLO `yolo11n.pt`) → `object_tracking` → `verify_task_step` / monitor UI.
- `AI-Recognition-System/` subfolder is an older duplicate — **do not touch**.
- `desktop_app.py` just runs `streamlit run app.py` in a webview.
- Environment: Streamlit **1.63.0** (`st.fragment` + `st.rerun(scope="fragment")` available — verified), ultralytics 8.4.149, OpenCV 5.0.0, Windows.

## 2. Root causes of the lag (verified with line numbers)

1. **Both YOLO models run on EVERY frame** — `camera.py::_task_verification_batch()` (~line 1840):
   `TV_BATCH_FRAMES = 5` loop calls `analyze_frame(frame)` (pose YOLO) per frame, and whenever an
   `object_context` exists also `detect_objects(frame, conf=0.10)` (object YOLO) per frame → 2× CPU inference
   per frame, ~2–4 effective FPS → visible lag.
2. **Monitor-only mode forces object YOLO on** — `show_task_verification()` (~line 1790): with no task active it
   creates a default `ObjectTaskContext(ObjectTracker())`, so object detection runs even for pose-only monitoring.
3. **Blocking `time.sleep(1.0)`** in the task-completed banner path (camera.py:1753) inside the script thread.
4. **Full-page `st.rerun()` every ~1 s** (end of `_task_verification_batch`, camera.py:2089) re-renders sidebar,
   header, CSS, all panels — instead of just the camera/panels.
5. **No reader thread / no latest-frame-only buffer** — `cap.read()` is called synchronously in the Streamlit
   script thread; slow inference makes stale frames queue up and display lag grows.
6. `TASK_CONFIRM_FRAMES = 12` (task_detection.py:66) — streak counts processed frames; throttling inference is OK
   (still real temporal confirmation), but the confirmation cadence must stay per-inference-frame, not per-render.

Already OK (no change needed):
- Models load ONCE per process (module-level `YOLO(...)` singletons in `pose_detection.py:30` and
  `object_detection.py:23`) — the "load once" requirement is already satisfied; just stop calling them every frame.
- Camera device persists in `st.session_state["tv_cap"]` between batches; Stop releases via `release_task_camera()`.
- Windows camera opens with `CAP_DSHOW` (`_task_camera()`).

## 3. Planned fix (minimal, keeps all features)

**Files to change:**
- NEW `camera_worker.py` — small module:
  - Background **reader thread** per camera: `cap.read()` paced to **~15 FPS, 640×480** (set `CAP_PROP_FRAME_WIDTH/HEIGHT`),
    writes into a **latest-frame-only buffer** (lock + single slot → stale frames discarded).
  - **Throttled inference** in the same worker: pose YOLO **~10 FPS**, object YOLO **~5–8 FPS** and only when an
    object context actually exists; skipped frames **reuse last persons/objects/tracks** for drawing + UI.
  - Thread-safe shared results dict: `{annotated_rgb, persons, objects, tracks, task_result, fps stats, timestamp}`.
  - **Duplicate-worker guard**: module-level singleton keyed by source + `threading.Event` stop signal + id stored in
    `st.session_state` → exactly ONE worker; Start on a running worker is a no-op.
- `camera.py` — only the live part of `show_task_verification` / `_task_verification_batch`:
  - Render loop becomes an `@st.fragment` (or `st.rerun(scope="fragment")`) that only refreshes the frame placeholder,
    metrics, detection banner, status column, task progress — **one persistent camera display + all existing panels stay**.
  - Start/Stop buttons do a full rerun only when the camera state actually changes; Stop signals the worker event,
    joins briefly, releases the camera (keep `release_task_camera()`).
  - Remove the blocking `time.sleep(1.0)` (banner timing uses existing `tv_completed_at` timestamp inside the fragment).
  - Do NOT auto-create the object context in monitor-only mode (pose-only runs skip object YOLO).
  - History logging (1 s), voice alert hooks, `verify_task_step` cadence, track drawing, `_render_task_status` — all kept.
- `pose_detection.py`, `object_detection.py`, `app.py`, task logic modules — **unchanged**.

**Do NOT:** fake/duplicate results, hide lag, remove task panels, touch the `AI-Recognition-System/` copy.

## 4. Test plan (after implementation)

1. `python test_task_verification.py`, `python test_object_tasks.py`, `python test_walking_fix.py` — no regression.
2. Manual `streamlit run app.py`: Start/Stop Camera repeatedly → exactly ONE worker thread (log/assert worker count),
   no duplicate `VideoCapture` on index 0.
3. Confirm ~15 FPS capture, pose ~10 FPS, objects ~5–8 FPS (show real FPS counters on screen — no faking).
4. Activity detection + sequential step + multi-step task still complete; confirmation streak works.
5. No blocking sleep — Stop/Reset/New-Task clicks respond immediately.

## 5. Progress log

- [x] Read root `app.py`, `camera.py` (full), `pose_detection.py`, `object_detection.py`, `step_validator.py`,
      `desktop_app.py`, test files; searched all `cap.read` / `VideoCapture` / `sleep` / `rerun` sites.
- [x] Verified Streamlit 1.63 APIs (`st.fragment`, `st.rerun(scope=...)` signatures).
- [x] Root causes documented (above).
- [x] Implement `camera_worker.py` — reader thread (~15 FPS capture, 640x480), pose YOLO ~10 FPS,
      object YOLO ~7 FPS only when a context exists, last-good reuse between throttle gaps,
      latest-writer-wins snapshot, real FPS counters, duplicate-worker singleton guard
      (start-on-running is a no-op + the extra cap is released).
- [x] Refactor `camera.py` live section — the live area is an `@st.fragment(run_every="0.25s")`;
      the fragment only renders the worker's latest snapshot + panels (no full-page rerun);
      worker auto-heal if it died; history drained on the UI thread; `tv_tracks` synced for
      the status expander; `time.sleep(1.0)` banner removed (timestamp-driven, ~2.5 s);
      monitor-only mode no longer auto-creates an object context (pose-only runs skip object YOLO);
      Stop Camera joins + releases through `stop_active_worker()`; old `_task_verification_batch` deleted.
- [x] Tests: NEW `test_camera_worker.py` — 21/21 PASS (snapshots + real FPS, throttle ordering,
      singleton no-op, stop/join/release, idempotent stop). Regressions: `test_task_verification.py`
      50/50, `test_object_tasks.py` 55/55, `test_walking_fix.py` ALL PASS. `camera.py` imports, `app.py` compiles.
- [ ] User: run `streamlit run app.py` -> Live Monitor -> Task Verification: confirm on-screen FPS
      counters (~15 capture / ~10 pose / ~7 obj with a context), Start/Stop/Clear click latency,
      and exactly one worker per Start (log line `[camera_worker] ...` only on camera loss).
