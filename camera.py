import threading
import sys
import cv2
import streamlit as st
import time
from datetime import datetime
import pyttsx3
import tempfile
import os

from pose_detection import analyze_frame, reset_tracker
from object_detection import detect_objects
from object_tracking import ObjectTracker, draw_tracks
from object_tasks import ObjectTaskContext, get_task_requirements
from step_validator import StepValidator
from camera_worker import (
    CameraWorker,
    start_worker,
    stop_active_worker,
    empty_task_result,
)
from task_detection import (
    TaskVerificationSession,
    parse_task_text,
    verify_task_step,
    ACTION_LABELS,
)
from task_eval import LiveTaskEvaluator, build_report


# --- Safe Voice Alert (Linux / Cloud Safe) ---
def get_voice_engine():
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        return engine
    except Exception:
        return None


def speak_alert(message):
    def _speak():
        try:
            engine = get_voice_engine()
            if engine:
                engine.say(message)
                engine.runAndWait()
                engine.stop()
        except Exception:
            pass

    threading.Thread(target=_speak, daemon=True).start()


# --- Parsing & Context Helpers ---
def _parse_any(text):
    action = parse_task_text(text)
    if action is not None:
        return "pose", action, ACTION_LABELS[action], {}

    try:
        from object_tasks import (
            parse_object_task,
            UNSUPPORTED_TASKS,
            TASK_LABELS,
        )
        key, detail = parse_object_task(text)
    except ImportError:
        return None, None, None, {}

    if key is None:
        return None, None, None, {}

    if detail.get("unsupported"):
        return (
            "unsupported", key,
            UNSUPPORTED_TASKS[key]["label"],
            UNSUPPORTED_TASKS[key],
        )

    return "object", key, TASK_LABELS.get(key, key), detail


def _texts_need_objects(texts):
    for text in texts:
        if not text or not text.strip():
            continue
        if parse_task_text(text) is not None:
            continue
        try:
            from object_tasks import parse_object_task
            key, detail = parse_object_task(text)
        except ImportError:
            return False
        if key is not None and not detail.get("unsupported"):
            return True
    return False


def _prepare_object_context(texts):
    if not _texts_need_objects(texts):
        st.session_state.tv_object_context = None
        return

    tracker = ObjectTracker()
    context = ObjectTaskContext(tracker)

    chosen_color = st.session_state.get("tv_opt_color")
    context.chosen_color = (
        None if chosen_color in (None, "(none)") else chosen_color
    )
    context.sorted_order = list(
        st.session_state.get("tv_opt_order") or []
    )

    try:
        context.required_count = int(
            st.session_state.get("tv_opt_count", 3)
        )
    except (TypeError, ValueError):
        context.required_count = None

    st.session_state.tv_object_context = context


OBJECT_BOX_COLOR = (0, 200, 255)


def draw_object_boxes(frame, objects):
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        label = f'{obj["class_name"]} {obj["confidence"] * 100:.0f}%'

        cv2.rectangle(frame, p1, p2, OBJECT_BOX_COLOR, 2)
        label_y = max(p1[1] - 6, 14)
        cv2.putText(
            frame, label, (p1[0], label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            OBJECT_BOX_COLOR, 1, cv2.LINE_AA,
        )
    return frame


# ============================================================
# 📷 1. SHOW CAMERA
# ============================================================
def show_camera():
    st.subheader("📷 Camera / Video Source")

    camera_type = st.radio(
        "Select Source",
        ["Upload Video File (Cloud Compatible)", "Laptop Camera (Local)", "Connect with Mobile"],
        horizontal=True,
        key="main_cam_type"
    )

    uploaded_video = None
    if camera_type == "Upload Video File (Cloud Compatible)":
        uploaded_video = st.file_uploader(
            "Upload recorded video (.mp4, .mov, .avi)", 
            type=["mp4", "mov", "avi", "mkv"],
            key="main_cam_uploader"
        )

    col1, col2 = st.columns(2)
    with col1:
        start_camera = st.button("🟢 Start Processing", use_container_width=True, key="main_cam_start")
    with col2:
        stop_camera = st.button("🔴 Stop", use_container_width=True, key="main_cam_stop")

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    if stop_camera:
        st.session_state.camera_running = False
        st.session_state.system_status = "OFFLINE"
        st.rerun()

    if start_camera:
        if camera_type == "Upload Video File (Cloud Compatible)" and uploaded_video is None:
            st.error("⚠️ Please upload a video file first.")
            return
        st.session_state.camera_running = True
        st.session_state.system_status = "ONLINE"

    if not st.session_state.camera_running:
        st.info("Select a source and press 🟢 Start Processing.")
        return

    frame_placeholder = st.empty()
    activity_metric = st.empty()

    if camera_type == "Upload Video File (Cloud Compatible)":
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.getvalue())
        cap = cv2.VideoCapture(tfile.name)
    elif camera_type == "Laptop Camera (Local)":
        cap = cv2.VideoCapture(0)
    else:
        mobile_ip = st.session_state.get("tv_mobile_ip", "")
        cap = cv2.VideoCapture(f"http://{mobile_ip}:8080/video")

    if not cap.isOpened():
        st.error("❌ Could not open video/camera stream.")
        st.session_state.camera_running = False
        return

    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.info("ℹ️ Video finished.")
            break
        detected_frame, persons = analyze_frame(frame)
        _, objects = detect_objects(frame)
        detected_frame = draw_object_boxes(detected_frame, objects)
        frame_placeholder.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        time.sleep(0.02)

    cap.release()
    st.session_state.camera_running = False


# ============================================================
# 📝 2. TASK EVALUATION
# ============================================================
def show_task_evaluation():
    st.subheader("📝 Task Evaluation")
    st.info("Use the Task Verification section below to run real-time evaluation.")


# ============================================================
# ✅ 3. MAIN LIVE MONITOR: TASK VERIFICATION (FULL RESTORED)
# ============================================================
def _apply_capture_profile(cap):
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15.0)
    except Exception:
        pass


def _task_camera(cap_type):
    if cap_type == "Upload Video File (Cloud Compatible)":
        up_file = st.session_state.get("tv_uploaded_file")
        if up_file is None:
            return None
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(up_file.getvalue())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
    elif cap_type == "Laptop Camera":
        if sys.platform == "win32":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(0)
    else:
        mobile_ip = st.session_state.get("tv_mobile_ip", "")
        port = int(st.session_state.get("tv_port", 8080))
        if not mobile_ip:
            return None
        cap = cv2.VideoCapture(f"http://{mobile_ip}:{port}/video")

    if cap is not None and cap.isOpened():
        _apply_capture_profile(cap)
        return cap
    return None


def _render_task_status(task_session, result, object_context=None):
    if result["all_completed"]:
        st.success("🎉 ALL TASKS COMPLETED")
    elif result["current_step"]:
        st.markdown(f"**CURRENT TASK**\n\n**{result['current_step']}**")

    if result["all_completed"]:
        st.success("✓ TASK COMPLETED")
    elif result["step_completed"]:
        st.success(f"✓ Step {result['completed_step_number']} COMPLETED")
    elif result["current_step"] is None:
        st.info("ℹ️ Start a task to begin verification.")
    elif result["state"] == "IN PROGRESS":
        if result["detected"]:
            st.warning(f"● IN PROGRESS — action detected! ({result['streak']}/{result['required_frames']} frames)")
        elif result["streak"] > 0:
            st.warning(f"● IN PROGRESS — hold pose ({result['streak']}/{result['required_frames']} frames)")
        else:
            st.info("● IN PROGRESS — perform the action now")
    else:
        st.info("● WAITING FOR PERSON")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Live Confidence", f"{result['confidence']:.1f}%")
    with c2:
        st.metric("Confirmation", f"{result['streak']}/{result['required_frames']}")

    st.progress(max(0.0, min(1.0, result["streak"] / max(1, result["required_frames"]))))

    with st.expander("🧠 Why this status? (AI reasoning)"):
        for reason in result["reasons"][:5]:
            st.caption(f"• {reason}")

    if task_session is not None and task_session.mode == "multi":
        st.subheader("TASK PROGRESS")
        for step in task_session.steps_view():
            if step["state"] == "COMPLETED":
                st.success(f"✓ Step {step['number']} — {step['text']} ({step['confidence']:.0f}%)")
            elif step["state"] == "IN PROGRESS":
                st.warning(f"→ Step {step['number']} — {step['text']} — IN PROGRESS")
            else:
                st.info(f"○ Step {step['number']} — {step['text']} — LOCKED")


def show_task_verification():
    st.subheader("✅ Task Verification")

    if "task_history" not in st.session_state:
        st.session_state.task_history = []

    if st.session_state.pop("tv_camera_lost", False):
        st.error("❌ Video/Camera stopped sending frames. Press START CAMERA to run again.")

    # 1. Task Type Input
    task_type = st.radio("Task Type", ["Single Task", "Multi-Step Task"], horizontal=True, key="tv_task_type")
    task_texts = []

    if task_type == "Single Task":
        task_text = st.text_input(
            "Enter Task",
            value=st.session_state.get("tv_single_text", "Raise your right hand"),
            key="tv_single_input"
        )
        st.session_state.tv_single_text = task_text
        if task_text.strip():
            task_texts.append(task_text)
            kind, key, label, detail = _parse_any(task_text)
            if kind == "pose":
                st.success(f"🤖 AI understood: **{label}** (pose task)")
            elif kind == "object":
                st.success(f"🤖 AI understood: **{label}** (object task)")
    else:
        if "tv_step_count" not in st.session_state:
            st.session_state.tv_step_count = 2
        step_count = st.session_state.tv_step_count

        for i in range(step_count):
            task_texts.append(
                st.text_input(
                    f"Step {i + 1}",
                    value=st.session_state.get(f"tv_step_{i}", "Stand up" if i == 0 else "Raise your right hand"),
                    key=f"tv_step_input_{i}"
                )
            )

        add_col, remove_col = st.columns(2)
        with add_col:
            if st.button("＋ Add Step", use_container_width=True):
                st.session_state.tv_step_count = min(8, step_count + 1)
                st.rerun()
        with remove_col:
            if st.button("－ Remove Step", use_container_width=True):
                st.session_state.tv_step_count = max(2, step_count - 1)
                st.rerun()

    st.session_state.tv_steps_text = task_texts

    # 2. Source Selection (Cloud Compatible Video Upload added)
    st.subheader("📷 Camera / Video Source")
    camera_type = st.radio(
        "Select Source",
        ["Upload Video File (Cloud Compatible)", "Laptop Camera", "Connect with Mobile"],
        horizontal=True,
        key="tv_camera_type"
    )

    if camera_type == "Upload Video File (Cloud Compatible)":
        st.file_uploader(
            "Upload action video or shoot from phone (.mp4, .mov)", 
            type=["mp4", "mov", "avi", "mkv"],
            key="tv_uploaded_file"
        )
    elif camera_type == "Connect with Mobile":
        st.session_state.tv_mobile_ip = st.text_input("Mobile IP Address", value=st.session_state.get("tv_mobile_ip", ""), placeholder="192.168.1.5")
        st.session_state.tv_port = st.number_input("Port", value=int(st.session_state.get("tv_port", 8080)))

    # 3. Action Buttons
    if "tv_cam_on" not in st.session_state:
        st.session_state.tv_cam_on = False

    cam_col, _ = st.columns([1, 2])
    with cam_col:
        if not st.session_state.tv_cam_on:
            start_cam_pressed = st.button("📷 START PROCESSING", use_container_width=True, type="primary", key="tv_cam_start_btn")
            stop_cam_pressed = False
        else:
            start_cam_pressed = False
            stop_cam_pressed = st.button("⏹️ STOP PROCESSING", use_container_width=True, key="tv_cam_stop_btn")

    if start_cam_pressed:
        if camera_type == "Upload Video File (Cloud Compatible)" and st.session_state.get("tv_uploaded_file") is None:
            st.error("⚠️ Pehle video upload karein.")
        else:
            st.session_state.tv_cam_on = True
            st.session_state.task_session = None
            reset_tracker()
            st.rerun()

    if stop_cam_pressed:
        st.session_state.tv_cam_on = False
        release_task_camera()
        st.session_state.tv_running = False
        st.session_state.task_session = None
        st.rerun()

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        start_pressed = st.button("🎯 Start / Verify Task", use_container_width=True, type="primary", disabled=not any(t.strip() for t in task_texts))
    with btn_col2:
        reset_pressed = st.button("🔄 Reset Task", use_container_width=True)
    with btn_col3:
        clear_pressed = st.button("🧹 CLEAR TASK", use_container_width=True)

    if start_pressed:
        texts = [t.strip() for t in task_texts if t.strip()]
        st.session_state.task_session = TaskVerificationSession(texts)
        st.session_state.tv_running = True
        _prepare_object_context(texts)
        st.rerun()

    if clear_pressed:
        st.session_state.task_session = None
        st.session_state.tv_running = False
        st.rerun()

    if reset_pressed and st.session_state.get("task_session") is not None:
        st.session_state.task_session.reset()
        st.session_state.tv_running = True
        st.rerun()

    if not st.session_state.tv_cam_on:
        st.info("⚪ Press **📷 START PROCESSING** above to begin continuous AI monitoring.")
        return

    # Layout for Running State
    main_col, status_col = st.columns([2, 1])
    with status_col:
        status_placeholder = st.empty()
    with main_col:
        frame_placeholder = st.empty()
        m1, m2, m3 = st.columns(3)
        with m1: activity_metric = st.empty()
        with m2: streak_metric = st.empty()
        with m3: progress_metric = st.empty()
        detection_placeholder = st.empty()

    status_placeholder.info("Processing: ⚪ starting…")
    frame_placeholder.info("Waiting for video frames…")
    activity_metric.metric("Activity", "—")
    streak_metric.metric("Confirmation", "—")
    progress_metric.metric("Progress", "0%")
    detection_placeholder.info("🤖 AI engine running…")

    @st.fragment(run_every="0.25s")
    def _tv_live_fragment():
        _tv_render_live_frame(
            status_placeholder, frame_placeholder,
            activity_metric, streak_metric,
            progress_metric, detection_placeholder,
        )

    _tv_live_fragment()


def _tv_open_camera(camera_kind):
    return _task_camera(camera_kind)


def _tv_render_live_frame(
    status_placeholder, frame_placeholder,
    activity_metric, streak_metric,
    progress_metric, detection_placeholder,
):
    worker = CameraWorker.get_active()
    camera_kind = st.session_state.get("tv_camera_type", "Upload Video File (Cloud Compatible)")

    if worker is None:
        cap = st.session_state.get("tv_cap")
        if cap is None or not cap.isOpened() or st.session_state.get("tv_cap_kind") != camera_kind:
            if cap is not None:
                try: cap.release()
                except Exception: pass
            cap = _tv_open_camera(camera_kind)
            if cap is None:
                status_placeholder.error("❌ Video/Camera could not be opened.")
                return
            st.session_state.tv_cap = cap
            st.session_state.tv_cap_kind = camera_kind
            reset_tracker()

        worker, _started = start_worker(cap, camera_kind, on_camera_lost=_tv_on_camera_lost)

    shared = worker._shared
    shared.task_session = st.session_state.get("task_session")
    shared.object_context = st.session_state.get("tv_object_context")
    shared.marked_location = st.session_state.get("tv_marked_location")

    if not shared.running or shared.camera_lost:
        release_task_camera()
        st.session_state.tv_running = False
        st.session_state.tv_camera_lost = True
        st.rerun(scope="fragment")

    snapshot = shared.latest()
    if snapshot is None:
        status_placeholder.success("Processing: 🟢 STARTING...")
        return

    annotated = snapshot["annotated"]
    persons = snapshot["persons"]
    result = snapshot["task_result"]

    try:
        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    except Exception:
        return

    status_placeholder.success("Processing: 🟢 LIVE")
    if persons:
        activity_metric.metric("Activity", persons[0]["activity"])
    else:
        activity_metric.metric("Activity", "No Person")

    streak_metric.metric("Confirmation", f"{result['streak']}/{result['required_frames']}")
    progress_metric.metric("Progress", f"{int(result['progress'] * 100)}%")

    if result["step_completed"]:
        detection_placeholder.success(f"✓ Step {result['completed_step_number']} COMPLETED!")
    elif result["detected"]:
        detection_placeholder.warning("🟢 Action detected — hold steady!")
    else:
        reason = result["reasons"][0] if result["reasons"] else "perform the step."
        detection_placeholder.info(f"🤖 {reason}")

    task_session = st.session_state.get("task_session")
    if task_session is not None:
        _render_task_status(task_session, result, shared.object_context)


def _tv_on_camera_lost():
    print("[camera_worker] stream ended")


def release_task_camera():
    stop_active_worker()
    st.session_state.pop("tv_cap", None)
    st.session_state.pop("tv_cap_kind", None)
