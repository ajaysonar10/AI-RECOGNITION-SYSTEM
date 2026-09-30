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


# --- Safe Voice Alert (Cloud Crash Fix) ---
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
        except Exception as e:
            print("Voice alert error:", e)

    threading.Thread(target=_speak, daemon=True).start()


# --- Functions ---
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
TV_BATCH_FRAMES = 5


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
# 📷 1. MAIN CAMERA / VIDEO PROCESSING (CLOUD COMPATIBLE)
# ============================================================
def show_camera():
    st.subheader("📷 Camera / Video Source")

    camera_type = st.radio(
        "Select Source",
        ["Upload Video File (Recommended for Cloud)", "Laptop Camera (Local)", "Connect with Mobile (Local WiFi)"],
        horizontal=True
    )

    uploaded_video = None
    if camera_type == "Upload Video File (Recommended for Cloud)":
        uploaded_video = st.file_uploader(
            "Upload recorded video or shoot from phone (.mp4, .mov, .avi)", 
            type=["mp4", "mov", "avi", "mkv"]
        )

    col1, col2 = st.columns(2)
    with col1:
        start_camera = st.button("🟢 Start Processing", use_container_width=True)
    with col2:
        stop_camera = st.button("🔴 Stop", use_container_width=True)

    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    if "activity_stream" not in st.session_state:
        st.session_state.activity_stream = []
    if "activity_history" not in st.session_state:
        st.session_state.activity_history = []

    if stop_camera:
        st.session_state.camera_running = False
        st.session_state.system_status = "OFFLINE"
        st.rerun()

    if start_camera:
        if camera_type == "Upload Video File (Recommended for Cloud)" and uploaded_video is None:
            st.error("⚠️ Pehle video file upload karein ya phone se record karein.")
            return
        st.session_state.camera_running = True
        st.session_state.system_status = "ONLINE"

    if not st.session_state.camera_running:
        st.info("Source chunein aur 🟢 Start Processing par click karein.")
        return

    frame_placeholder = st.empty()
    st.divider()

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1: activity_metric = st.empty()
    with metric_col2: confidence_metric = st.empty()
    with metric_col3: system_metric = st.empty()
    with metric_col4: objects_metric = st.empty()

    st.subheader("Current AI Detection")
    progress_placeholder = st.empty()
    detection_placeholder = st.empty()

    st.subheader("🎯 Sequential Step Evaluation")
    step_col1, step_col2, step_col3 = st.columns(3)
    with step_col1: current_step_placeholder = st.empty()
    with step_col2: step_status_placeholder = st.empty()
    with step_col3: step_message_placeholder = st.empty()

    st.subheader("📦 Objects in Frame")
    objects_placeholder = st.empty()

    st.subheader("👤 Person Position & Justification")
    analysis_placeholder = st.empty()

    st.subheader("Live Activity Stream")
    stream_placeholder = st.empty()

    # Capture open logic
    if camera_type == "Upload Video File (Recommended for Cloud)":
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
    elif camera_type == "Laptop Camera (Local)":
        cap = cv2.VideoCapture(0)
    else:
        mobile_ip = st.session_state.get("tv_mobile_ip", "")
        cap = cv2.VideoCapture(f"http://{mobile_ip}:8080/video")

    if not cap.isOpened():
        system_metric.metric("System", "OFFLINE")
        st.error("❌ Video/Camera could not be opened.")
        st.session_state.camera_running = False
        return

    last_log_time = 0
    voice_engine = get_voice_engine()
    last_voice_time = 0
    frame_count = 0

    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.info("ℹ️ Video processing complete.")
            break

        detected_frame, persons = analyze_frame(frame)
        _, objects = detect_objects(frame)
        detected_frame = draw_object_boxes(detected_frame, objects)
        frame_count += 1

        if persons:
            main_person = persons[0]
            activity = main_person["activity"]
            confidence = main_person["confidence"]
        else:
            activity = "No Person"
            confidence = 0.0

        if "step_validator" not in st.session_state:
            st.session_state.step_validator = StepValidator()
        validator = st.session_state.step_validator

        if activity != "No Person":
            result = validator.check_activity(activity)
            st.session_state.step_status = result["status"]
            st.session_state.step_message = result["message"]
            if result["status"] == "WRONG" and voice_engine:
                now = time.time()
                if now - last_voice_time >= 3:
                    try:
                        voice_engine.say(result["message"])
                        voice_engine.runAndWait()
                    except Exception:
                        pass
                    last_voice_time = now
        else:
            st.session_state.step_status = "WAITING"
            st.session_state.step_message = "Person not detected."

        current_step = validator.get_current_step()
        current_step_placeholder.metric("Current Step", "🎉 COMPLETED" if current_step is None else current_step)

        if st.session_state.step_status == "CORRECT": step_status_placeholder.success("✅ CORRECT")
        elif st.session_state.step_status == "WRONG": step_status_placeholder.error("❌ WRONG")
        elif st.session_state.step_status == "COMPLETED": step_status_placeholder.success("🎉 COMPLETED")
        else: step_status_placeholder.info("⏳ WAITING")

        step_message_placeholder.write(st.session_state.step_message)
        activity_metric.metric("Current Activity", activity)
        confidence_metric.metric("Confidence", f"{confidence:.1f}%")
        system_metric.metric("System", "ONLINE")
        objects_metric.metric("Objects", len(objects))
        progress_placeholder.progress(max(0.0, min(1.0, confidence / 100)))

        # Justification
        if persons:
            analysis_blocks = []
            for p in persons:
                reasons_text = "\n".join(f"   • {reason}" for reason in p["justification"])
                pos = p["position"]
                analysis_blocks.append(f"**👤 Person ID {p['track_id']}** — `{p['activity']}` ({p['confidence']:.0f}%)\n\n- **Position:** {pos['description']}\n- **Justification:**\n{reasons_text}")
            analysis_placeholder.markdown("\n\n---\n\n".join(analysis_blocks))

        frame_placeholder.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        time.sleep(0.02)

    cap.release()
    st.session_state.camera_running = False


# ============================================================
# 📝 2. LIVE TASK EVALUATION (CLOUD COMPATIBLE)
# ============================================================
def show_task_evaluation():
    st.subheader("📝 Task")
    task = st.text_area(
        "What should the person do?",
        placeholder="Example: Pick up the bottle and place it on the table",
        key="task_eval_input"
    )

    if task.strip():
        from task_eval import parse_task
        preview = parse_task(task)
        p1, p2, p3 = st.columns(3)
        with p1: st.metric("Activities", ", ".join(preview["activities"]) or "None")
        with p2: st.metric("Objects", ", ".join(preview["objects"]) or "None")
        with p3: st.metric("Interaction", "Required" if preview["interaction_required"] else "Optional")

    st.subheader("📷 Video / Camera Source")
    camera_type = st.radio(
        "Select Source",
        ["Upload Video File (Cloud)", "Laptop Camera (Local)"],
        horizontal=True,
        key="task_eval_camera_type"
    )

    uploaded_eval_vid = None
    if camera_type == "Upload Video File (Cloud)":
        uploaded_eval_vid = st.file_uploader(
            "Upload task video (.mp4, .mov, .avi)", 
            type=["mp4", "mov", "avi"],
            key="task_eval_upload"
        )

    if "task_eval_running" not in st.session_state:
        st.session_state.task_eval_running = False
    if "task_evaluator" not in st.session_state:
        st.session_state.task_evaluator = None
    if "task_eval_verdict" not in st.session_state:
        st.session_state.task_eval_verdict = None

    col1, col2 = st.columns(2)
    with col1:
        start_eval = st.button("🟢 Start Task Evaluation", use_container_width=True, disabled=not task.strip())
    with col2:
        stop_eval = st.button("🔴 Stop & Get Verdict", use_container_width=True)

    if start_eval:
        st.session_state.task_evaluator = LiveTaskEvaluator(task)
        st.session_state.task_eval_verdict = None
        st.session_state.task_eval_running = True
        st.rerun()

    if stop_eval and st.session_state.task_eval_running:
        st.session_state.task_eval_running = False
        evaluator = st.session_state.task_evaluator
        if evaluator is not None and evaluator.frames_processed > 0:
            st.session_state.task_eval_verdict = evaluator.evaluate()

    if not st.session_state.task_eval_running and st.session_state.task_eval_verdict is not None:
        _show_verdict(st.session_state.task_eval_verdict, st.session_state.task_evaluator)
        return

    if st.session_state.task_eval_running:
        evaluator = st.session_state.task_evaluator
        if camera_type == "Upload Video File (Cloud)":
            if uploaded_eval_vid is None:
                st.error("❌ Please upload a video file first.")
                st.session_state.task_eval_running = False
                return
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_eval_vid.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Video/Camera could not be opened.")
            st.session_state.task_eval_running = False
            return

        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        completed_automatically = False
        while st.session_state.task_eval_running:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, status = evaluator.step(frame)
            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            progress_value = evaluator.progress()
            progress_bar.progress(progress_value)
            status_text.write(f"**{status}** — {progress_value * 100:.0f}%")
            if evaluator.finished:
                completed_automatically = True
                break

        cap.release()
        st.session_state.task_eval_running = False
        if evaluator.frames_processed > 0:
            st.session_state.task_eval_verdict = evaluator.evaluate()
        st.rerun()


def _checklist_markdown(evaluator):
    lines = []
    for check in evaluator.verdict_checklist():
        state = check["state"]
        icon = "✅" if state in ("confirmed", "interacted") else ("⚠️" if state == "seen" else "⬜")
        label = check.get("activity") or check.get("object")
        lines.append(f"{icon} **{label}**")
    return "\n\n".join(lines) or "_No requirements._"


def _show_verdict(verdict, evaluator):
    st.divider()
    st.header("📊 Task Verdict")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Score", f"{verdict['score']}%")
    with c2: st.metric("Progress", f"{verdict['progress'] * 100:.0f}%")
    with c3: st.metric("Frames Analyzed", verdict["stats"]["frames_processed"])

    if verdict["status"] == "TASK COMPLETED": st.success("✅ " + verdict["status"])
    else: st.info("ℹ️ " + verdict["status"])
    st.write(verdict["explanation"])


# ============================================================
# ✅ 3. TASK VERIFICATION
# ============================================================
def _apply_capture_profile(cap):
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15.0)
    except Exception:
        pass


def _task_camera(cap_type):
    if cap_type == "Laptop Camera":
        cap = cv2.VideoCapture(0)
    else:
        mobile_ip = st.session_state.get("tv_mobile_ip", "")
        port = int(st.session_state.get("tv_port", 8080))
        if not mobile_ip: return None
        cap = cv2.VideoCapture(f"http://{mobile_ip}:{port}/video")

    if cap is not None and cap.isOpened():
        _apply_capture_profile(cap)
        return cap
    return None


def show_task_verification():
    st.subheader("✅ Task Verification")
    st.info("Task verification is active. Use Laptop Camera or Connect with Mobile locally, or use Task Evaluation tab above for Uploaded Video.")
