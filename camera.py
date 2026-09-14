from pose_detection import analyze_frame, reset_tracker
from object_detection import detect_objects
from object_tracking import ObjectTracker, draw_tracks
from object_tasks import ObjectTaskContext, get_task_requirements
from step_validator import StepValidator
from task_detection import (
    TaskVerificationSession,
    parse_task_text,
    verify_task_step,
    ACTION_LABELS,
)

import threading
import sys
import cv2
import streamlit as st
import time
from datetime import datetime
import pyttsx3

from pose_detection import analyze_frame
from object_detection import detect_objects


#functions
def _parse_any(text):
    """
    Parse the text with the pose-action OR object-task library.

    Returns (kind, key, label, detail):
        kind: "pose" | "object" | "unsupported" | None
    """
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
    """
    Kya koi text object-interaction task hai?
    (The object options UI + per-frame object detection run only then.)
    """
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
    """
    Build a fresh ObjectTaskContext + tracker for object-interaction
    tasks (on Start/Reset). In pose-only runs the context stays None —
    the live loop then skips the extra YOLO-object inference (same
    performance as before).
    """
    if not _texts_need_objects(texts):
        st.session_state.tv_object_context = None
        return

    tracker = ObjectTracker()
    context = ObjectTaskContext(tracker)

    # Config-screen options (widgets store their own keys)
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


def speak_alert(message):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(message)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("Voice alert error:", e)

    threading.Thread(target=_speak, daemon=True).start()
# ------------------------------------------------------------
# Helper to draw object boxes on the pose frame
# (a separate color, so they are not confused with pose/person boxes)
# ------------------------------------------------------------

OBJECT_BOX_COLOR = (0, 200, 255)   # BGR: orange

# Task verification: how many frames are processed per rerun.
# ~5 frames ≈ 1 second — Stop/Reset/New Task clicks take effect
# within this window, and the camera stays in session_state
# so the device is not reopened on every batch.
TV_BATCH_FRAMES = 5


def draw_object_boxes(frame, objects):
    """
    Draws the detected objects' boxes + labels on the frame.
    """

    for obj in objects:

        x1, y1, x2, y2 = obj["box"]

        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        label = (
            f'{obj["class_name"]} '
            f'{obj["confidence"] * 100:.0f}%'
        )

        cv2.rectangle(frame, p1, p2, OBJECT_BOX_COLOR, 2)

        label_y = max(p1[1] - 6, 14)

        cv2.putText(
            frame,
            label,
            (p1[0], label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            OBJECT_BOX_COLOR,
            1,
            cv2.LINE_AA,
        )

    return frame


def show_camera():

    st.subheader("📷 Camera Source")

    camera_type = st.radio(
        "Select Camera Source",
        ["Laptop Camera", "Connect with Mobile"],
        horizontal=True
    )

    col1, col2 = st.columns(2)

    with col1:
        start_camera = st.button(
            "🟢 Start Camera",
            use_container_width=True
        )

    with col2:
        stop_camera = st.button(
            "🔴 Stop Camera",
            use_container_width=True
        )

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
        st.session_state.camera_running = True
        st.session_state.system_status = "ONLINE"

    if not st.session_state.camera_running:
        st.info(
            "To start Camera 🟢 Press Start Camera."
        )
        return

    frame_placeholder = st.empty()

    st.divider()

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        activity_metric = st.empty()

    with metric_col2:
        confidence_metric = st.empty()

    with metric_col3:
        system_metric = st.empty()

    with metric_col4:
        objects_metric = st.empty()

    st.subheader("Current AI Detection")

    progress_placeholder = st.empty()
    detection_placeholder = st.empty()

    st.subheader("🎯 Sequential Step Evaluation")

    st.subheader("🎯 Sequential Step Evaluation")

    step_col1, step_col2, step_col3 = st.columns(3)

    with step_col1:
        current_step_placeholder = st.empty()

    with step_col2:
        step_status_placeholder = st.empty()

    with step_col3:
        step_message_placeholder = st.empty()

    st.subheader("📦 Objects in Frame")

    objects_placeholder = st.empty()

    st.subheader("👤 Person Position & Justification")

    analysis_placeholder = st.empty()

    st.subheader("Live Activity Stream")

    stream_placeholder = st.empty()

    activity_metric.metric("Current Activity", "Detecting...")
    confidence_metric.metric("Confidence", "0.0%")
    system_metric.metric("System", "ONLINE")
    objects_metric.metric("Objects", "0")
    progress_placeholder.progress(0)
    detection_placeholder.info("🤖 AI camera detection running...")
    objects_placeholder.info("📦 Object detection active — boxes will be shown on the frame.")

    if camera_type == "Laptop Camera":

        st.info("💻 Laptop camera connecting...")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            system_metric.metric("System", "OFFLINE")
            st.error("❌ Laptop camera could not be opened.")
            st.session_state.camera_running = False
            return

    else:

        st.info(
            "📱 Mobile and laptop must be on the same Wi-Fi network."
        )

        mobile_ip = st.text_input(
            "Mobile IP Address",
            placeholder="Example: 192.168.1.5"
        )

        port = st.number_input(
            "Port",
            min_value=1,
            max_value=65535,
            value=8080
        )

        if not mobile_ip:
            st.warning("⚠️ Enter the Mobile IP Address first.")
            return

        video_url = f"http://{mobile_ip}:{port}/video"

        st.caption(f"Stream URL: {video_url}")

        cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():

            system_metric.metric("System", "OFFLINE")

            st.error(
                "❌ Mobile camera could not connect. "
                "Check the IP address, port and Wi-Fi connection."
            )

            st.session_state.camera_running = False
            return

        st.success("🟢 Mobile camera connected!")

    last_log_time = 0
    voice_engine = pyttsx3.init()
    voice_engine.setProperty("rate", 150)
    last_voice_time = 0
    frame_count = 0

    while st.session_state.camera_running:

        ret, frame = cap.read()

        if not ret:

            system_metric.metric("System", "OFFLINE")

            st.error("❌ Camera frames are not being received.")
            break

        # ----------------------------------------------
        # Naya full analysis (position + activity +
        # justification per person)
        # ----------------------------------------------

        detected_frame, persons = analyze_frame(frame)

        # ----------------------------------------------
        # Object detection (saare 80 COCO objects)
        # ----------------------------------------------

        _, objects = detect_objects(frame)

        detected_frame = draw_object_boxes(
            detected_frame,
            objects
        )

        frame_count += 1

        # ----------------------------------------------
        # Metrics (pehle person ke basis par)
        # ----------------------------------------------

        if persons:

            main_person = persons[0]

            activity = main_person["activity"]
            confidence = main_person["confidence"]

        else:

            activity = "No Person"
            confidence = 0.0

        # ============================================================
        # SEQUENTIAL STEP VALIDATION
        # ============================================================

        if "step_validator" not in st.session_state:
            st.session_state.step_validator = StepValidator()

        validator = st.session_state.step_validator

        if activity != "No Person":

            result = validator.check_activity(activity)

            st.session_state.step_status = result["status"]
            st.session_state.step_message = result["message"]

            # 🔊 VOICE ALERT FOR WRONG ACTIVITY
            if result["status"] == "WRONG":

                now = time.time()

                if now - last_voice_time >= 3:

                    voice_engine.say(result["message"])
                    voice_engine.runAndWait()

                    last_voice_time = now

        else:

            st.session_state.step_status = "WAITING"
            st.session_state.step_message = "Person not detected."

        current_step = validator.get_current_step()

        if current_step is None:
            current_step_text = "🎉 COMPLETED"
        else:
            current_step_text = current_step

        current_step_placeholder.metric(
            "Current Step",
            current_step_text
        )

        if st.session_state.step_status == "CORRECT":
            step_status_placeholder.success("✅ CORRECT")

        elif st.session_state.step_status == "WRONG":
            step_status_placeholder.error("❌ WRONG")

        elif st.session_state.step_status == "COMPLETED":
            step_status_placeholder.success("🎉 COMPLETED")

        else:
            step_status_placeholder.info("⏳ WAITING")

        step_message_placeholder.write(
            st.session_state.step_message
        )

        activity_metric.metric(
            "Current Activity",
            activity
        )

        confidence_metric.metric(
            "Confidence",
            f"{confidence:.1f}%"
        )

        system_metric.metric(
            "System",
            "ONLINE"
        )

        objects_metric.metric(
            "Objects",
            len(objects)
        )

        progress_value = max(
            0.0,
            min(1.0, confidence / 100)
        )

        progress_placeholder.progress(progress_value)

        # ----------------------------------------------
        # Detection banner
        # ----------------------------------------------

        person_word = (
            "person" if len(persons) == 1 else "persons"
        )

        if activity == "Walking":
            detection_placeholder.success(
                f"🟢 Walking detected — {confidence:.1f}% confidence "
                f"({len(persons)} {person_word} in frame)"
            )
        elif activity == "Standing":
            detection_placeholder.success(
                f"🟢 Standing detected — {confidence:.1f}% confidence "
                f"({len(persons)} {person_word} in frame)"
            )
        elif activity == "Sitting":
            detection_placeholder.success(
                f"🟢 Sitting detected — {confidence:.1f}% confidence "
                f"({len(persons)} {person_word} in frame)"
            )
        elif activity == "No Person":
            detection_placeholder.warning(
                "🟡 No person detected in the frame."
            )
        else:
            detection_placeholder.warning(
                f"🟡 {activity} — {confidence:.1f}% confidence "
                f"({len(persons)} {person_word} in frame)"
            )

        # ----------------------------------------------
        # Person Position & Justification section
        # ----------------------------------------------

        if persons:

            analysis_blocks = []

            for p in persons:

                reasons_text = "\n".join(
                    f"   • {reason}"
                    for reason in p["justification"]
                )

                pos = p["position"]

                analysis_blocks.append(
                    f"**👤 Person ID {p['track_id']}** — "
                    f"`{p['activity']}` ({p['confidence']:.0f}%)\n\n"
                    f"- **Position:** {pos['description']} "
                    f"(center: {pos['center_pixel'][0]:.0f}, "
                    f"{pos['center_pixel'][1]:.0f})\n"
                    f"- **Justification:**\n\n"
                    f"{reasons_text}"
                )

            analysis_placeholder.markdown(
                "\n\n---\n\n".join(analysis_blocks)
            )

        else:

            analysis_placeholder.info(
                "Person position and justification will appear here "
                "when a person enters the frame."
            )

        # ----------------------------------------------
        # Objects summary (class-wise count)
        # ----------------------------------------------

        if objects:

            class_counts = {}

            for obj in objects:
                name = obj["class_name"]
                class_counts[name] = class_counts.get(name, 0) + 1

            summary_text = "  |  ".join(
                f"**{name}** x{count}"
                for name, count in class_counts.items()
            )

            objects_placeholder.markdown(
                f"📦 {len(objects)} objects: {summary_text}"
            )

        else:

            objects_placeholder.info(
                "📦 No object was detected in this frame.",
            )

        # ----------------------------------------------
        # History logging (1 second interval)
        # ----------------------------------------------

        current_time = time.time()

        # Save one real detection per second instead of every video frame.
        if current_time - last_log_time >= 1:

            now = datetime.now()

            timestamp = now.strftime("%H:%M:%S")
            date_text = now.strftime("%d %b %Y")

            # Treat Unknown/uncertain output as non-normal.
            status = (
                "Normal"
                if activity in ["Walking", "Standing", "Sitting"]
                else "Review"
            )

            history_item = {
                "date": date_text,
                "time": timestamp,
                "activity": activity,
                "confidence": float(confidence),
                "status": status
            }

            st.session_state.activity_history.insert(
                0,
                history_item
            )

            # Keep latest 200 records in memory.
            st.session_state.activity_history = (
                st.session_state.activity_history[:200]
            )

            log_entry = (
                f"🟢 {timestamp} — "
                f"{activity} — "
                f"{confidence:.1f}% "
                f"({len(persons)} {person_word})"
            )

            st.session_state.activity_stream.insert(
                0,
                log_entry
            )

            st.session_state.activity_stream = (
                st.session_state.activity_stream[:5]
            )

            last_log_time = current_time

        if st.session_state.activity_stream:

            stream_placeholder.markdown(
                "\n\n".join(
                    st.session_state.activity_stream
                )
            )

        # ----------------------------------------------
        # Frame display (BGR -> RGB)
        # ----------------------------------------------

        detected_frame = cv2.cvtColor(
            detected_frame,
            cv2.COLOR_BGR2RGB
        )

        frame_placeholder.image(
            detected_frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()

    system_metric.metric(
        "System",
        "OFFLINE"
    )


# ============================================================
# LIVE TASK EVALUATION
# (The user gives a task -> the AI verifies it from the live camera)
# ============================================================

from task_eval import LiveTaskEvaluator, build_report


def show_task_evaluation():

    st.subheader("📝 Task")

    task = st.text_area(
        "What should the person do? (perform it on the live camera)",
        placeholder="Example: Pick up the bottle and place it on the table",
        key="task_eval_input"
    )

    # Live parse preview — the user sees what the AI understood
    if task.strip():

        from task_eval import parse_task

        preview = parse_task(task)

        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric(
                "Activities",
                ", ".join(preview["activities"]) or "None"
            )

        with p2:
            st.metric(
                "Objects",
                ", ".join(preview["objects"]) or "None"
            )

        with p3:
            st.metric(
                "Interaction",
                "Required" if preview["interaction_required"] else "Optional"
            )

    else:
        st.info("Task likho — jaise: 'walk across the room', "
                "'sit on the chair', 'pick up the bottle'.")

    # ----------------------------------------------
    # Camera source
    # ----------------------------------------------

    st.subheader("📷 Camera Source")

    camera_type = st.radio(
        "Select Camera Source",
        ["Laptop Camera", "Connect with Mobile"],
        horizontal=True,
        key="task_eval_camera_type"
    )

    if camera_type == "Connect with Mobile":

        st.info("📱 Mobile and laptop must be on the same Wi-Fi network.")

        mobile_ip = st.text_input(
            "Mobile IP Address",
            placeholder="Example: 192.168.1.5",
            key="task_eval_mobile_ip"
        )

        port = st.number_input(
            "Port",
            min_value=1,
            max_value=65535,
            value=8080,
            key="task_eval_port"
        )

    # ----------------------------------------------
    # Session state init
    # ----------------------------------------------

    if "task_eval_running" not in st.session_state:
        st.session_state.task_eval_running = False

    if "task_evaluator" not in st.session_state:
        st.session_state.task_evaluator = None

    if "task_eval_verdict" not in st.session_state:
        st.session_state.task_eval_verdict = None

    col1, col2 = st.columns(2)

    with col1:
        start_eval = st.button(
            "🟢 Start Task Evaluation",
            use_container_width=True,
            disabled=not task.strip()
        )

    with col2:
        stop_eval = st.button(
            "🔴 Stop & Get Verdict",
            use_container_width=True
        )

    # ----------------------------------------------
    # Start
    # ----------------------------------------------

    if start_eval:

        if not task.strip():
            st.error("Please specify a task first.")
        else:
            st.session_state.task_evaluator = LiveTaskEvaluator(task)
            st.session_state.task_eval_verdict = None
            st.session_state.task_eval_running = True
            st.rerun()

    # ----------------------------------------------
    # Stop -> final verdict
    # ----------------------------------------------

    if stop_eval and st.session_state.task_eval_running:

        st.session_state.task_eval_running = False

        evaluator = st.session_state.task_evaluator

        if evaluator is not None and evaluator.frames_processed > 0:
            st.session_state.task_eval_verdict = evaluator.evaluate()

    # ----------------------------------------------
    # Verdict screen (stop/auto-complete ke baad)
    # ----------------------------------------------

    if (
        not st.session_state.task_eval_running
        and st.session_state.task_eval_verdict is not None
    ):

        _show_verdict(
            st.session_state.task_eval_verdict,
            st.session_state.task_evaluator
        )
        return

    # ----------------------------------------------
    # LIVE LOOP
    # ----------------------------------------------

    if st.session_state.task_eval_running:

        evaluator = st.session_state.task_evaluator

        # ------------------------------------------
        # Open the camera
        # ------------------------------------------

        if camera_type == "Laptop Camera":

            cap = cv2.VideoCapture(0)

        else:

            if not st.session_state.get("task_eval_mobile_ip"):
                st.error("❌ Enter the Mobile IP Address first.")
                st.session_state.task_eval_running = False
                return

            video_url = (
                f"http://{st.session_state.task_eval_mobile_ip}:"
                f"{int(st.session_state.task_eval_port)}/video"
            )

            st.caption(f"Stream URL: {video_url}")

            cap = cv2.VideoCapture(video_url)

        if not cap.isOpened():
            st.error("❌ Camera could not be opened.")
            st.session_state.task_eval_running = False
            return

        frame_placeholder = st.empty()

        progress_bar = st.progress(0)
        status_text = st.empty()

        m1, m2, m3 = st.columns(3)

        with m1:
            activity_metric = st.empty()

        with m2:
            elapsed_metric = st.empty()

        with m3:
            frames_metric = st.empty()

        st.subheader("✅ Requirement Checklist (live)")

        checklist_placeholder = st.empty()

        st.subheader("🧠 Live Justification")

        justify_placeholder = st.empty()

        auto_note = st.info(
            "🤖 Live evaluation running... Complete the task. "
            "Jaise hi saari requirements confirm ho jayengi, "
            "verdict AUTO dikh jayega. Ya 'Stop & Get Verdict' dabao."
        )

        completed_automatically = False

        while st.session_state.task_eval_running:

            ret, frame = cap.read()

            if not ret:
                st.error("❌ Camera frames are not being received.")
                break

            annotated, status = evaluator.step(frame)

            voice_message = evaluator.get_voice_alert()

            if voice_message:
                speak_alert(voice_message)
            # --------------------------------------
            # Frame display
            # --------------------------------------

            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

            # --------------------------------------
            # Metrics
            # --------------------------------------

            current_activity = (
                max(
                    evaluator.activity_best,
                    key=evaluator.activity_best.get
                )
                if evaluator.activity_best else "Detecting..."
            )

            activity_metric.metric(
                "Latest Activity",
                current_activity
            )

            elapsed_metric.metric(
                "Elapsed",
                f"{evaluator.stats_elapsed():.0f}s"
            )

            frames_metric.metric(
                "Frames",
                evaluator.frames_processed
            )

            progress_value = evaluator.progress()
            progress_bar.progress(progress_value)
            status_text.write(
                f"**{status}** — {progress_value * 100:.0f}%"
            )

            # --------------------------------------
            # Live checklist
            # --------------------------------------

            checklist_placeholder.markdown(
                _checklist_markdown(evaluator)
            )

            # --------------------------------------
            # Live justification (latest person)
            # --------------------------------------

            reasons_md = ""

            if evaluator.activity_reasons:

                latest_activity = max(
                    evaluator.activity_best,
                    key=evaluator.activity_best.get
                )

                reasons = evaluator.activity_reasons.get(
                    latest_activity, []
                )

                reasons_md = "\n".join(
                    f"- {reason}" for reason in reasons[:4]
                )

            justify_placeholder.markdown(
                reasons_md or "_Waiting for person..._"
            )

            # --------------------------------------
            # Auto-complete?
            # --------------------------------------

            if evaluator.finished:
                completed_automatically = True
                break

        cap.release()

        # ------------------------------------------
        # Loop finished -> show the verdict
        # ------------------------------------------

        st.session_state.task_eval_running = False

        if evaluator.frames_processed > 0:
            st.session_state.task_eval_verdict = evaluator.evaluate()

        if completed_automatically:
            st.success(
                "🎉 All requirements confirmed — task auto-completed!"
            )
        else:
            st.info("Evaluation stopped — here is the verdict.")

        st.rerun()


def _checklist_markdown(evaluator):
    """Builds the live checklist markdown."""

    lines = []

    for check in evaluator.verdict_checklist():

        state = check["state"]

        if state in ("confirmed", "interacted"):
            icon = "✅"
            note = "confirmed"
        elif state in ("seen",):
            icon = "⚠️"
            note = "seen only"
        else:
            icon = "⬜"
            note = "waiting"

        label = check.get("activity") or check.get("object")

        lines.append(
            f"{icon} **{label}** — {note}"
        )

    return "\n\n".join(lines) or "_No requirements parsed._"


def _show_verdict(verdict, evaluator):
    """Renders the final verdict + report."""

    st.divider()

    st.header("📊 Task Verdict")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Score", f"{verdict['score']}%")

    with c2:
        st.metric("Progress", f"{verdict['progress'] * 100:.0f}%")

    with c3:
        stats = verdict["stats"]
        st.metric("Frames Analyzed", stats["frames_processed"])

    if verdict["status"] == "TASK COMPLETED":
        st.success("✅ " + verdict["status"])
    elif verdict["status"] == "TASK PARTIALLY COMPLETED":
        st.warning("⚠️ " + verdict["status"])
    elif verdict["status"] == "TASK NOT COMPLETED":
        st.error("❌ " + verdict["status"])
    else:
        st.info("ℹ️ " + verdict["status"])

    st.write(verdict["explanation"])

    st.subheader("📝 Task")
    st.info(evaluator.task_text)

    st.subheader("🚶 Activities")

    if verdict["activity_checks"]:

        for check in verdict["activity_checks"]:

            if check["state"] == "confirmed":
                st.success(
                    f"✅ {check['activity']} — "
                    f"best confidence {check['best_confidence']}%"
                )
            elif check["state"] == "seen":
                st.warning(
                    f"⚠️ {check['activity']} seen briefly — "
                    f"best confidence {check['best_confidence']}%"
                )
            else:
                st.error(f"❌ {check['activity']} never observed")

            reasons = verdict.get("activity_reasons", {}).get(
                check["activity"]
            )

            if reasons:
                st.caption("Justification: " + " | ".join(reasons[:3]))

    else:
        st.info("No activity was required by this task.")

    st.subheader("📦 Objects")

    if verdict["object_checks"]:

        for check in verdict["object_checks"]:

            if check["state"] == "interacted":
                st.success(
                    f"✅ {check['object']} found near the person — "
                    f"confidence {check['best_confidence']}%"
                )
            elif check["state"] == "seen":
                st.warning(
                    f"⚠️ {check['object']} seen but never near the person — "
                    f"confidence {check['best_confidence']}%"
                )
            else:
                st.error(f"❌ {check['object']} not detected")

    else:
        st.info("No object was required by this task.")

    st.subheader("📄 Report")

    report = build_report(evaluator)

    st.code(report)

    st.download_button(
        "⬇️ Download Report",
        report,
        file_name="task_evaluation_report.txt",
        mime="text/plain"
    )


# ============================================================
# ✅ TASK VERIFICATION / TASK COMPLETION SYSTEM
# (Single task ya multi-step task — REAL camera + YOLO Pose
#  keypoints — no dummy values)
# ============================================================


def _task_camera(cap_type):
    """Opens the existing camera sources (laptop or mobile)."""

    if cap_type == "Laptop Camera":
        if sys.platform == "win32":
            # Windows: DirectShow backend MSMF se fast/reliable hai
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

        cap = cv2.VideoCapture(
            f"http://{mobile_ip}:{port}/video"
        )

    return cap if (cap is not None and cap.isOpened()) else None


def _render_task_status(task_session, result, object_context=None):
    """Shows the task status + multi-step progress beside the camera."""

    # ----------------------------------------------
    # CURRENT TASK
    # ----------------------------------------------

    if result["all_completed"]:
        st.success("🎉 ALL TASKS COMPLETED")
    elif result["current_step"]:
        st.markdown(
            f"**CURRENT TASK**\n\n"
            f"**{result['current_step']}**"
        )

    # ----------------------------------------------
    # STATUS (with the real-time streak)
    # ----------------------------------------------

    if result["all_completed"]:
        st.success("✓ TASK COMPLETED")

    elif result["step_completed"]:
        # The step completed on this frame (green flash)
        st.success(
            f"✓ Step {result['completed_step_number']} COMPLETED"
        )

    elif result["current_step"] is None:
        st.info("ℹ️ Start a task to begin verification.")

    elif result["state"] == "IN PROGRESS":

        if result["detected"]:
            st.warning(
                f"● IN PROGRESS — action detected! "
                f"Hold it ({result['streak']}/"
                f"{result['required_frames']} frames)"
            )
        elif result["streak"] > 0:
            st.warning(
                f"● IN PROGRESS — hold the pose "
                f"({result['streak']}/"
                f"{result['required_frames']} frames)"
            )
        else:
            st.info("● IN PROGRESS — perform the action now")

    else:
        st.info("● WAITING FOR PERSON")

    # ----------------------------------------------
    # Confidence + streak
    # ----------------------------------------------

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Live Confidence",
            f"{result['confidence']:.1f}%"
        )

    with c2:
        st.metric(
            "Confirmation",
            f"{result['streak']}/{result['required_frames']}"
        )

    st.progress(
        max(
            0.0,
            min(
                1.0,
                result["streak"] /
                max(1, result["required_frames"])
            )
        )
    )

    # ----------------------------------------------
    # WHY (real reasons — geometry se)
    # ----------------------------------------------

    with st.expander("🧠 Why this status? (AI reasoning)"):
        for reason in result["reasons"][:5]:
            st.caption(f"• {reason}")

    # ----------------------------------------------
    # WHAT THE AI SEES (live object tracks — detection debugging)
    # ----------------------------------------------

    with st.expander("👁 What the AI sees (live objects)"):

        tracks = st.session_state.get("tv_tracks") or []

        if not tracks:
            st.caption(
                "No objects detected right now. Hold a recognizable "
                "item toward the camera: phone, bottle, cup, book, "
                "banana, remote, scissors..."
            )
        else:
            for t in tracks:
                state = ""

                if t.is_held():
                    state = (
                        f" — ✋ HELD, lift "
                        f"{t.effective_lift_px():.0f}px "
                        f"({t.hold_frames} frames)"
                    )
                elif t.ever_held:
                    state = " — was held"

                st.caption(
                    f"• #{t.track_id} **{t.class_name}** — "
                    f"{t.confidence * 100:.0f}% confidence"
                    f"{state}"
                )

        if (
            object_context is not None
            and object_context.marked_location is not None
        ):
            st.caption("🎯 Marked location is ACTIVE")

    # ----------------------------------------------
    # TASK PROGRESS (multi-step)
    # ----------------------------------------------

    if task_session is not None and task_session.mode == "multi":

        st.subheader("TASK PROGRESS")

        for step in task_session.steps_view():

            if step["state"] == "COMPLETED":
                st.success(
                    f"✓ Step {step['number']} — "
                    f"{step['text']} "
                    f"({step['confidence']:.0f}%)"
                )
            elif step["state"] == "IN PROGRESS":
                st.warning(
                    f"→ Step {step['number']} — "
                    f"{step['text']} — IN PROGRESS"
                )
            else:
                st.info(
                    f"○ Step {step['number']} — "
                    f"{step['text']} — LOCKED"
                )

        st.caption(
            f"Progress: {task_session.completed_count()} / "
            f"{len(task_session.steps)} Steps Completed"
        )
        st.progress(task_session.progress())


def show_task_verification():
    """Task Verification page — uses the existing camera + pose pipeline."""

    st.subheader("✅ Task Verification")

    # Task history (app.py also initializes it — double safety)
    if "task_history" not in st.session_state:
        st.session_state.task_history = []

    # Did the camera frames stop in the previous batch?
    if st.session_state.pop("tv_camera_lost", False):
        st.error(
            "❌ Camera stopped sending frames — verification "
            "paused. Resume or start again."
        )

    # ----------------------------------------------
    # TASK TYPE
    # ----------------------------------------------

    task_type = st.radio(
        "Task Type",
        ["Single Task", "Multi-Step Task"],
        horizontal=True,
        key="tv_task_type"
    )

    task_texts = []

    if task_type == "Single Task":

        task_text = st.text_input(
            "Enter Task",
            value=st.session_state.get("tv_single_text", ""),
            placeholder="Example: Raise your right hand",
            key="tv_single_input"
        )

        st.session_state.tv_single_text = task_text

        # FIX: adding the single-task text to task_texts was necessary,
        # otherwise the Start button stayed disabled forever.
        if task_text.strip():
            task_texts.append(task_text)

            kind, key, label, detail = _parse_any(task_text)

            if kind == "pose":
                st.success(
                    f"🤖 AI understood: **{label}** (pose task)"
                )
            elif kind == "object":
                reqs = get_task_requirements(key)
                st.success(
                    f"🤖 AI understood: **{label}** "
                    f"(object task — needs: {', '.join(reqs)})"
                )
            elif kind == "unsupported":
                st.error(
                    f"❌ **{label}** is NOT supported from a single RGB "
                    f"camera: {detail.get('reason', '')} "
                    "Try a similar verifiable task instead."
                )
            else:
                st.warning(
                    "⚠️ Task not recognized. Supported: "
                    + ", ".join(ACTION_LABELS.values())
                    + " — or try: Pick up a bottle, Put bottle on "
                    "table, Hand object to another person ..."
                )

    else:

        # ------------------------------------------
        # Dynamic multi-step input
        # ------------------------------------------

        if "tv_step_count" not in st.session_state:
            st.session_state.tv_step_count = 2

        step_count = st.session_state.tv_step_count

        for i in range(step_count):
            task_texts.append(
                st.text_input(
                    f"Step {i + 1}",
                    value=st.session_state.get(f"tv_step_{i}", ""),
                    placeholder="Example: Stand up",
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

    # ----------------------------------------------
    # OBJECT TASK OPTIONS (Task Library extras —
    # shown only when the task is an object-interaction task)
    # ----------------------------------------------

    if _texts_need_objects(task_texts):
        with st.expander(
            "🧰 Task Library Options "
            "(needed by some object tasks)",
            expanded=False,
        ):
            st.caption(
                "These feed REAL verification — e.g. color matching "
                "measures the object's actual HSV color, arrange "
                "checks real left-to-right positions."
            )

            st.session_state.tv_opt_color = st.selectbox(
                "Color (for 'Put correct colored object into box')",
                ["(none)", "red", "blue", "green", "yellow"],
                index=(
                    ["(none)", "red", "blue", "green", "yellow"].index(
                        st.session_state.get("tv_opt_color", "(none)")
                    )
                ),
                key="tv_opt_color_sel",
            )

            sortable = [
                "bottle", "cup", "book", "cell phone",
                "apple", "banana", "bowl",
            ]
            st.session_state.tv_opt_order = st.multiselect(
                "Objects to arrange in order (left to right)",
                sortable,
                default=st.session_state.get("tv_opt_order", []),
                key="tv_opt_order_sel",
            )

            st.session_state.tv_opt_count = st.number_input(
                "How many objects (multiple/collect tasks)",
                min_value=2,
                max_value=8,
                value=int(st.session_state.get("tv_opt_count", 3)),
                key="tv_opt_count_sel",
            )

            # marked location editor (initial region)
            st.markdown(
                "**Marked Location** — a region on screen used by "
                "marked-location / place tasks when no box is "
                "detected."
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                mx = st.number_input(
                    "X", 0, 1920,
                    value=int(st.session_state.get("tv_mk_x", 400)),
                    key="tv_mk_x_sel",
                )
            with c2:
                my = st.number_input(
                    "Y", 0, 1080,
                    value=int(st.session_state.get("tv_mk_y", 400)),
                    key="tv_mk_y_sel",
                )
            with c3:
                mw = st.number_input(
                    "W", 10, 1920,
                    value=int(st.session_state.get("tv_mk_w", 200)),
                    key="tv_mk_w_sel",
                )
            with c4:
                mh = st.number_input(
                    "H", 10, 1080,
                    value=int(st.session_state.get("tv_mk_h", 200)),
                    key="tv_mk_h_sel",
                )
            st.session_state.tv_marked_location = [mx, my, mw, mh]

    # ----------------------------------------------
    # CAMERA SOURCE (existing pattern)
    # ----------------------------------------------

    # ----------------------------------------------
    # CAMERA SOURCE (existing pattern)
    # ----------------------------------------------

    st.subheader("📷 Camera Source")

    camera_type = st.radio(
        "Select Camera Source",
        ["Laptop Camera", "Connect with Mobile"],
        horizontal=True,
        key="tv_camera_type"
    )

    if camera_type == "Connect with Mobile":
        st.session_state.tv_mobile_ip = st.text_input(
            "Mobile IP Address",
            value=st.session_state.get("tv_mobile_ip", ""),
            placeholder="Example: 192.168.1.5",
            key="tv_mobile_ip_input"
        )
        st.session_state.tv_port = st.number_input(
            "Port",
            min_value=1,
            max_value=65535,
            value=int(st.session_state.get("tv_port", 8080)),
            key="tv_port_input"
        )

    # ----------------------------------------------
    # ACTION BUTTONS
    # ----------------------------------------------

    # CAMERA LIFECYCLE — task lifecycle se BILKUL independent.
    # The camera turns ON once, then stops only via the STOP CAMERA button
    # (never on task complete/fail/reset).
    if "tv_cam_on" not in st.session_state:
        st.session_state.tv_cam_on = False

    cam_col, _ = st.columns([1, 2])

    with cam_col:
        if not st.session_state.tv_cam_on:
            start_cam_pressed = st.button(
                "📷 START CAMERA",
                use_container_width=True,
                type="primary",
                key="tv_cam_start_btn",
            )
            stop_cam_pressed = False
        else:
            start_cam_pressed = False
            stop_cam_pressed = st.button(
                "⏹️ STOP CAMERA",
                use_container_width=True,
                key="tv_cam_stop_btn",
            )

    if start_cam_pressed:
        st.session_state.tv_cam_on = True
        # fresh monitor state (trackers reset — naya session)
        st.session_state.task_session = None
        _prepare_object_context(texts=[])
        st.session_state.tv_object_context = None
        reset_tracker()
        st.rerun()

    if stop_cam_pressed:
        # The camera stops ONLY here
        st.session_state.tv_cam_on = False
        release_task_camera()
        st.session_state.tv_running = False
        st.session_state.task_session = None
        st.rerun()

    # The legacy buttons no longer touch the camera —
    # they only reset the TASK state
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        start_pressed = st.button(
            "🎯 Start / Verify Task",
            use_container_width=True,
            type="primary",
            disabled=not any(t.strip() for t in task_texts)
        )

    with btn_col2:
        reset_pressed = st.button(
            "🔄 Reset Task",
            use_container_width=True
        )

    with btn_col3:
        clear_pressed = st.button(
            "🧹 CLEAR TASK",
            use_container_width=True
        )

    # ----------------------------------------------
    # TASK START — whether the camera is running or not, the task gets set
    # (a task can be entered even while the camera is off)
    # ----------------------------------------------

    if start_pressed:

        texts = [t.strip() for t in task_texts if t.strip()]

        if not texts:
            st.error("Please enter at least one task/step.")
        else:
            st.session_state.task_session = TaskVerificationSession(texts)
            st.session_state.tv_running = True
            st.session_state.tv_step_count = max(2, len(texts))
            _prepare_object_context(texts)
            st.rerun()

    if clear_pressed:
        # Task clear — THE CAMERA IS NOT TOUCHED
        st.session_state.task_session = None
        st.session_state.tv_running = False
        st.rerun()

    # ----------------------------------------------
    # RESET (spec: sab steps NOT COMPLETED -> step 1 IN PROGRESS)
    # ----------------------------------------------

    if reset_pressed and st.session_state.get("task_session") is not None:

        st.session_state.task_session.reset()
        st.session_state.tv_running = True
        _prepare_object_context(texts=st.session_state.get(
            "tv_steps_text", []
        ))
        st.success("🔄 Task reset — Step 1 is IN PROGRESS again.")
        st.rerun()

    # ----------------------------------------------
    # SESSION INIT (if never started so far)
    # ----------------------------------------------

    if "task_session" not in st.session_state:
        st.session_state.task_session = None

    if "tv_running" not in st.session_state:
        st.session_state.tv_running = False

    task_session = st.session_state.get("task_session")

    # ----------------------------------------------
    # CAMERA STATE vs TASK STATE — COMPLETELY SEPARATE.
    # tv_cam_on stops only via STOP CAMERA.
    # task_session complete/fail/clear never touches the camera.
    # ----------------------------------------------

    if not st.session_state.tv_cam_on:

        st.warning(
            "⚪ Camera stopped — press **📷 START CAMERA** above to "
            "begin continuous monitoring."
        )
        return

    # ----------------------------------------------
    # LIVE MONITORING LAYOUT (camera + status side by side)
    # ----------------------------------------------

    main_col, status_col = st.columns([2, 1])

    with status_col:
        status_placeholder = st.empty()

    with main_col:
        frame_placeholder = st.empty()

        m1, m2, m3 = st.columns(3)

        with m1:
            activity_metric = st.empty()

        with m2:
            streak_metric = st.empty()

        with m3:
            progress_metric = st.empty()

        detection_placeholder = st.empty()

    status_placeholder.success("Camera: 🟢 RUNNING")

    # ----------------------------------------------
    # TASK COMPLETED — show the banner (~2.5s), then the task clears
    # automatically: the operator can give the next task, THE CAMERA
    # KEEPS RUNNING.
    # ----------------------------------------------

    if task_session is not None and task_session.all_completed:

        first_seen = st.session_state.get("tv_completed_at")

        if first_seen is None:
            st.session_state.tv_completed_at = time.time()
            first_seen = st.session_state.tv_completed_at

        if time.time() - first_seen < 2.5:

            st.success("🎉 ALL TASKS COMPLETED — camera is still "
                       "running. Next task coming up...")

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric("Total Steps", len(task_session.steps))

            with summary_col2:
                avg_confidence = (
                    sum(s["confidence"] for s in task_session.steps)
                    / len(task_session.steps)
                )
                st.metric(
                    "Avg Confidence",
                    f"{avg_confidence:.0f}%"
                )

            with summary_col3:
                st.metric("Mode", task_session.mode.title())

            for step in task_session.steps_view():
                st.success(
                    f"✓ Step {step['number']} — {step['text']} "
                    f"({step['confidence']:.0f}%)"
                )

            detection_placeholder.success(
                "✅ TASK COMPLETED — camera continues. "
                "Enter the next task anytime."
            )

            time.sleep(1.0)   # banner readable, phir auto-reset
            st.rerun()

        # 2.5s passed -> task reset (already logged in history)
        st.session_state.task_session = None
        st.session_state.tv_running = False
        st.session_state.tv_completed_at = None
        st.session_state.tv_object_context = None
        task_session = None
        st.rerun()

    st.session_state.tv_completed_at = None

    # ----------------------------------------------
    # CAMERA — stays in session_state so the device does not have to
    # be reopened between Stop/Resume batches.
    # ----------------------------------------------

    cap = st.session_state.get("tv_cap")

    if cap is None or not cap.isOpened():

        cap = _task_camera(camera_type)

        if cap is None:
            status_placeholder.error("❌ Camera could not be opened.")
            return

        st.session_state.tv_cap = cap

        # Give the person tracker a fresh start (so the old run's
        # movement history does not affect the new analysis)
        reset_tracker()

        # Object tracker/context fresh too (so the old run's track
        # history does not corrupt displacement measurement)
        obj_ctx = st.session_state.get("tv_object_context")

        if obj_ctx is not None:
            obj_ctx.tracker.reset()
            st.session_state.tv_tracks = []

    # ----------------------------------------------
    # MONITOR-ONLY MODE (no task active): the camera still shows
    # pose + activity + live objects — just no task
    # verification runs.
    # ----------------------------------------------

    if task_session is None and st.session_state.get(
        "tv_object_context"
    ) is None:
        st.session_state.tv_object_context = ObjectTaskContext(
            ObjectTracker()
        )

    # ~1 second ka processing batch (batch design ka reason
    # function docstring me hai)
    _task_verification_batch(
        cap, task_session, main_col, status_col,
        frame_placeholder, activity_metric, streak_metric,
        progress_metric, detection_placeholder,
        st.session_state.get("tv_object_context"),
    )


def release_task_camera():
    """Task verification camera release (idempotent)."""
    cap = st.session_state.pop("tv_cap", None)
    if cap is not None:
        try:
            cap.release()
        except Exception as exc:
            print("Camera release error:", exc)


def _task_verification_batch(
    cap, task_session, main_col, status_col,
    frame_placeholder, activity_metric, streak_metric,
    progress_metric, detection_placeholder,
    object_context=None,
):
    """
    Processes a ~1 second frame batch, then reruns.

    Reason for the batch design: one long Python loop blocks the
    Streamlit script — Stop/Reset/New Task clicks then sit pending
    in the queue and never execute. With small batches every click
    takes effect within 1-2 seconds. The camera stays in
    session_state, so the device is not reopened every batch
    (no flicker).
    """

    # ----------------------------------------------
    # LIVE BATCH — camera frame → YOLO Pose → (task logic)
    # The camera runs ON EVERY FRAME — with or without a task.
    # ----------------------------------------------

    for _ in range(TV_BATCH_FRAMES):

        if not st.session_state.tv_cam_on:
            break

        ret, frame = cap.read()

        if not ret:
            # Frame loss -> camera release + paused state
            release_task_camera()
            st.session_state.tv_running = False
            st.session_state.tv_camera_lost = True
            st.rerun()

        # ------------------------------------------------
        # EXISTING pose pipeline reuse (keypoints + activity)
        # ------------------------------------------------

        annotated, persons = analyze_frame(frame)

        # ------------------------------------------------
        # OBJECT TASKS: YOLO objects -> tracker -> context
        # (extra YOLO-object inference only in object-interaction
        # runs me hota hai — pose-only runs unchanged performance)
        # ------------------------------------------------

        current_action = None

        if task_session is not None:
            current_step = task_session.current_step()
            if current_step is not None:
                current_action = current_step.get("action")

        if object_context is not None:
            # conf 0.10: weak-but-real detections (a dining table at
            # a webcam angle scores 0.07-0.11) must reach the tracker —
            # task-level gates (MIN_OBJECT_CONF/MIN_TABLE_CONF +
            # avg_confidence) filter further downstream. A detector-level
            # 0.30 removed the table from the pipeline entirely.
            _, objects = detect_objects(frame, conf=0.10)

            object_context.persons = persons
            object_context.objects = objects
            object_context.last_frame = frame

            marked = st.session_state.get("tv_marked_location")
            object_context.marked_location = (
                list(marked) if marked else None
            )

            tracks = object_context.tracker.update(objects, persons)
            st.session_state.tv_tracks = tracks

            region = None

            if object_context.marked_location is not None:
                m = object_context.marked_location
                region = [
                    m[0], m[1], m[0] + m[2], m[1] + m[3]
                ]

            # Move Bottle (Place 1 -> Place 2) demo: left/right
            # third overlays (for stage guidance)
            if current_action == "move_bottle_places":
                fw = float(annotated.shape[1])
                fh = float(annotated.shape[0])
                region = [
                    [0.0, 0.0, fw / 3.0, fh],
                    [fw * 2.0 / 3.0, 0.0, fw, fh],
                ]

            draw_tracks(annotated, tracks, region_box=region)

        # ------------------------------------------------
        # TASK LOGIC — only while a task is active. No task = general
        # monitoring (pose + activity + objects), no verification.
        # ------------------------------------------------

        if task_session is not None:
            result = verify_task_step(
                task_session, persons, object_context=object_context
            )
        else:
            result = {
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

        # ------------------------------------------------
        # HISTORY EVENTS (task history — activity history se alag)
        # ------------------------------------------------

        for event in result["events"]:
            st.session_state.task_history.insert(0, event)

        st.session_state.task_history = (
            st.session_state.task_history[:200]
        )

        # ------------------------------------------------
        # FRAME + TASK LABEL DRAW
        # ------------------------------------------------

        label_text = None

        if result["all_completed"]:
            label_text = "ALL TASKS COMPLETED"
        elif result["current_step"]:
            label_text = (
                f"TASK: {result['current_step'].upper()} "
                f"| {result['streak']}/"
                f"{result['required_frames']}"
            )

        if label_text:

            cv2.putText(
                annotated,
                label_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 120),
                2,
                cv2.LINE_AA
            )

        # ------------------------------------------------
        # UI UPDATE (camera ke beside status)
        # ------------------------------------------------

        with main_col:

            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True
            )

            if persons:
                main_person = persons[0]
                activity_metric.metric(
                    "Activity",
                    main_person["activity"]
                )
            else:
                activity_metric.metric("Activity", "No Person")

            streak_metric.metric(
                "Confirmation",
                f"{result['streak']}/{result['required_frames']}"
            )

            progress_metric.metric(
                "Progress",
                f"{int(result['progress'] * 100)}%"
            )

            if result["step_completed"]:
                detection_placeholder.success(
                    f"✓ Step {result['completed_step_number']} "
                    f"COMPLETED — next step unlocked!"
                )
            elif result["detected"]:
                detection_placeholder.warning(
                    "🟢 Action detected — keep it steady "
                    "for temporal confirmation!"
                )
            else:
                # Show the REAL reason ("No object detected..." etc.) —
                # instead of a generic message (detection debugging)
                reason = (
                    result["reasons"][0] if result["reasons"]
                    else "perform the current step."
                )
                detection_placeholder.info(
                    f"🤖 {reason}"
                )

        with status_col:

            if task_session is not None:
                _render_task_status(task_session, result, object_context)
            else:
                # Monitor-only: live status without task verification
                st.markdown("**CURRENT TASK**\n\n*No task selected*")
                st.info("ℹ️ Enter a task above anytime — camera keeps "
                        "running.")

                with st.expander("👁 What the AI sees (live objects)"):
                    tracks = st.session_state.get("tv_tracks") or []

                    if not tracks:
                        st.caption(
                            "No objects detected right now. Hold a "
                            "recognizable item toward the camera: "
                            "phone, bottle, cup, book, banana, remote..."
                        )
                    else:
                        for t in tracks:
                            state = ""

                            if t.is_held():
                                state = (
                                    f" — ✋ HELD, lift "
                                    f"{t.effective_lift_px():.0f}px "
                                    f"({t.hold_frames} frames)"
                                )
                            elif t.ever_held:
                                state = " — was held"

                            st.caption(
                                f"• #{t.track_id} **{t.class_name}** — "
                                f"{t.confidence * 100:.0f}% "
                                f"confidence{state}"
                            )

        # ------------------------------------------------
        # ALL COMPLETED
        # ------------------------------------------------

        # NOTE: the camera does NOT stop on completion — the task clear
        # happens in show_task_verification above; the batch keeps running.

    # ----------------------------------------------
    # BATCH DONE -> fresh render (next batch). Only the STOP CAMERA
    # button (above) is responsible for the camera.
    # ----------------------------------------------

    st.rerun()
