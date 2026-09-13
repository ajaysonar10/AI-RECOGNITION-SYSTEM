from pose_detection import analyze_frame
from object_detection import detect_objects
from step_validator import StepValidator

import threading
import cv2
import streamlit as st
import time
from datetime import datetime
import pyttsx3

from pose_detection import analyze_frame
from object_detection import detect_objects


#functions
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
# Object boxes ko pose frame par draw karne ka helper
# (alag color, taaki pose/person boxes se confuse na ho)
# ------------------------------------------------------------

OBJECT_BOX_COLOR = (0, 200, 255)   # BGR: orange


def draw_object_boxes(frame, objects):
    """
    Detected objects ke boxes + labels frame par draw karta hai.
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
    objects_placeholder.info("📦 Object detection active — boxes frame par dikhenge.")

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
                "Person position aur justification yahan dikhega "
                "jab koi person frame mein aayega."
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
                "📦 Koi object detect nahi hua is frame mein."
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
# (User task deta hai -> live camera se AI verify karta hai)
# ============================================================

from task_eval import LiveTaskEvaluator, build_report


def show_task_evaluation():

    st.subheader("📝 Task")

    task = st.text_area(
        "What should the person do? (live camera par perform karo)",
        placeholder="Example: Pick up the bottle and place it on the table",
        key="task_eval_input"
    )

    # Live parse preview — user ko dikhta hai AI ne kya samjha
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
        # Camera open karo
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
            "🤖 Live evaluation running... Task poora karo. "
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
            # ==================================================
            # 🔊 10-SECOND WRONG ACTIVITY VOICE ALERT
            # ==================================================

            voice_message = evaluator.get_voice_alert()

            if voice_message:

                voice_engine.say(voice_message)
                voice_engine.runAndWait()
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
        # Loop khatam -> verdict dikhao
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
    """Live checklist markdown banata hai."""

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
    """Final verdict + report render karta hai."""

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
