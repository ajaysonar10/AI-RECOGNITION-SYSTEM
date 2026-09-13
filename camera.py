import cv2
import streamlit as st
import time
from datetime import datetime

from pose_detection import analyze_frame
from object_detection import detect_objects


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
