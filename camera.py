import threading
import sys
import cv2
import streamlit as st
import time
from datetime import datetime
import tempfile
import os
import av
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)

from pose_detection import analyze_frame, reset_tracker
from object_detection import detect_objects
from object_tracking import ObjectTracker, draw_tracks
from object_tasks import ObjectTaskContext, get_task_requirements
from step_validator import StepValidator
from task_detection import (
    TaskVerificationSession,
    parse_task_text,
    ACTION_LABELS,
)

# Metered Dedicated TURN/STUN Configuration
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.relay.metered.ca:80"]},
            {
                "urls": [
                    "turn:standard.relay.metered.ca:80",
                    "turn:standard.relay.metered.ca:443",
                    "turn:standard.relay.metered.ca:443?transport=tcp",
                ],
                "username": "1b4e30c55c4f585a09221da6",
                "credential": "rkYnm4VLtnSbdoqx",
            },
        ]
    }
)

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


# ============================================================
# 🎥 CRASH-PROOF WEBRTC AI PROCESSOR
# ============================================================
class LiveCameraProcessor(VideoProcessorBase):
    def __init__(self):
        self.step_validator = StepValidator()
        self.latest_activity = "Detecting..."
        self.latest_confidence = 0.0
        self.object_count = 0
        self.step_status = "WAITING"
        self.step_message = "Waiting for detection..."
        self.persons = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")

            # Optimization for cloud processing stability
            h, w = img.shape[:2]
            if w > 480:
                scale = 480.0 / w
                img = cv2.resize(img, (480, int(h * scale)))

            # 1. Pose Analysis
            detected_frame, persons = analyze_frame(img)
            self.persons = persons or []

            # 2. Object Detection
            _, objects = detect_objects(detected_frame)
            self.object_count = len(objects)
            detected_frame = draw_object_boxes(detected_frame, objects)

            # 3. Sequential Step Validation
            if self.persons:
                main_p = self.persons[0]
                self.latest_activity = main_p.get("activity", "Unknown")
                self.latest_confidence = float(main_p.get("confidence", 0.0))

                res = self.step_validator.check_activity(self.latest_activity)
                self.step_status = res.get("status", "WAITING")
                self.step_message = res.get("message", "")
            else:
                self.latest_activity = "No Person"
                self.latest_confidence = 0.0
                self.step_status = "WAITING"
                self.step_message = "No person detected in frame."

            return av.VideoFrame.from_ndarray(detected_frame, format="bgr24")

        except Exception:
            return frame


# ============================================================
# 📷 1. MAIN CAMERA VIEW
# ============================================================
def show_camera():
    st.subheader("📷 Camera & Video Source")

    source_type = st.radio(
        "Select Source",
        ["Live Device Camera (WebRTC)", "Upload Recorded Video (.mp4)"],
        horizontal=True,
        key="main_source_selection",
    )

    if source_type == "Live Device Camera (WebRTC)":
        st.caption(
            "Click **START** below. When prompted by your browser, click **Allow** to enable camera access."
        )

        ctx = webrtc_streamer(
            key="webrtc-stream-main",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=LiveCameraProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 480},
                    "height": {"ideal": 360},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            async_processing=True,
        )

        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            act_metric = st.empty()
        with c2:
            conf_metric = st.empty()
        with c3:
            sys_metric = st.empty()
        with c4:
            obj_metric = st.empty()

        st.subheader("🎯 Step Validation Status")
        s1, s2, s3 = st.columns(3)
        with s1:
            step_name_box = st.empty()
        with s2:
            step_status_box = st.empty()
        with s3:
            step_msg_box = st.empty()

        st.subheader("👤 Detection Reasoning & Person Analysis")
        analysis_box = st.empty()

        if ctx.video_processor:
            proc = ctx.video_processor
            act_metric.metric("Current Activity", proc.latest_activity)
            conf_metric.metric("Confidence", f"{proc.latest_confidence:.1f}%")
            sys_metric.metric(
                "System Status", "ONLINE 🟢" if ctx.state.playing else "PAUSED"
            )
            obj_metric.metric("Detected Objects", proc.object_count)

            curr_step = proc.step_validator.get_current_step()
            step_name_box.metric("Current Step", curr_step or "COMPLETED")

            if proc.step_status == "CORRECT":
                step_status_box.success("✅ CORRECT")
            elif proc.step_status == "WRONG":
                step_status_box.error("❌ WRONG")
            else:
                step_status_box.info("⏳ WAITING")

            step_msg_box.write(proc.step_message)

            if proc.persons:
                p = proc.persons[0]
                reasons = "\n".join(f"- {r}" for r in p.get("justification", []))
                pos_desc = p.get("position", {}).get("description", "Unknown")
                analysis_box.markdown(
                    f"**Activity:** `{p.get('activity')}` | **Position:** {pos_desc}\n\n{reasons}"
                )
        else:
            sys_metric.metric("System Status", "STANDBY ⚪")
            act_metric.metric("Current Activity", "—")
            conf_metric.metric("Confidence", "—")
            obj_metric.metric("Detected Objects", "—")

    else:
        uploaded_video = st.file_uploader(
            "Upload action video (.mp4, .mov, .avi)",
            type=["mp4", "mov", "avi", "mkv"],
            key="fallback_uploader",
        )

        col_start, col_stop = st.columns(2)
        with col_start:
            start_proc = st.button("🟢 Start Processing", use_container_width=True)
        with col_stop:
            stop_proc = st.button("🔴 Stop Processing", use_container_width=True)

        if "video_file_running" not in st.session_state:
            st.session_state.video_file_running = False

        if stop_proc:
            st.session_state.video_file_running = False
            st.rerun()

        if start_proc:
            if uploaded_video is None:
                st.error("Please upload a video file first.")
                return
            st.session_state.video_file_running = True

        if not st.session_state.video_file_running:
            st.info("Upload a video and press 🟢 Start Processing.")
            return

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.getvalue())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        frame_box = st.empty()

        while st.session_state.video_file_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.info("Video processing completed.")
                break

            h, w = frame.shape[:2]
            if w > 480:
                scale = 480.0 / w
                frame = cv2.resize(frame, (480, int(h * scale)))

            detected_frame, persons = analyze_frame(frame)
            _, objects = detect_objects(detected_frame)
            detected_frame = draw_object_boxes(detected_frame, objects)

            frame_box.image(
                cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            time.sleep(0.03)

        cap.release()
        st.session_state.video_file_running = False


# ============================================================
# 📝 2. TASK EVALUATION
# ============================================================
def show_task_evaluation():
    st.subheader("📝 Live Task Evaluation")
    st.info("Task evaluation is running on the continuous monitoring pipeline.")
    show_camera()


# ============================================================
# ✅ 3. TASK VERIFICATION
# ============================================================
def show_task_verification():
    st.subheader("✅ Live Task Verification")
    show_camera()
