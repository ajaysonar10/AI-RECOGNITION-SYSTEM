import threading
import sys
import cv2
import streamlit as st
import time
from datetime import datetime
import pyttsx3
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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

# STUN Server (Cloud par smoothly video stream karne ke liye)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
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
            frame, label, (p1[0], label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            OBJECT_BOX_COLOR, 1, cv2.LINE_AA,
        )
    return frame


# ============================================================
# 🎥 LIVE WEBRTC AI PROCESSOR
# (Har ek frame direct browser se aayega aur process hoga)
# ============================================================
class LiveCameraProcessor(VideoProcessorBase):
    def __init__(self):
        self.step_validator = StepValidator()
        self.latest_activity = "Detecting..."
        self.latest_confidence = 0.0
        self.object_count = 0
        self.step_status = "WAITING"
        self.step_message = "Waiting for person..."
        self.persons = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # 1. Pose Analysis
        detected_frame, persons = analyze_frame(img)
        self.persons = persons

        # 2. Object Detection
        _, objects = detect_objects(detected_frame)
        self.object_count = len(objects)
        detected_frame = draw_object_boxes(detected_frame, objects)

        # 3. Step & Activity Logic
        if persons:
            main_p = persons[0]
            self.latest_activity = main_p["activity"]
            self.latest_confidence = main_p["confidence"]

            res = self.step_validator.check_activity(self.latest_activity)
            self.step_status = res["status"]
            self.step_message = res["message"]
        else:
            self.latest_activity = "No Person"
            self.latest_confidence = 0.0
            self.step_status = "WAITING"
            self.step_message = "Person not detected."

        return av.VideoFrame.from_ndarray(detected_frame, format="bgr24")


# ============================================================
# 📷 1. MAIN LIVE CAMERA VIEW
# ============================================================
def show_camera():
    st.subheader("📷 Live Hardware Camera (Laptop / Mobile)")
    st.caption("Browser camera permission maangega — use **Allow** karein.")

    ctx = webrtc_streamer(
        key="main-live-cam",
        video_processor_factory=LiveCameraProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1: act_box = st.empty()
    with c2: conf_box = st.empty()
    with c3: sys_box = st.empty()
    with c4: obj_box = st.empty()

    st.subheader("🎯 Sequential Step Evaluation")
    s1, s2, s3 = st.columns(3)
    with s1: step_name_box = st.empty()
    with s2: step_status_box = st.empty()
    with s3: step_msg_box = st.empty()

    st.subheader("👤 Person Justification")
    analysis_box = st.empty()

    if ctx.video_processor:
        proc = ctx.video_processor
        act_box.metric("Activity", proc.latest_activity)
        conf_box.metric("Confidence", f"{proc.latest_confidence:.1f}%")
        sys_box.metric("System", "LIVE 🟢" if ctx.state.playing else "PAUSED")
        obj_box.metric("Objects", proc.object_count)

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
            reasons = "\n".join(f"• {r}" for r in p["justification"])
            analysis_box.markdown(
                f"**Activity:** `{p['activity']}` | **Position:** {p['position']['description']}\n\n{reasons}"
            )
    else:
        sys_box.metric("System", "OFFLINE ⚪")
        act_box.metric("Activity", "—")
        conf_box.metric("Confidence", "—")
        obj_box.metric("Objects", "—")


# ============================================================
# 📝 2. TASK EVALUATION
# ============================================================
def show_task_evaluation():
    st.subheader("📝 Live Task Evaluation")
    st.info("Task evaluation is active on the live stream. Perform tasks directly in front of the camera.")
    show_camera()


# ============================================================
# ✅ 3. TASK VERIFICATION
# ============================================================
def show_task_verification():
    st.subheader("✅ Live Task Verification")
    st.caption("Live Camera se task verify karne ke liye niche **START** dabayein:")
    show_camera()
