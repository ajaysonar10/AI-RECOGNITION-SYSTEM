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
    verify_task_step,
    ACTION_LABELS,
)

# Dedicated Metered STUN & TURN Configuration
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


def _parse_any(text):
    action = parse_task_text(text)
    if action is not None:
        return "pose", action, ACTION_LABELS.get(action, str(action)), {}

    try:
        from object_tasks import parse_object_task, UNSUPPORTED_TASKS, TASK_LABELS
        key, detail = parse_object_task(text)
    except ImportError:
        return None, None, None, {}

    if key is None:
        return None, None, None, {}

    if detail.get("unsupported"):
        return (
            "unsupported",
            key,
            UNSUPPORTED_TASKS.get(key, {}).get("label", key),
            UNSUPPORTED_TASKS.get(key, {}),
        )

    return "object", key, TASK_LABELS.get(key, key), detail


def _prepare_object_context(texts):
    tracker = ObjectTracker()
    context = ObjectTaskContext(tracker)
    st.session_state.tv_object_context = context


# ============================================================
# 🎥 WEBRTC PROCESSOR FOR REAL-TIME TASK VERIFICATION
# ============================================================
class LiveCameraProcessor(VideoProcessorBase):
    def __init__(self):
        self.step_validator = StepValidator()
        self.task_session = None
        self.object_context = None

        self.latest_activity = "Detecting..."
        self.latest_confidence = 0.0
        self.object_count = 0
        self.step_status = "WAITING"
        self.step_message = "Waiting for detection..."
        self.persons = []

        self.task_result = {
            "all_completed": False,
            "step_completed": False,
            "completed_step_number": None,
            "current_step": None,
            "state": "WAITING",
            "detected": False,
            "confidence": 0.0,
            "streak": 0,
            "required_frames": 10,
            "progress": 0.0,
            "reasons": [],
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")

            # Cloud processing optimization
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

            # 3. Pose Evaluation
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

            # 4. Live Verification Pipeline
            if self.task_session is not None:
                try:
                    res = verify_task_step(
                        self.task_session,
                        self.persons,
                        objects,
                        self.object_context,
                    )
                    if res:
                        self.task_result = res
                except TypeError:
                    try:
                        res = verify_task_step(self.task_session, self.persons)
                        if res:
                            self.task_result = res
                    except Exception:
                        pass
                except Exception:
                    pass

            return av.VideoFrame.from_ndarray(detected_frame, format="bgr24")

        except Exception:
            return frame


# ============================================================
# 📡 EXACT UI IMPLEMENTATION (LIVE MONITOR + TASK VERIFICATION)
# ============================================================
def show_task_verification():
    st.markdown("## 📡 Live Monitor")
    st.caption("Real-time monitoring + live task evaluation")

    st.markdown("### ✅ Task Verification")

    # 1. Task Type Selection
    task_type = st.radio(
        "Task Type",
        ["Single Task", "Multi-Step Task"],
        horizontal=True,
        key="tv_mode_select",
    )

    task_texts = []
    if task_type == "Single Task":
        task_text = st.text_input(
            "Enter Task",
            value=st.session_state.get("tv_single_text", ""),
            placeholder="Example: Raise your right hand",
            key="tv_single_input",
        )
        st.session_state.tv_single_text = task_text
        if task_text.strip():
            task_texts.append(task_text.strip())
    else:
        if "tv_multi_step_count" not in st.session_state:
            st.session_state.tv_multi_step_count = 2

        count = st.session_state.tv_multi_step_count
        for i in range(count):
            val = st.text_input(
                f"Step {i + 1}",
                value="Stand up" if i == 0 else "Raise your right hand",
                key=f"tv_step_field_{i}",
            )
            if val.strip():
                task_texts.append(val.strip())

        b_add, b_rem = st.columns(2)
        with b_add:
            if st.button("＋ Add Step", use_container_width=True):
                st.session_state.tv_multi_step_count = min(6, count + 1)
                st.rerun()
        with b_rem:
            if st.button("－ Remove Step", use_container_width=True):
                st.session_state.tv_multi_step_count = max(2, count - 1)
                st.rerun()

    # 2. Camera Source Selection
    st.markdown("### 📷 Camera Source")
    camera_source = st.radio(
        "Select Camera Source",
        ["Laptop Camera", "Connect with Mobile"],
        horizontal=True,
        key="tv_camera_source_radio",
    )

    if camera_source == "Connect with Mobile":
        st.session_state.tv_mobile_ip = st.text_input(
            "Mobile IP Address",
            value=st.session_state.get("tv_mobile_ip", ""),
            placeholder="e.g. 192.168.1.5",
        )

    # State management for camera toggle
    if "tv_cam_started" not in st.session_state:
        st.session_state.tv_cam_started = False

    # 3. Start Camera Toggle Button
    if not st.session_state.tv_cam_started:
        if st.button("📷 START CAMERA", use_container_width=True, type="primary"):
            st.session_state.tv_cam_started = True
            st.rerun()
    else:
        if st.button("⏹️ STOP CAMERA", use_container_width=True):
            st.session_state.tv_cam_started = False
            st.rerun()

    # 4. Action Buttons Row
    col_start, col_reset, col_clear = st.columns(3)
    with col_start:
        start_task = st.button(
            "🎯 Start / Verify Task",
            use_container_width=True,
            disabled=not task_texts,
        )
    with col_reset:
        reset_task = st.button("🔄 Reset Task", use_container_width=True)
    with col_clear:
        clear_task = st.button("🧹 CLEAR TASK", use_container_width=True)

    if start_task:
        st.session_state.task_session = TaskVerificationSession(task_texts)
        _prepare_object_context(task_texts)
        reset_tracker()
        st.success("Task verification loaded! Perform the action in front of the camera.")

    if reset_task:
        if st.session_state.get("task_session"):
            st.session_state.task_session.reset()
        reset_tracker()
        st.info("Task reset.")

    if clear_task:
        st.session_state.task_session = None
        st.session_state.tv_object_context = None
        st.session_state.tv_single_text = ""
        reset_tracker()
        st.rerun()

    # 5. Camera Active State / Stopped Banner
    if not st.session_state.tv_cam_started:
        st.info("⚪ Camera stopped — press 📷 START CAMERA above to begin continuous monitoring.")
    else:
        st.divider()
        st.caption("Live video stream connected. Allow browser camera permissions if prompted.")

        ctx = webrtc_streamer(
            key="webrtc-exact-monitor",
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

        # Real-time Metrics and AI Justification
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            act_box = st.empty()
        with m2:
            conf_box = st.empty()
        with m3:
            streak_box = st.empty()
        with m4:
            sys_box = st.empty()

        st.markdown("#### 🎯 Verification Progress")
        task_status_container = st.empty()
        progress_bar = st.empty()
        reasoning_container = st.empty()

        if ctx.video_processor:
            proc = ctx.video_processor
            proc.task_session = st.session_state.get("task_session")
            proc.object_context = st.session_state.get("tv_object_context")

            act_box.metric("Activity", proc.latest_activity)
            conf_box.metric("Confidence", f"{proc.latest_confidence:.1f}%")
            sys_box.metric("System", "ONLINE 🟢" if ctx.state.playing else "PAUSED")

            res = proc.task_result
            req_frames = max(1, res.get("required_frames", 10))
            cur_streak = res.get("streak", 0)
            streak_box.metric("Confirmation", f"{cur_streak}/{req_frames}")

            progress_val = float(res.get("progress", 0.0))
            progress_bar.progress(max(0.0, min(1.0, progress_val)))

            if res.get("all_completed"):
                task_status_container.success("🎉 ALL TASKS COMPLETED!")
            elif res.get("step_completed"):
                task_status_container.success(
                    f"✓ Step {res.get('completed_step_number')} COMPLETED"
                )
            elif res.get("current_step"):
                if res.get("detected"):
                    task_status_container.warning(
                        f"● IN PROGRESS — action detected! ({cur_streak}/{req_frames} frames)"
                    )
                else:
                    task_status_container.info(
                        f"● IN PROGRESS — perform the action: **{res.get('current_step')}**"
                    )
            else:
                if st.session_state.get("task_session"):
                    task_status_container.info("● WAITING FOR PERSON")
                else:
                    task_status_container.info(
                        "Enter task above and click 🎯 Start / Verify Task."
                    )

            reasons = res.get("reasons", [])
            if reasons:
                with reasoning_container.expander("🧠 Why this status? (AI reasoning)", expanded=True):
                    for r in reasons[:4]:
                        st.write(f"• {r}")
        else:
            sys_box.metric("System", "CONNECTING 🟡")
            act_box.metric("Activity", "—")
            conf_box.metric("Confidence", "—")
            streak_box.metric("Confirmation", "—")


def show_camera():
    show_task_verification()


def show_task_evaluation():
    show_task_verification()
