# 🛰️ AI-RECOGNITION-SYSTEM

AI powered **Human Activity Recognition System** for real-time monitoring — built for on-board space experiment validation (BAS•AI — Smart India Hackathon 2026).

---

## 📖 Overview

BAS•AI is an intelligent on-board AI system that **detects persons** from a live camera feed, tracks their **position**, and classifies their **activity** (Standing / Sitting / Walking) **with justification** — all in real time.

---

## ✨ Features

- 🎯 **Real-time Pose Detection** — 17 body keypoints from YOLO11n-pose (nose, shoulders, hips, knees, ankles...)
- 👤 **Multi-Person Tracking** — every person gets a stable **Track ID** (frame-to-frame)
- 📍 **Position Analysis** — bounding box, body center (pixels), and frame region (e.g. `middle-center`)
- 🧠 **Activity Recognition with Justification** — transparent reasons with every decision:
  - 🧍 **Standing** — legs straight (knee angle ≥ 155°)
  - 🪑 **Sitting** — knees bent (knee angle < 125°)
  - 🚶 **Walking** — jitter-proof detection: net displacement + direction consistency + hysteresis (4-frame confirm)
  - ❓ **Unknown** — legs not visible, so no false claim
- 🖥️ **Streamlit Dashboard** — Live Monitor, Activity History, Analytics, System Status
- 📊 **Live Metrics** — activity, confidence %, persons count, activity stream (1-sec logging)
- 📱 **Dual Camera Support** — laptop webcam or mobile camera (IP Webcam / DroidCam over Wi-Fi)
- ✅ **Task Verification** — pose tasks (raise hand, walk...) + 22 object-interaction tasks (pick up bottle, place in box, move Place 1 → Place 2...) verified with real YOLO detection, object tracking and multi-frame temporal confirmation

---

## 🗂️ Project Structure

```
AI-Recognition-System/
├── app.py                    # Streamlit dashboard (main entry)
├── camera.py                 # Live camera feed + UI integration
├── pose_detection.py         # ⭐ Core: pose + position + activity + justification
├── activity_detection.py     # Legacy activity logic (merged into pose_detection.py)
├── object_detection.py       # Object detection module
├── object_tracking.py        # Object tracking: hold, lift, displacement
├── object_tasks.py           # Object-interaction task library (22 tasks)
├── task_detection.py         # Task session: temporal + sequential verification
├── live_pose_detection.py    # Standalone webcam demo (without UI)
└── README.md
```

### 🔑 Core API (`pose_detection.py`)

```python
from pose_detection import analyze_frame

annotated_frame, persons = analyze_frame(frame)

for p in persons:
    print(p["track_id"])           # stable person ID
    print(p["position"])           # bbox, center, region description
    print(p["activity"])           # Standing / Sitting / Walking / Unknown
    print(p["confidence"])         # 0-100
    print(p["justification"])      # list of reasons (knee angles, movement...)
    print(p["keypoints"])          # {name: (x, y)} — 17 COCO keypoints
```

The legacy interface `detect_pose(frame)` is also available (backward compatible).

---

## 🚀 Setup & Run

### Requirements

```bash
pip install streamlit ultralytics opencv-python
```

### Run

```bash
cd AI-Recognition-System
streamlit run app.py
```

The browser opens `http://localhost:8501` → sidebar → **📡 Live Monitor** → **🟢 Start Camera**

> ⚠️ On the first run, the `yolo11n-pose.pt` model (~6 MB) auto-downloads.

### Standalone demo (without UI)

```bash
python pose_detection.py        # webcam + console output (position + justification)
python live_pose_detection.py   # simple webcam skeleton demo
```

### 📱 Connecting a mobile camera

1. Install the **IP Webcam** (or DroidCam) app on your phone
2. Phone and laptop must be on the **same Wi-Fi**
3. Start the video stream in the app — you get an IP (e.g. `192.168.1.5:8080`)
4. Live Monitor → **Connect with Mobile** → enter the IP + port

---

## 🖥️ Offline Desktop Application

No browser and no Python installation needed — BAS•AI runs as a native Windows desktop app.

### Option A — Standalone .exe (for any Windows PC)

Build it once (on a PC with Python):

```bash
pip install pywebview pyinstaller
python build_exe.py
```

Then share the whole `dist/BAS-AI/` folder (zip it). Double-click **BAS-AI.exe** — the dashboard opens in a native app window with both YOLO models bundled.

### Option B — Python launcher (developer mode)

```bash
pip install pywebview
python desktop_app.py
```

Launches the Streamlit server on a free local port and opens it in a native desktop window (falls back to the default browser if WebView2 is unavailable).

> 💡 Everything works fully **offline**: AI inference (YOLO), voice alerts (pyttsx3), and the dashboard (localhost server).

---

## 🧠 Activity Detection Logic

| Activity | Signal | Threshold |
|----------|--------|-----------|
| 🚶 Walking | Body center consistently moving | movement > 4 px/frame **+** net displacement > 25 px **+** direction consistency ≥ 0.55 **+** 4-frame streak |
| 🪑 Sitting | Knees bent | avg knee angle < 125° |
| 🧍 Standing | Legs straight | avg knee angle ≥ 155° |
| ❓ Unknown | Legs not clearly visible | — |

**Why is Walking jitter-proof?** While standing, the body/camera has micro-jitter (2–5 px random movement). So per-frame movement alone is not enough — **net displacement** (straight start→end distance) and **direction consistency** (net ÷ total path) are also checked. In jitter, net displacement is ~0; while walking it is large. Verified with simulation tests ✓

---

## 📊 Dashboard Pages

| Page | What it shows |
|------|---------|
| 🏠 Dashboard | Mission overview + recent activity |
| 📡 Live Monitor | Live video + per-person position & justification + task verification |
| 🕒 Activity History | All detections (date/time/activity/confidence/status) |
| 📊 Analytics | Activity distribution chart + average confidence |

---

## 🛰️ Mission

**BAS Experiment** — On-board space intelligence for crew activity monitoring during space-based experiments.

**Event:** Smart India Hackathon 2026 🏆

---

<div align="center">

**BAS•AI** | ON-BOARD HUMAN ACTIVITY RECOGNITION

</div>
