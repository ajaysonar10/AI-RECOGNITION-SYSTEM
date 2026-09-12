# 🛰️ AI-RECOGNITION-SYSTEM

AI powered **Human Activity Recognition System** for real-time monitoring — built for on-board space experiment validation (BAS•AI — Smart India Hackathon 2026).

---

## 📖 Overview

BAS•AI ek intelligent on-board AI system hai jo live camera feed se **persons detect** karta hai, unki **position** track karta hai, aur unki **activity** (Standing / Sitting / Walking) **justification ke saath** classify karta hai — sab kuch real-time mein.

---

## ✨ Features

- 🎯 **Real-time Pose Detection** — YOLO11n-pose se 17 body keypoints (nose, shoulders, hips, knees, ankles...)
- 👤 **Multi-Person Tracking** — har person ko stable **Track ID** milti hai (frame-to-frame)
- 📍 **Position Analysis** — bounding box, body center (pixels), aur frame region (jaise `middle-center`)
- 🧠 **Activity Recognition with Justification** — har decision ke saath transparent reasons:
  - 🧍 **Standing** — taangein seedhi (knee angle ≥ 155°)
  - 🪑 **Sitting** — knees mudi hui (knee angle < 125°)
  - 🚶 **Walking** — jitter-proof detection: net displacement + direction consistency + hysteresis (4-frame confirm)
  - ❓ **Unknown** — legs visible nahi, to galat claim nahi
- 🖥️ **Streamlit Dashboard** — Live Monitor, Activity History, Analytics, System Status
- 📊 **Live Metrics** — activity, confidence %, persons count, activity stream (1-sec logging)
- 📱 **Dual Camera Support** — Laptop webcam ya mobile camera (IP Webcam / DroidCam over Wi-Fi)

---

## 🗂️ Project Structure

```
AI-Recognition-System/
├── app.py                    # Streamlit dashboard (main entry)
├── camera.py                 # Live camera feed + UI integration
├── pose_detection.py         # ⭐ Core: pose + position + activity + justification
├── activity_detection.py     # Legacy activity logic (pose_detection.py me merged)
├── object_detection.py       # Object detection module
├── live_pose_detection.py    # Standalone webcam demo (bina UI ke)
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

Purana interface `detect_pose(frame)` bhi available hai (backward compatible).

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

Browser mein `http://localhost:8501` khulega → sidebar se **📡 Live Monitor** → **🟢 Start Camera**

> ⚠️ Pehli baar run karne par `yolo11n-pose.pt` model (~6 MB) auto-download hoga.

### Standalone demo (bina UI ke)

```bash
python pose_detection.py        # webcam + console output (position + justification)
python live_pose_detection.py   # simple webcam skeleton demo
```

### 📱 Mobile camera connect karna

1. Phone par **IP Webcam** (ya DroidCam) app install karo
2. Phone aur laptop **same Wi-Fi** par ho
3. App mein video stream start karo — IP milega (jaise `192.168.1.5:8080`)
4. Live Monitor → **Connect with Mobile** → IP + port daalo

---

## 🧠 Activity Detection Logic

| Activity | Signal | Threshold |
|----------|--------|-----------|
| 🚶 Walking | Body center consistently move karta hai | movement > 4 px/frame **+** net displacement > 25 px **+** direction consistency ≥ 0.55 **+** 4-frame streak |
| 🪑 Sitting | Knees mudi hui | avg knee angle < 125° |
| 🧍 Standing | Taangein seedhi | avg knee angle ≥ 155° |
| ❓ Unknown | Legs clearly visible nahi | — |

**Walking jitter-proof kyun hai?** Standing ke time body/camera micro-jitter (2–5 px random movement) hota hai. Isliye sirf per-frame movement nahi — **net displacement** (start→end seedha distance) aur **direction consistency** (net ÷ total path) bhi check hote hain. Jitter mein net displacement ~0 hota hai, walking mein bada. Simulation tests se verified ✓

---

## 📊 Dashboard Pages

| Page | Kya hai |
|------|---------|
| 🏠 Dashboard | Mission overview + recent activity |
| 📡 Live Monitor | Live video + per-person position & justification |
| 🕒 Activity History | Saare detections (date/time/activity/confidence/status) |
| 📊 Analytics | Activity distribution chart + average confidence |

---

## 🛰️ Mission

**BAS Experiment** — On-board space intelligence for crew activity monitoring during space-based experiments.

**Event:** Smart India Hackathon 2026 🏆

---

<div align="center">

**BAS•AI** | ON-BOARD HUMAN ACTIVITY RECOGNITION

</div>
