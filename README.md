# 🛰️ BAS-AI — AI-Recognition-System

Real-time human-activity recognition + verifiable task completion from a
single camera: YOLO11 Pose + YOLO11 Object Detection + object tracking +
temporal confirmation. No fake results — every completion is backed by
real keypoints, real detections and a 12-frame confirmation streak.

Works on **localhost** (desktop webcam) *and* as a **public web app**
(visitors use their own device camera through the browser).

---

## ✅ Quick start (localhost — unchanged)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 → **Live Monitor** → **Task Verification** →
choose a camera source → **START CAMERA**.

The task list, supported tasks and honest refusals are documented in
[`TASK_LIST.txt`](TASK_LIST.txt).

---

## 🌐 Public deployment (Hugging Face Spaces, free)

The recommended target is **Hugging Face Spaces with the Docker SDK**
(free, 2 vCPU, always-on container, public HTTPS URL). The repo ships
with everything needed: `Dockerfile`, `.streamlit/config.toml`,
`run_app.py` (production launcher) and the two YOLO weights.

### Why not other platforms?

| Option | Verdict |
|---|---|
| **HF Spaces (Docker)** | ✅ recommended — free, websocket proxying works, persistent container |
| Railway / Fly.io | works too (same Dockerfile), paid |
| Streamlit Community Cloud | ❌ 0.7 vCPU — far too weak for real-time YOLO |
| Vercel / Netlify | ❌ serverless — kills the inference worker |
| WebRTC (`streamlit-webrtc`) | ❌ needs STUN/TURN servers that PaaS containers don't provide |

### Deploy in 6 steps

1. **Push this repo to GitHub** (the two `*.pt` weight files are part of
   it — they are whitelisted in `.gitignore` on purpose).

2. **Create the Space**: [huggingface.co/new-space](https://huggingface.co/new-space)
   → Space name e.g. `BAS-AI` → **License**: choose one →
   **SDK**: select **Docker** → **Blank** template → create.

3. **Connect the code** — either link the GitHub repo (Settings →
   *Repository sync*) or upload the files:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/BAS-AI
   # copy all project files into that folder, then:
   cd BAS-AI
   git add . && git commit -m "BAS-AI initial deployment" && git push
   ```
   Required files: `app.py`, `camera.py`, `camera_worker.py`,
   `web_camera.py`, `run_app.py`, `pose_detection.py`,
   `object_detection.py`, `object_tracking.py`, `object_tasks.py`,
   `task_detection.py`, `step_validator.py`, `live_task_suite.py`,
   `task_eval.py`, `requirements.txt`, `Dockerfile`,
   `.streamlit/config.toml`, `yolo11n.pt`, `yolo11n-pose.pt`,
   `README.md`.

4. **Wait for the build** (~5–10 min first time: CPU torch + opencv are
   large). The Space log shows the YOLO warmup line `weights ok` before
   the app goes live.

5. **Your public URL is ready**:
   `https://<your-username>-<space-name>.hf.space`
   Open it on any phone or laptop → the sidebar **🔗 Share BAS-AI**
   section shows the same URL with a **QR code** — scan it from a phone.

6. **(Optional) pin the URL for the Share section** — the app
   auto-detects its own URL from the browser, but you can hardcode it:
   Space → **Settings** → *Variables and secrets* → add a **variable**
   (not a secret — it is not sensitive):
   - Name: `BASAI_PUBLIC_URL`
   - Value: `https://<your-username>-<space-name>.hf.space`

> **Status:** the deployment files are prepared and the ingest
> pipeline is simulation-tested. The app is **not yet publicly
> accessible** — complete steps 1–5 above, then verify per the
> checklist below.

### Using the camera from the browser (public visitors)

- Camera source **🌐 Web Browser Camera** is the default on the public
  URL (localhost keeps **Laptop Camera** by default).
- The browser asks for **camera permission** once, over HTTPS. Denying
  it shows a retry hint instead of breaking the app.
- Browser frames are downscaled to 640 px and uploaded to
  `/_basai/frame` (~12 FPS); the same YOLO pose/object pipeline,
  tracking and task verification runs on them unchanged.
- Users can revoke permission at any time via the browser address bar;
  the app shows a stopped-stream warning with a retry hint.

---

## 🏗️ Architecture

```
Browser (HTTPS)                      Container (HF Space / Docker)
┌─────────────────────────┐          ┌────────────────────────────────┐
│ getUserMedia → canvas   │  JPEG    │ Starlette route POST           │
│ → JPEG (640px, ~12fps)  │ ───────► │ /_basai/frame                  │
└─────────────────────────┘          │   → session frame buffer       │
                                     │        ↓                       │
localhost: cv2.VideoCapture ───────► │ camera_worker.CameraWorker     │
                                     │  (reader + throttled YOLO)     │
                                     │   pose YOLO ~10fps             │
                                     │   object YOLO ~7fps (opt-in)   │
                                     │        ↓                       │
                                     │ task verification + tracking   │
                                     │        ↓                       │
                                     │ Streamlit UI (st.App)          │
                                     └────────────────────────────────┘
```

Key files:

| File | Role |
|---|---|
| `run_app.py` | production launcher: `st.App` + ingest/health routes (uvicorn) |
| `web_camera.py` | browser capture component + session frame buffers |
| `camera_worker.py` | single background inference worker per session |
| `camera.py` | Live Monitor UI, camera sources, task verification |
| `task_detection.py` / `object_tasks.py` | task logic (never fakes results) |

### Local production check (no Docker needed)

```bash
pip install -r requirements.txt
python run_app.py            # serves on PORT (default 8501) via uvicorn
# or: uvicorn run_app:app --host 0.0.0.0 --port 8501
```

---

## 🔒 Security notes

- No API keys, secrets or tokens exist in this project; `.env` /
  `.streamlit/secrets.toml` are git-ignored and never shipped.
- The frame-ingest route accepts only well-formed JPEG frames
  (≤ 3 MB) and stores them per-session in memory only — frames are
  never written to disk and never leave the container.
- Model weights are standard public COCO/YOLO11n files.

---

## 🧪 Tests

```bash
python test_deployment_prep.py     # deployment readiness (this README's claims)
python test_camera_worker.py       # worker lifecycle + throttling
python test_task_verification.py   # pose task verification
python test_object_tasks.py        # object task verification
python test_walking_fix.py         # walking detection
```

---

BAS-AI | AI-RECOGNITION-SYSTEM — SMART INDIA HACKATHON 2026
