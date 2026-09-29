# BAS-AI — Hugging Face Spaces (Docker SDK) / generic container image
# Public URL after deploy: https://<owner>-<space>.hf.space

FROM python:3.11-slim

# OpenCV needs these system libs at runtime; libgl1/libglib are the
# classic cv2 import blockers on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cpu

# App code + the two YOLO weights (committed to the repo by choice)
COPY app.py camera.py camera_worker.py web_camera.py run_app.py \
     pose_detection.py object_detection.py object_tracking.py \
     object_tasks.py task_detection.py step_validator.py \
     live_task_suite.py task_eval.py \
     ./
COPY yolo11n.pt yolo11n-pose.pt ./
COPY .streamlit ./.streamlit/

# Non-root user (platform best practice)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# HF Spaces expects the app on PORT (default 7860)
ENV PORT=7860
EXPOSE 7860

# Warm the models once so the first visitor doesn't wait ~20 s for
# YOLO cold start (also fails fast if a weight file is corrupted).
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-pose.pt'); print('weights ok')"

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn run_app:app --host 0.0.0.0 --port ${PORT}"]
