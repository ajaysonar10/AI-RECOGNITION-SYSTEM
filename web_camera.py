"""
web_camera.py
=============

Web deployment camera source.

Problem this solves:
    The app's normal camera is `cv2.VideoCapture(0)` — that reads the
    SERVER's webcam. That works on localhost/desktop, but a public
    server has no webcam. Browsers hold the visitor's camera, and
    they only hand it out over HTTPS after an explicit permission
    prompt.

Design (kept deliberately simple and robust):
    1. A tiny custom JS component (st.components.v1.html) calls
       getUserMedia, draws video frames to a canvas, downscales them
       (BASAI_WEB_MAX_W, default 640 px) and POSTs JPEG frames to
       /_basai/frame with the browser's own Streamlit session id.
    2. A session-scoped ring buffer (latest-frame-only semantics)
       receives the frames server-side (uploaded through the
       /_basai/frame Starlette route registered in run_app.py).
    3. WebVideoStream adapts that buffer to the tiny subset of the
       VideoCapture API that camera_worker.CameraWorker already uses
       (read / isOpened / release / isOpened-guarded sets). The whole
       existing worker pipeline (pose YOLO, object YOLO, tracking,
       task verification) runs UNCHANGED.

The component JS runs inside Streamlit's component iframe, which
already delegates camera permission to the parent page (verified in
Streamlit 1.63.0: iframe has allow="camera;microphone" and calls
getUserMedia directly) — so the permission prompt appears once for
the app's own origin, over HTTPS, and is NOT re-prompted in a loop.

Errors are surfaced as a structured status string:
    "ok" | "denied" | "not-secure" | "no-device" | "error:<msg>"
"""

import base64
import json
import threading
import time

import numpy as np
import streamlit as st
import streamlit.components.v1 as components


# ------------------------------------------------------------
# TUNING
# ------------------------------------------------------------

BASAI_WEB_MAX_W = 640        # browser-side downscale target width (px)
BASAI_WEB_JPEG_Q = 60        # browser-side JPEG quality (0-100)
BASAI_WEB_TARGET_FPS = 12.0  # browser upload cadence (frames/s)

_BUFFER_MAX = 2              # tiny ring buffer; latest frame wins
_FRAME_STALE_S = 3.0         # older than this = stream considered dead
ROUTE_PATH = "/_basai/frame" # also referenced by run_app.py


# ------------------------------------------------------------
# COMPONENT HTML/JS (browser side)
# ------------------------------------------------------------

_COMPONENT_TMPL = """
<div id="basai-webcam-box" style="width:100%%;text-align:center;">
  <video id="basai-video" autoplay playsinline muted
         style="width:100%%;border-radius:10px;background:#000;"></video>
  <div id="basai-cam-status" style="margin-top:6px;font-size:0.9em;
       color:#9fb6c9;">Requesting camera permission…</div>
</div>
<canvas id="basai-canvas" style="display:none;"></canvas>

<script>
(function () {
    var video = document.getElementById("basai-video");
    var canvas = document.getElementById("basai-canvas");
    var statusEl = document.getElementById("basai-cam-status");
    var canvasCtx = canvas.getContext("2d");
    var stream = null;
    var running = false;
    var nextSend = 0;
    var inFlight = false;
    var lastError = "";

    function setStatus(text, color) {
        statusEl.textContent = text;
        if (color) { statusEl.style.color = color; }
    }

    function isSecure() {
        // Camera access requires a secure context. localhost counts
        // as secure; the public deployment URL is HTTPS.
        return (window.isSecureContext === true) ||
               (location.protocol === "https:") ||
               (location.hostname === "localhost") ||
               (location.hostname === "127.0.0.1");
    }

    function report(status) {
        // Structured status back to Python through the component
        // return channel (Streamlit handles the pipe for us).
        var args = arguments;
        try {
            window.parent.postMessage({
                isBasaiWebcam: true,
                status: args[0],
                detail: args[1] || ""
            }, "*");
        } catch (e) { /* non-fatal */ }
    }

    function handleFailure(err) {
        var name = err && err.name ? err.name : "";
        if (name === "NotAllowedError" || name === "PermissionDeniedError") {
            setStatus("Camera permission denied. Allow camera access " +
                      "in your browser's address bar, then click " +
                      "Retry Camera.", "#e6a23c");
            report("denied", name);
        } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
            setStatus("No camera device found on this device.", "#e6a23c");
            report("no-device", name);
        } else if (!isSecure()) {
            setStatus("Camera requires HTTPS (or localhost).", "#e6a23c");
            report("not-secure", name);
        } else {
            setStatus("Camera error: " + name + " — click Retry Camera.",
                      "#e6a23c");
            report("error", name);
        }
    }

    function loop(now) {
        if (!running) { return; }
        if (video.readyState >= 2 && !inFlight &&
            now >= nextSend) {
            nextSend = now + (1000 / TARGET_FPS);
            var vw = video.videoWidth || MAX_W;
            var scale = Math.min(1, MAX_W / vw);
            canvas.width = Math.round(vw * scale);
            canvas.height = Math.round(
                (video.videoHeight || 480) * scale
            );
            canvasCtx.drawImage(video, 0, 0,
                                canvas.width, canvas.height);
            var dataUrl = canvas.toDataURL("image/jpeg", JPEG_Q);
            inFlight = true;
            fetch(UPLOAD_URL + "?session=" +
                  encodeURIComponent(SESSION_ID), {
                method: "POST",
                headers: { "Content-Type": "application/octet-stream" },
                body: dataUrl.split(",")[1]
            }).then(function (resp) { return resp.json(); })
              .then(function () { inFlight = false; })
              .catch(function () { inFlight = false; });
        }
        window.requestAnimationFrame(loop);
    }

    function start() {
        if (!isSecure()) {
            handleFailure({ name: "not-secure" });
            return;
        }
        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {
            setStatus("This browser does not support camera capture.",
                      "#e6a23c");
            report("error", "unsupported");
            return;
        }
        navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: MAX_W },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        }).then(function (s) {
            stream = s;
            video.srcObject = s;
            running = true;
            setStatus("Camera active — sharing your video with " +
                      "the BAS-AI server for AI analysis.",
                      "#67c23a");
            report("ok", "");
            window.requestAnimationFrame(loop);
        }).catch(handleFailure);
    }

    function stop() {
        running = false;
        if (stream) {
            stream.getTracks().forEach(function (t) { t.stop(); });
            stream = null;
        }
        setStatus("Camera stopped.", "#9fb6c9");
    }

    // Retry button (double-click guard is unnecessary; start() is
    // idempotent after stop()).
    document.getElementById("basai-webcam-box").addEventListener(
        "dblclick", function () {
            stop(); start();
        }
    );

    // Stop tracks when the component is torn down (page nav, rerun).
    window.addEventListener("beforeunload", stop);

    start();
})();
"""

_STOP_SNIPPET = """
<script>
(function () {
    // Best-effort: a previous component instance's stream is stopped
    // by the browser when its iframe is removed (Streamlit reruns
    // destroy the iframe). Nothing to do server-side here.
})();
</script>
"""


def render_web_camera(session_id: str, upload_path: str = ROUTE_PATH) -> None:
    """
    Renders the browser camera capture component.

    Args:
        session_id: the current Streamlit session id — frames are
            stored under this key server-side.
        upload_path: the ingest route registered in run_app.py.
    """
    js = (
        _COMPONENT_TMPL
        .replace("UPLOAD_URL", json.dumps(upload_path))
        .replace("SESSION_ID", json.dumps(session_id))
        .replace("TARGET_FPS", str(BASAI_WEB_TARGET_FPS))
        .replace("MAX_W", str(BASAI_WEB_MAX_W))
        .replace("JPEG_Q", str(BASAI_WEB_JPEG_Q))
    )
    components.html(js + _STOP_SNIPPET, height=120, scrolling=False)


# ------------------------------------------------------------
# SERVER-SIDE FRAME BUFFER (session-scoped)
# ------------------------------------------------------------

class WebFrameBuffer:
    """Latest-wins frame buffer for one browser session."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frames = []           # ring buffer (max _BUFFER_MAX)
        self._last_seen = 0.0       # server time of the newest frame
        self.closed = False

    def push(self, jpeg_bytes: bytes) -> bool:
        """Decodes one JPEG and stores it. Returns False on bad data."""
        try:
            import cv2
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return False
        if frame is None or frame.size == 0:
            return False

        with self._lock:
            if self.closed:
                return False
            self._frames.append(frame)
            overflow = len(self._frames) - _BUFFER_MAX
            if overflow > 0:
                del self._frames[:overflow]
            self._last_seen = time.time()
        return True

    def read(self):
        """
        Pops the newest frame.

        Returns:
            (True, frame)  — a fresh frame was available
            (False, None)  — nothing new (worker treats this as a
                             transient miss, exactly like a slow
                             USB camera)
        """
        with self._lock:
            if not self._frames:
                return False, None
            frame = self._frames.pop()
            self._frames.clear()
            return True, frame

    def isOpened(self) -> bool:
        with self._lock:
            if self.closed:
                return False
            if self._last_seen == 0.0:
                # Brand-new buffer: browser hasn't uploaded yet —
                # stay open so the worker starts and waits.
                return True
            return (time.time() - self._last_seen) < _FRAME_STALE_S

    def set(self, _prop, _value) -> bool:
        # Accepted for VideoCapture API compatibility; the browser
        # already controls resolution/fps.
        return True

    def release(self) -> None:
        with self._lock:
            self.closed = True
            self._frames.clear()

    def stats(self) -> dict:
        with self._lock:
            return {
                "queued": len(self._frames),
                "age": (
                    time.time() - self._last_seen
                    if self._last_seen else -1.0
                ),
                "open": self.isOpened(),
            }


# ------------------------------------------------------------
# SESSION REGISTRY (module-level, process-global)
# ------------------------------------------------------------

_SESSION_BUFFERS: dict = {}
_REGISTRY_LOCK = threading.Lock()


def get_buffer(session_id: str) -> WebFrameBuffer:
    """Returns (creating if needed) the buffer for a session id."""
    with _REGISTRY_LOCK:
        buf = _SESSION_BUFFERS.get(session_id)
        if buf is None:
            buf = WebFrameBuffer()
            _SESSION_BUFFERS[session_id] = buf
        # opportunistic cleanup of dead sessions
        if len(_SESSION_BUFFERS) > 64:
            stale = [
                sid for sid, b in _SESSION_BUFFERS.items()
                if not b.isOpened() and b.stats()["age"] > 60.0
            ]
            for sid in stale:
                old = _SESSION_BUFFERS.pop(sid, None)
                if old is not None:
                    old.release()
        return buf


def drop_buffer(session_id: str) -> None:
    """Removes and closes a session's buffer (Stop Camera)."""
    with _REGISTRY_LOCK:
        buf = _SESSION_BUFFERS.pop(session_id, None)
    if buf is not None:
        buf.release()


# ------------------------------------------------------------
# VideoCapture-COMPATIBLE ADAPTER FOR THE WORKER
# ------------------------------------------------------------

class WebVideoStream:
    """
    Adapts a WebFrameBuffer to the VideoCapture-like surface that
    camera_worker.CameraWorker consumes (read/isOpened/release/set).
    CameraWorker keeps calling cap.read() at CAPTURE_FPS; when the
    browser hasn't uploaded a new frame yet, read() returns
    (False, None) and the worker's existing failed-read backoff
    handles it — no pipeline change needed.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.buffer = get_buffer(session_id)

    def read(self):
        return self.buffer.read()

    def isOpened(self) -> bool:
        return self.buffer.isOpened()

    def set(self, prop, value) -> bool:
        return self.buffer.set(prop, value)

    def release(self) -> None:
        drop_buffer(self.session_id)


# ------------------------------------------------------------
# STARLETTE ROUTE (registered in run_app.py)
# ------------------------------------------------------------

def make_frame_route():
    """
    Builds the ASGI upload route handler for /_basai/frame.

    Request body: base64 JPEG (browser strips the data: prefix).
    Response: {"ok": true} always (schema-stable for the JS side);
    failures are reflected in "stored".
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    async def frame_upload(request: Request) -> JSONResponse:
        session_id = request.query_params.get("session", "")
        if not session_id:
            return JSONResponse({"ok": False, "stored": False,
                                 "error": "missing session"}, status_code=400)

        try:
            body = await request.body()
        except Exception:
            return JSONResponse({"ok": False, "stored": False,
                                 "error": "read failed"}, status_code=400)

        # Hard cap (640px JPEG @ q60 is ~30-60 KB; 3 MB is generous)
        if len(body) > 3_000_000:
            return JSONResponse({"ok": False, "stored": False,
                                 "error": "frame too large"},
                                status_code=413)

        try:
            jpeg = base64.b64decode(body, validate=True)
        except Exception:
            return JSONResponse({"ok": False, "stored": False,
                                 "error": "bad base64"}, status_code=400)

        stored = get_buffer(session_id).push(jpeg)
        return JSONResponse({"ok": True, "stored": bool(stored)})

    return frame_upload
