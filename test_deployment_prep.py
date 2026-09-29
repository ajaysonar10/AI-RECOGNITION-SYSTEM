"""
Test: deployment readiness (public HF Spaces / Docker deployment).

Verifies:
  1. requirements.txt is pinned and complete (incl. web deps).
  2. .streamlit/config.toml production settings exist.
  3. Dockerfile: uvicorn CMD, healthcheck, weights copied, no secrets.
  4. run_app.py builds st.App with the ingest + health routes.
  5. web_camera buffer: push/read, latest-wins, open/close lifecycle.
  6. Browser component: permission handling + upload wiring present.
  7. camera.py web-source integration present.
  8. YOLO weights committed/present; .gitignore un-ignores them.
  9. Share section deps (qrcode) importable.
 10. Secret hygiene: no obvious keys/secret files in the repo.

Run:  python test_deployment_prep.py
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


def read(path):
    with open(os.path.join(ROOT, path), "r", encoding="utf-8",
              errors="replace") as f:
        return f.read()


# -----------------------------------------------------
print("\n--- TEST 1: requirements.txt pinned + complete ---")

reqs = read("requirements.txt")
pin_map = {}
for line in reqs.splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "==" in line:
        name, ver = line.split("==", 1)
        pin_map[name.strip().lower()] = ver

for lib in ["streamlit", "ultralytics", "opencv-python",
            "numpy", "pandas", "pillow", "qrcode", "uvicorn"]:
    check(f"{lib} pinned", lib in pin_map,
          "(missing from requirements.txt)")

check("no unpinned core libs",
      not re.search(r"^(streamlit|ultralytics|opencv-python|numpy)$",
                    "\n".join(
                        l.strip() for l in reqs.splitlines()
                        if l.strip() and not l.startswith("#")
                    ), re.M))

# -----------------------------------------------------
print("\n--- TEST 2: .streamlit/config.toml ---")

cfg = read(os.path.join(".streamlit", "config.toml"))
check("headless server", "headless" in cfg and "true" in cfg)
check("binds 0.0.0.0", 'address = "0.0.0.0"' in cfg)
check("no secrets in config", "secrets" not in cfg.lower())

# -----------------------------------------------------
print("\n--- TEST 3: Dockerfile ---")

docker = read("Dockerfile")
check("uvicorn launch command", "uvicorn run_app:app" in docker)
check("healthcheck present", "HEALTHCHECK" in docker
      and "/health" in docker)
check("weights copied", "yolo11n.pt" in docker
      and "yolo11n-pose.pt" in docker)
check("runs as non-root", "USER appuser" in docker)
check("no private folder shipped",
      "AI-Recognition-System" not in docker)
check("PORT exposed", "EXPOSE" in docker)

# -----------------------------------------------------
print("\n--- TEST 4: run_app.py builds st.App with routes ---")

import run_app  # noqa: E402  (builds the App object)

route_paths = [
    getattr(r, "path", "") for r in run_app.app._user_routes
]
check("/_basai/frame ingest route", "/_basai/frame" in route_paths)
check("/health route", "/health" in route_paths)
check("ingest route is POST-only",
      any(getattr(r, "methods", None) == {"POST"}
          for r in run_app.app._user_routes
          if getattr(r, "path", "") == "/_basai/frame"))

# -----------------------------------------------------
print("\n--- TEST 5: web_camera buffer lifecycle ---")

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import web_camera  # noqa: E402

buf = web_camera.WebFrameBuffer()
check("fresh buffer counts as open (worker can start)",
      buf.isOpened() is True)

frame = np.zeros((480, 640, 3), np.uint8)
frame[:, :, 0] = 7
ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
check("push decodes JPEG", buf.push(enc.tobytes()) is True)
ret, got = buf.read()
check("read returns newest frame",
      ret is True and got.shape == (480, 640, 3))

# latest-wins semantics (stale frames never queue up -> no lag)
for i in range(3):
    f = np.full((480, 640, 3), i + 10, np.uint8)
    _, e = cv2.imencode(".jpg", f)
    buf.push(e.tobytes())
ret, got = buf.read()
check("latest-wins (stale frames dropped)",
      ret is True and int(got[0, 0, 0]) == 12)
ret2, _ = buf.read()
check("buffer drained after read", ret2 is False)

time.sleep(0.1)
check("fresh buffer still open before first frame",
      web_camera.WebFrameBuffer().isOpened() is True)

buf.release()
check("release closes buffer",
      buf.isOpened() is False and buf.read() == (False, None))

# registry helpers
rb = web_camera.get_buffer("deploy-test-sess")
check("registry returns stable buffer per session",
      web_camera.get_buffer("deploy-test-sess") is rb)
web_camera.drop_buffer("deploy-test-sess")
check("drop_buffer removes session",
      web_camera.get_buffer("deploy-test-sess") is not rb)

# -----------------------------------------------------
print("\n--- TEST 6: browser component wiring ---")

tmpl = web_camera._COMPONENT_TMPL
check("uses getUserMedia", "getUserMedia" in tmpl)
check("handles permission denial (NotAllowedError)",
      "NotAllowedError" in tmpl)
check("handles no-device", "NotFoundError" in tmpl)
check("handles insecure context", "isSecureContext" in tmpl)
check("posts JPEG to ingest route", "toDataURL" in tmpl
      and "fetch(UPLOAD_URL" in tmpl)
check("sends session id", "SESSION_ID" in tmpl)
check("session id placeholder is substituted",
      "SESSION_ID" not in tmpl.replace("SESSION_ID", "", 1))

# -----------------------------------------------------
print("\n--- TEST 7: camera.py web-source integration ---")

cam_src = read("camera.py")
check("web source in camera picker",
      "🌐 Web Browser Camera" in cam_src)
check("_tv_open_camera dispatches web stream",
      "_tv_open_camera" in cam_src
      and "WebVideoStream" in cam_src)
check("stop closes web buffer",
      "drop_buffer" in cam_src)
check("web module optional (localhost keeps working)",
      "except ImportError" in cam_src and "web_camera = None" in cam_src)

# -----------------------------------------------------
print("\n--- TEST 8: YOLO weights present + tracked ---")

for w in ["yolo11n.pt", "yolo11n-pose.pt"]:
    p = os.path.join(ROOT, w)
    check(f"{w} exists", os.path.exists(p)
          and os.path.getsize(p) > 1_000_000)

gi = read(".gitignore")
check(".gitignore un-ignores deployment weights",
      "!yolo11n.pt" in gi and "!yolo11n-pose.pt" in gi)

# -----------------------------------------------------
print("\n--- TEST 9: Share section dependency ---")

try:
    import qrcode  # noqa: F401
    check("qrcode importable", True)
except ImportError:
    check("qrcode importable", False,
          "(pip install qrcode[pil])")

app_src = read("app.py")
check("Share section renders URL + QR",
      "_render_share_section" in app_src
      and "qrcode" in app_src)
check("BASAI_PUBLIC_URL override supported",
      "BASAI_PUBLIC_URL" in app_src)

# -----------------------------------------------------
print("\n--- TEST 10: secret hygiene ---")

secret_files = [".env", ".streamlit/secrets.toml"]
check("no .env committed",
      not os.path.exists(os.path.join(ROOT, ".env")))
check("no secrets.toml committed",
      not os.path.exists(
          os.path.join(ROOT, ".streamlit", "secrets.toml")))

key_pattern = re.compile(
    r"(sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|"
    r"(api[_-]?key|secret|password)\s*=\s*[\"'][^\"']{8,}[\"'])",
    re.I,
)
leaks = []
for fn in os.listdir(ROOT):
    if fn.endswith(".py") and fn != "AI-Recognition-System":
        try:
            with open(fn, "r", encoding="utf-8",
                      errors="replace") as f:
                if key_pattern.search(f.read()):
                    leaks.append(fn)
        except OSError:
            pass
check("no hardcoded keys in root .py files", not leaks,
      f"(found in: {leaks})")

# -----------------------------------------------------
print(f"\n{'=' * 46}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'=' * 46}")
sys.exit(1 if failed else 0)
