"""
Test: deployment readiness (public HF Spaces / Docker deployment).

Verifies:
  1. requirements.txt is pinned and complete (incl. web deps).
  2. .streamlit/config.toml production settings exist.
  3. Dockerfile: uvicorn CMD, healthcheck, weights copied, no secrets.
  4. run_app.py builds st.App with the health route.
  5. camera sources: laptop + mobile only (web browser camera removed).
  6. YOLO weights committed/present; .gitignore un-ignores them.
  7. Share section deps (qrcode) importable.
  8. Secret hygiene: no obvious keys/secret files in the repo.

Run:  python test_deployment_prep.py
"""

import os
import re
import sys

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
check("/health route", "/health" in route_paths)
check("web frame ingest route removed",
      "/_basai/frame" not in route_paths)

# -----------------------------------------------------
print("\n--- TEST 5: camera.py web-source removal ---")

cam_src = read("camera.py")
check("web source removed from camera picker",
      "🌐 Web Browser Camera" not in cam_src)
check("no web_camera imports left",
      "import web_camera" not in cam_src
      and "web_camera." not in cam_src)
check("web_camera.py file deleted", not os.path.exists(
    os.path.join(ROOT, "web_camera.py")))
run_app_src = read("run_app.py")
check("run_app.py no longer imports web_camera",
      "web_camera" not in run_app_src)

# -----------------------------------------------------
print("\n--- TEST 6: YOLO weights present + tracked ---")

for w in ["yolo11n.pt", "yolo11n-pose.pt"]:
    p = os.path.join(ROOT, w)
    check(f"{w} exists", os.path.exists(p)
          and os.path.getsize(p) > 1_000_000)

gi = read(".gitignore")
check(".gitignore un-ignores deployment weights",
      "!yolo11n.pt" in gi and "!yolo11n-pose.pt" in gi)

# -----------------------------------------------------
print("\n--- TEST 7: Share section dependency ---")

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
print("\n--- TEST 8: secret hygiene ---")

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
