"""
desktop_app.py
==============
BAS-AI Offline Desktop Application.

Launches the Streamlit app inside a native desktop window
(no browser needed). Designed to also work when frozen into
a standalone .exe with PyInstaller (see build_exe.py).

How it works:
  1) Start the Streamlit server on a free localhost port.
  2) Wait until the server responds.
  3) Open a native pywebview window pointed at the server.
  4) When the window is closed, shut the server down.
"""

import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

try:
    import webview
except Exception:  # WebView2 missing etc. -> browser fallback
    webview = None

# ============================================================
# PATHS (work both as script and frozen .exe)
# ============================================================

if getattr(sys, "frozen", False):
    # Running as a PyInstaller .exe
    BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    # Make YOLO model files (.pt) in the bundle discoverable:
    # ultralytics resolves relative paths against the CWD.
    os.chdir(BASE_DIR)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

APP_ENTRY = os.path.join(BASE_DIR, "app.py")

PREFERRED_PORT = 8765


# ============================================================
# FREE PORT PICKER
# ============================================================

def find_free_port():
    """Use the preferred port when free; otherwise ask the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", PREFERRED_PORT))
            return PREFERRED_PORT
        except OSError:
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ============================================================
# STREAMLIT SERVER
# ============================================================

def start_streamlit(port):
    """Start the Streamlit server as a subprocess."""
    cmd = [
        sys.executable,
        "-m", "streamlit", "run", APP_ENTRY,
        "--server.headless", "true",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
        "--server.address", "127.0.0.1",
        # New tabs/links open inside the same window
        "--server.enableXsrfProtection", "false",
        "--global.showWarningOnDirectExecution", "false",
    ]

    # In a frozen .exe, sys.executable is the exe itself;
    # streamlit is bundled as a library, run via our stub.
    if getattr(sys, "frozen", False):
        cmd = [
            sys.executable,
            "--frozen-streamlit",
            "--port", str(port),
        ]

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW

    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def wait_for_server(port, timeout=90):
    """Wait until the Streamlit server responds."""
    url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)

    return False


# ============================================================
# SINGLE-INSTANCE GUARD
# ============================================================

def acquire_port_lock(port):
    """Try to bind the chosen port to detect another instance."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", port))
        return s
    except OSError:
        return None


# ============================================================
# MAIN
# ============================================================

def main():
    port = find_free_port()

    server_proc = None

    if getattr(sys, "frozen", False):
        # Frozen mode: relaunch ourselves in "server mode".
        # The .exe has two modes:
        #   desktop mode  -> open window, spawn server child
        #   server mode   -> run streamlit programmatically
        if "--frozen-streamlit" in sys.argv:
            run_frozen_streamlit(port_from_args=True)
            return

        # Desktop mode: spawn ourselves as the server child.
        server_proc = subprocess.Popen(
            [sys.executable, "--frozen-streamlit", "--port", str(port)],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    else:
        server_proc = start_streamlit(port)

    if not wait_for_server(port):
        # Fall back: just open the browser (still works offline)
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")
        if server_proc:
            server_proc.wait()
        return

    # --------------------------------------------------------
    # Native desktop window
    # --------------------------------------------------------
    if webview is not None:
        try:
            webview.create_window(
                title="BAS-AI | Human Activity Recognition",
                url=f"http://127.0.0.1:{port}",
                width=1500,
                height=920,
                min_size=(1100, 700),
                background_color="#030812",
            )
            webview.start()
        except Exception:
            # Native window failed (e.g. WebView2 runtime missing)
            # -> fall back to the default browser, still offline.
            import webbrowser
            webbrowser.open(f"http://127.0.0.1:{port}")
            try:
                server_proc.wait()
            except Exception:
                pass
            return
    else:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}")
        try:
            server_proc.wait()
        except Exception:
            pass
        return

    # Window closed -> stop the server
    if server_proc:
        try:
            server_proc.terminate()
            server_proc.wait(timeout=10)
        except Exception:
            pass


def run_frozen_streamlit(port_from_args=False):
    """
    Run Streamlit programmatically inside the frozen .exe
    (server mode of the child process).
    """
    port = 8501

    if port_from_args:
        if "--port" in sys.argv:
            port = int(sys.argv[sys.argv.index("--port") + 1])

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", APP_ENTRY,
        "--server.headless", "true",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
        "--server.address", "127.0.0.1",
        "--server.enableXsrfProtection", "false",
        "--global.showWarningOnDirectExecution", "false",
    ]

    stcli.main()


if __name__ == "__main__":
    main()
