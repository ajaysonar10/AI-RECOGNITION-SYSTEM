"""
build_exe.py
============
Builds the BAS-AI standalone Windows .exe with PyInstaller.

Run:
    python build_exe.py              -> SINGLE-FILE .exe (default)
    python build_exe.py --onedir     -> folder-based .exe (faster startup)

Output:
    dist/BAS-AI.exe            (single file mode)
    dist/BAS-AI/BAS-AI.exe     (folder mode)

The .exe includes:
  - All source modules (app.py, camera.py, pose_detection.py, ...)
  - Both YOLO models (yolo11n-pose.pt, yolo11n.pt)
  - Streamlit runtime + static assets
  - pywebview + WebView2 loader
  - pyttsx3 voice engine data

Note: the single-file .exe unpacks itself to a temp folder on each
launch, so the first window can take 30-60 s to appear. The folder
build starts faster.
"""

import os
import subprocess
import sys

APP = "desktop_app.py"
NAME = "BAS-AI"

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(">>", " ".join(cmd))
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        sys.exit(f"Command failed with exit code {result.returncode}")


def main():
    # --------------------------------------------------------
    # Streamlit ships large static asset folders (frontend JS/CSS)
    # that PyInstaller cannot find automatically. Collect them.
    # --------------------------------------------------------
    import streamlit
    import webview
    import pyttsx3

    streamlit_dir = os.path.dirname(streamlit.__file__)
    webview_dir = os.path.dirname(webview.__file__)

    data_args = [
        # ---- Streamlit frontend + runtime ----
        f"--add-data={streamlit_dir}{os.pathsep}streamlit",
        # ---- pywebview (js interface files) ----
        f"--add-data={webview_dir}{os.pathsep}webview",
        # ---- YOLO model weights (offline AI) ----
        f"--add-data={os.path.join(HERE, 'yolo11n-pose.pt')}{os.pathsep}.",
        f"--add-data={os.path.join(HERE, 'yolo11n.pt')}{os.pathsep}.",
    ]

    hidden = [
        # ---- Streamlit internals ----
        "--hidden-import=streamlit.runtime.scriptrunner.magic_funcs",
        "--hidden-import=streamlit.web.cli",
        "--hidden-import=streamlit.web.bootstrap",
        # ---- pywebview backends ----
        "--hidden-import=webview.platforms.winforms",
        "--hidden-import=webview.platforms.edgechromium",
        "--hidden-import=webview.platforms.cef",
        "--hidden-import=clr_loader",
        "--hidden-import=pythonnet",
        # ---- pyttsx3 voice drivers ----
        "--hidden-import=pyttsx3.drivers",
        "--hidden-import=pyttsx3.drivers.sapi5",
        "--hidden-import=win32com.client",
        "--hidden-import=win32com.gen_py",
        # ---- ultralytics/torch support ----
        "--hidden-import=ultralytics",
        "--hidden-import=ultralytics.nn.tasks",
        "--hidden-import=ultralytics.utils.__init__",
        # ---- misc runtime deps ----
        "--hidden-import=altair",
        "--hidden-import=pandas",
        "--hidden-import=pyarrow",
        "--hidden-import=pil",
    ]

    excludes = [
        # Shrink the bundle: GUI toolkits we do not use
        "--exclude-module=tkinter",
        "--exclude-module=matplotlib",
        "--exclude-module=IPython",
        "--exclude-module=notebook",
        "--exclude-module=jupyter",
        "--exclude-module=pytest",
    ]

    onedir = "--onedir" in sys.argv

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir" if onedir else "--onefile",
        "--console",                # keep console for first-run diagnostics
        f"--name={NAME}",
        *data_args,
        *hidden,
        *excludes,
        APP,
    ]

    run(cmd)

    print()
    print("=" * 60)

    if onedir:
        folder = os.path.join(HERE, "dist", NAME)
        total = 0
        for root, _, files in os.walk(folder):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        print(f"BUILD OK -> dist/{NAME}/{NAME}.exe  "
              f"(folder: {total / (1024 * 1024):.0f} MB)")
        print("Share the whole dist/BAS-AI folder (zip it) - the .exe")
        print("needs the DLLs next to it. Double-click BAS-AI.exe.")
    else:
        exe = os.path.join(HERE, "dist", f"{NAME}.exe")
        size = os.path.getsize(exe) / (1024 * 1024)
        print(f"BUILD OK -> dist/{NAME}.exe  (single file: {size:.0f} MB)")
        print("This ONE file is the whole app - share it directly.")
        print("First launch unpacks itself, give it 30-60 seconds.")

    print("=" * 60)


if __name__ == "__main__":
    main()
