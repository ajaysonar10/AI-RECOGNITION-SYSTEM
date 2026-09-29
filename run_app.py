"""
run_app.py
==========

Production launcher for BAS-AI (public deployment).

Why this file exists:
    `streamlit run app.py` stays the developer flow for localhost.
    For the public deployment we need ONE extra HTTP endpoint
    (POST /_basai/frame) so browser cameras can upload frames to
    the server. Streamlit 1.63's official `st.App` API accepts
    custom Starlette routes, and uvicorn serves the whole thing
    under one port/origin — HTTPS on the public URL comes from the
    platform's proxy (HF Spaces terminates TLS in front of the
    container).

Run:
    uvicorn run_app:app --host 0.0.0.0 --port $PORT
    (or: python run_app.py)

Localhost still works exactly as before:
    streamlit run app.py
"""

import os

import streamlit as st
import uvicorn
from starlette.responses import JSONResponse
from starlette.routing import Route

from web_camera import ROUTE_PATH, make_frame_route

# Public URL override (used by the Share section + QR code).
# Set it to the final public URL in the platform's settings once
# known; empty = the app falls back to browser-detected origin.
PUBLIC_URL = os.environ.get("BASAI_PUBLIC_URL", "").strip().rstrip("/")


async def health(request):
    """Simple liveness probe for platforms that ping the container."""
    return JSONResponse({"status": "ok", "app": "BAS-AI"})


def build_app() -> "st.App":
    script = os.path.join(os.path.dirname(__file__), "app.py")

    return st.App(
        script,
        routes=[
            Route(ROUTE_PATH, make_frame_route(), methods=["POST"]),
            Route("/health", health, methods=["GET"]),
        ],
    )


app = build_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8501"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        # The deployment platform provides TLS; inside the container
        # we speak plain HTTP on the internal port.
        log_level="info",
        timeout_keep_alive=75,   # Streamlit websockets are long-lived
        ws_ping_interval=20.0,   # keep platform proxies from
        ws_ping_timeout=20.0,    # idle-closing the session socket
    )
