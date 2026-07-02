"""Streamlit Community Cloud entrypoint for the LPL 2026 dashboard.

The dashboard talks to the existing FastAPI prediction engine over localhost.
Community Cloud starts only one command, so this entrypoint starts that engine
in a daemon thread before executing the Streamlit UI.
"""

from __future__ import annotations

import os
import runpy
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
API_HOST = "127.0.0.1"
API_PORT = int(os.environ.get("LPL_API_PORT", "8000"))
API_URL = f"http://{API_HOST}:{API_PORT}"

os.chdir(ROOT)
os.environ.setdefault("LPL_API_URL", API_URL)

_startup_error: list[BaseException] = []


def _api_is_ready() -> bool:
    try:
        with urlopen(f"{API_URL}/", timeout=1) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _run_api() -> None:
    try:
        import uvicorn

        uvicorn.run(
            "backend:app",
            host=API_HOST,
            port=API_PORT,
            log_level="warning",
            access_log=False,
        )
    except BaseException as exc:
        _startup_error.append(exc)


if not _api_is_ready():
    threading.Thread(
        target=_run_api,
        name="lpl-prediction-api",
        daemon=True,
    ).start()

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not _api_is_ready() and not _startup_error:
        time.sleep(0.25)

if not _api_is_ready():
    import streamlit as st

    st.set_page_config(page_title="LPL 2026", page_icon="🏏", layout="wide")
    details = str(_startup_error[0]) if _startup_error else "startup timed out"
    st.error(f"The prediction engine could not start: {details}")
    st.stop()

# run_path is intentional: Streamlit reruns must execute the complete UI script.
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
