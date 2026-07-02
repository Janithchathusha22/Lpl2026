"""Start the bundled FastAPI prediction engine for a Streamlit process."""

from __future__ import annotations

import threading
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


_lock = threading.Lock()
_server_thread: threading.Thread | None = None
_startup_error: BaseException | None = None


def _is_ready(api_url: str) -> bool:
    try:
        with urlopen(f"{api_url.rstrip('/')}/", timeout=1) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _serve(host: str, port: int) -> None:
    global _startup_error
    try:
        import uvicorn

        uvicorn.run(
            "backend:app",
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
    except BaseException as exc:
        _startup_error = exc


def ensure_prediction_api(api_url: str, timeout: float = 75) -> tuple[bool, str | None]:
    """Ensure a localhost API is available and return ``(ready, error)``."""
    global _server_thread, _startup_error

    if _is_ready(api_url):
        return True, None

    parsed = urlparse(api_url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        return False, f"External prediction API is unreachable: {api_url}"

    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    with _lock:
        if _server_thread is None or not _server_thread.is_alive():
            _startup_error = None
            _server_thread = threading.Thread(
                target=_serve,
                args=(host, port),
                name="lpl-prediction-api",
                daemon=True,
            )
            _server_thread.start()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_ready(api_url):
            return True, None
        if _startup_error is not None:
            return False, str(_startup_error)
        time.sleep(0.25)

    return False, "Prediction engine startup timed out."
