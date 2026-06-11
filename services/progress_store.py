"""In-memory progress reporting for long-running training uploads.

The wizard frontend polls GET /api/packagings/{key}/training/progress while
the POST .../training/{seed|full}/start request is still in flight. Sync
FastAPI endpoints run in a threadpool, so the poll and the upload proceed
concurrently within the same process.

Single-process only by design: progress is best-effort UX, not state. If a
poll lands on a different Cloud Run instance it sees "idle" and the frontend
falls back to the indeterminate spinner.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_progress: dict[str, dict] = {}

_IDLE = {"phase": "idle", "done": 0, "total": 0, "percent": None, "detail": ""}


def report(key: str, phase: str, done: int = 0, total: int = 0, detail: str = "") -> None:
    """Replace the progress snapshot for a draft key (never mutates in place)."""
    percent = round(done / total * 100) if total > 0 else None
    snapshot = {
        "phase": phase,
        "done": done,
        "total": total,
        "percent": percent,
        "detail": detail,
    }
    with _lock:
        _progress[key] = snapshot


def get(key: str) -> dict:
    """Current snapshot for a draft key — idle placeholder when none."""
    with _lock:
        return _progress.get(key, _IDLE)


def clear(key: str) -> None:
    with _lock:
        _progress.pop(key, None)
