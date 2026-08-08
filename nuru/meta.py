"""Petites métriques système sans dépendance externe (aucun psutil)."""

from __future__ import annotations

import os
import resource
import time


def rss_mb() -> float:
    """RAM résidente du processus, en Mo (compatible Linux + macOS)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return rss / (1024.0 * 1024.0)  # macOS: octets
    return rss / 1024.0  # Linux: Ko


def uptime_s(started_at: float) -> float:
    return time.time() - started_at
