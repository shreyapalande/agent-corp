"""
Central logging configuration for AgentCorp.

All modules import `get_logger` from here:

    from utils.logger import get_logger
    logger = get_logger(__name__)

Log levels:
- Console: INFO and above
- File:    DEBUG and above  (logs/agentcorp.log, rotating 5 MB × 3 backups)
"""

import logging
import logging.handlers
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = _LOG_DIR / "agentcorp.log"

# ── Format ────────────────────────────────────────────────────────────────────
_FMT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'agentcorp' hierarchy.

    Pass the module name:
        logger = get_logger(__name__)
    or a short label:
        logger = get_logger("nodes")
    """
    # Ensure the root logger is configured before handing out children
    _ensure_configured()
    label = name if name.startswith("agentcorp") else f"agentcorp.{name}"
    return logging.getLogger(label)


def _ensure_configured() -> None:
    root = logging.getLogger("agentcorp")
    if root.handlers:
        return  # already set up

    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Console handler — INFO+
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    # Rotating file handler — DEBUG+
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file_handler)
    root.propagate = False  # don't bubble up to the root Python logger
