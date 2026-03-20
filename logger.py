"""
Structured logging setup.

Call get_logger(name) in every module to obtain a named logger that writes to
both stdout and bot.log with a consistent timestamp format.

Usage:
    from logger import get_logger
    logger = get_logger(__name__)
"""
import logging
import sys

_configured: bool = False


def setup_logging() -> None:
    """Configure the root logger with console + file handlers (idempotent)."""
    global _configured
    if _configured:
        return

    # Late import to avoid circular dependency at module load time
    from config import LOG_FILE, LOG_LEVEL

    root = logging.getLogger()

    # Remove any handlers that client.py's basicConfig may have already added
    root.handlers.clear()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler (non-fatal if the path is not writable)
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError as exc:
        root.warning(
            "Could not open log file '%s': %s — logging to console only.", LOG_FILE, exc
        )

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initialising root handlers on the first call."""
    setup_logging()
    return logging.getLogger(name)
