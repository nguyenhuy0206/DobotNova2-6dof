"""Logging setup: console + rotating file in data/logs/."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from common.paths import LOGS_DIR

_CONFIGURED = False


def setup_logging(level: str = "INFO", filename: str = "nova2.log") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level.upper())
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file = RotatingFileHandler(LOGS_DIR / filename, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
