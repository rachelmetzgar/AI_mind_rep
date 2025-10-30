"""
utils/log_helpers.py

Lightweight logging utility for simulation scripts.
Writes timestamped messages to both stdout and a shared or per-run log file.

Author: Rachel C. Metzgar
Date: 2025-10-08
"""

from datetime import datetime
from pathlib import Path

def log_message(msg: str, log_file: Path):
    """Append timestamped message to stdout and a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")
