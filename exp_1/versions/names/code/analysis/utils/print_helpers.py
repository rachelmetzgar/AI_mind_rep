"""
Script name: print_helpers.py
Purpose: Shared print-formatting utilities for behavioral and fMRI analysis scripts.
    - Standardized headers and section breaks.
    - Convenience wrappers for file/figure save messages.

Author: Rachel C. Metzgar
Date: 2025-09-29
"""

from __future__ import annotations
import os
from typing import Optional

__all__ = ["print_header", "print_save", "print_warn", "print_info"]

# --------------------------
# Console formatting helpers
# --------------------------


def print_header(title: str) -> None:
    """Print a standardized section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_save(path: str, kind: str = "file") -> None:
    """Print a standardized save message."""
    print(f"[SAVE] {kind}: {os.path.abspath(path)}")


def print_warn(msg: str) -> None:
    """Print a standardized warning message."""
    print(f"[WARN] {msg}")


def print_info(msg: str) -> None:
    """Print a standardized info message."""
    print(f"[INFO] {msg}")
