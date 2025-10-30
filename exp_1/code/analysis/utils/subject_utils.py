"""
 Script name: subject_utils.py
 Purpose:Standard utilities for subject ID validation and mapping between legacy behavioral IDs (e.g., 'P08', 's08') and  standardized MRI IDs ('sub-###'). 
 Author: Rachel C. Metzgar
 Date: 2025-08-04
"""

from __future__ import annotations

import re
from typing import Dict

from utils.globals import get_sub_id_map

__all__ = ["standardize_sub_id", "validate_subject_id", "find_old_id"]

# Strict BIDS-style subject pattern: 'sub-' + three digits.
_SUB_ID_RE = re.compile(r"^sub-\d{3}$")


def standardize_sub_id(raw_id: str) -> str:
    """Map a behavioral/legacy ID to the standardized MRI ID. """
    # Note: do not alter case or pad here to avoid changing mapping semantics.
    # Trimming is safe for typical CSV whitespace artifacts.
    key = raw_id.strip()
    mapping: Dict[str, str] = get_sub_id_map()
    return mapping.get(key, key)


def validate_subject_id(sub_id: str) -> str:
    """Ensure a subject ID follows the 'sub-###' format."""
    if not _SUB_ID_RE.match(sub_id):
        raise ValueError(f"Invalid subject ID format: {sub_id!r} (expected 'sub-###')")
    return sub_id

def find_old_id(new_id: str) -> str | None:
    """Return the old subject ID (e.g. s36, P36) given sub-###."""
    id_map = get_sub_id_map()
    old_ids = [k for k, v in id_map.items() if v == new_id]
    if not old_ids:
        return None
    for old in old_ids:
        if old.startswith("s"):
            return old
    return old_ids[0]
