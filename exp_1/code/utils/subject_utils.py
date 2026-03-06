"""
Subject ID validation and mapping utilities.

Author: Rachel C. Metzgar
"""

from __future__ import annotations
import re
from typing import Dict


_SUB_ID_RE = re.compile(r"^sub-\d{3}$")

_SUB_ID_MAP = {
    "P08": "sub-001", "s08": "sub-001",
    "P12": "sub-002", "s12": "sub-002",
    "P13": "sub-003", "s13": "sub-003",
    "P14": "sub-004", "s14": "sub-004",
    "P15": "sub-005", "s15": "sub-005",
    "P16": "sub-006", "s16": "sub-006",
    "P17": "sub-007", "s17": "sub-007",
    "P18": "sub-008", "s18": "sub-008",
    "P20": "sub-009", "s20": "sub-009",
    "P21": "sub-010", "s21": "sub-010",
    "P22": "sub-011", "s22": "sub-011",
    "P24": "sub-012", "s24": "sub-012",
    "P25": "sub-013", "s25": "sub-013",
    "P26": "sub-014", "s26": "sub-014",
    "P27": "sub-015", "s27": "sub-015",
    "P28": "sub-016", "s28": "sub-016",
    "P30": "sub-017", "s30": "sub-017",
    "P31": "sub-018", "s31": "sub-018",
    "P32": "sub-019", "s32": "sub-019",
    "P33": "sub-020", "s33": "sub-020",
    "P34": "sub-021", "s34": "sub-021",
    "P35": "sub-022", "s35": "sub-022",
    "P36": "sub-023", "s36": "sub-023",
}


def get_sub_id_map() -> Dict[str, str]:
    return _SUB_ID_MAP


def standardize_sub_id(raw_id: str) -> str:
    key = raw_id.strip()
    return _SUB_ID_MAP.get(key, key)


def validate_subject_id(sub_id: str) -> str:
    if not _SUB_ID_RE.match(sub_id):
        raise ValueError(f"Invalid subject ID format: {sub_id!r} (expected 'sub-###')")
    return sub_id


def find_old_id(new_id: str) -> str | None:
    old_ids = [k for k, v in _SUB_ID_MAP.items() if v == new_id]
    if not old_ids:
        return None
    for old in old_ids:
        if old.startswith("s"):
            return old
    return old_ids[0]
