"""
CSV/file I/O utilities.

Author: Rachel C. Metzgar
"""

from __future__ import annotations
import os
import json
from typing import List, Optional, Dict, Any
import pandas as pd


def list_csvs(dir_path: str) -> List[str]:
    """List CSV files in a directory (sorted by name)."""
    if not os.path.isdir(dir_path):
        return []
    return [
        os.path.join(dir_path, n)
        for n in sorted(os.listdir(dir_path))
        if n.endswith(".csv")
    ]


def pick_col(cols, candidates) -> Optional[str]:
    """Return the first matching column (case-insensitive)."""
    norm_to_orig = {}
    for c in cols:
        k = c.strip().lower()
        if k not in norm_to_orig:
            norm_to_orig[k] = c
    for cand in candidates:
        key = cand.strip().lower()
        if key in norm_to_orig:
            return norm_to_orig[key]
    return None


def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return "{}"


def safe_load_tsv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)
