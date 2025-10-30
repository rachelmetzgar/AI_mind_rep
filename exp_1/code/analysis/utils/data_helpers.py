"""
 Script name: data_helpers.py
 Purpose: Lightweight filesystem and parsing utilities for CSV/NPZ IO and stem normalization.
 Author: Rachel C. Metzgar
 Date: 2025-08-29
"""

from __future__ import annotations
import os, json, re, logging
from typing import Iterable, Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from utils.file_parsing import bids_run_label
from utils.globals import EXP_CSV_DIR, get_sub_id_map

__all__ = [
    "list_csvs", "pick_col", "pick_token_column",
    "normalize_output_stem", "infer_run_label_from_stem",
    "safe_json", "safe_load_tsv", "safe_json_loads", "load_behavior_csv", 
    "load_npz_embeddings", "load_tr_npz", "resolve_stem_and_token2tr", 
    "get_file_path",
]

# --------------------------
# Data helpers
# --------------------------


def list_csvs(dir_path: str) -> List[str]:
    """List CSV files in a directory (sorted by name)."""
    if not os.path.isdir(dir_path):
        return []
    return [
        os.path.join(dir_path, n)
        for n in sorted(os.listdir(dir_path))
        if n.endswith(".csv")
    ]


def pick_col(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """Return the first matching column (case-insensitive), preserving original case."""
    norm_to_orig: dict = {}
    for c in cols:
        k = c.strip().lower()
        if k not in norm_to_orig:
            norm_to_orig[k] = c
    for cand in candidates:
        key = cand.strip().lower()
        if key in norm_to_orig:
            return norm_to_orig[key]
    return None


def pick_token_column(cols: Iterable[str]) -> Optional[str]:
    """Pick a token column name if available."""
    lower = {c.lower(): c for c in cols}
    for cand in ("word", "token", "text"):
        if cand in lower:
            return lower[cand]
    return None

def normalize_output_stem(stem: str, sub_id: str) -> str:
    """Normalize a CSV stem for output naming."""
    s = stem
    if s.endswith("_timing"):
        s = s[: -len("_timing")]

    def _repl(match: "re.Match") -> str:
        rnum = int(match.group(1))
        return bids_run_label(rnum)

    s = re.sub(r"run[-_]?0*(\d+)", _repl, s, count=1)
    s = s.replace("task-None_", "").replace("_task-None", "").replace("task-None", "")
    if not s.startswith(sub_id + "_"):
        s = f"{sub_id}_{s}"
    return s


def infer_run_label_from_stem(stem: str) -> Optional[str]:
    """Infer a BIDS run label from a normalized stem."""
    m = re.search(r"(run-\d{2})", stem)
    return m.group(1) if m else None

def safe_json(obj: Any) -> str:
    """Return a JSON string; fall back to an empty object on failure."""
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return "{}"

def safe_json_loads(s: Any) -> Dict[str, Any]:
    """Parse a JSON-like object safely."""
    try:
        if isinstance(s, bytes):
            s = s.decode("utf-8", "ignore")
        if isinstance(s, str):
            return json.loads(s)
    except Exception:
        pass
    return {}

def safe_load_tsv(path: str) -> pd.DataFrame:
    """Load TSV/CSV with graceful fallback."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)

def load_behavior_csv(old_id: str, logger=None) -> pd.DataFrame:
    """Load behavioral CSV for a given old subject ID and log status."""
    behav_path = os.path.join(EXP_CSV_DIR, f"{old_id}.csv")
    if not os.path.exists(behav_path):
        raise FileNotFoundError(f"Behavioral file not found: {behav_path}")
    df = pd.read_csv(behav_path)
    if logger:
        logger.info("Loaded %d behavioral rows for %s", len(df), old_id)
    return df


def load_npz_embeddings(npz_path: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Load embeddings and meta from an NPZ file."""
    with np.load(npz_path, allow_pickle=True) as npz:
        if "embeddings" not in npz:
            raise ValueError(f"Missing 'embeddings' in NPZ: {npz_path}")
        emb = npz["embeddings"]
        if emb.ndim != 2:
            raise ValueError(f"'embeddings' must be 2D; got shape {emb.shape} in {npz_path}")
        dim = int(emb.shape[1])

        meta: Dict[str, Any] = {}
        # Prefer JSON string under 'meta_json'; fallback to object under 'meta'
        if "meta_json" in npz:
            try:
                meta = json.loads(str(npz["meta_json"].item()))
            except Exception:
                meta = {}
        elif "meta" in npz:
            try:
                m = npz["meta"].item()
                if isinstance(m, (bytes, str)):
                    meta = json.loads(m if isinstance(m, str) else m.decode("utf-8", "ignore"))
                elif isinstance(m, dict):
                    meta = m
                else:
                    meta = {}
            except Exception:
                meta = {}
    return emb, dim, meta

def load_tr_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    """Load TR array and embeddings_TR from a NPZ created by make_TR_embeddings.py."""
    with np.load(npz_path, allow_pickle=True) as npz:
        if "TR" not in npz or "embeddings_TR" not in npz:
            raise ValueError(f"Missing TR/embeddings_TR in: {npz_path}")
        TR = np.asarray(npz["TR"], dtype=np.int64)
        E = np.asarray(npz["embeddings_TR"], dtype=np.float32)
        if TR.ndim != 1 or E.ndim != 2 or E.shape[0] != TR.shape[0]:
            raise ValueError(f"Shape mismatch TR {TR.shape} vs embeddings {E.shape} in {npz_path}")
        dim = int(npz["dim"]) if "dim" in npz else int(E.shape[1])
        meta = {}
        if "meta_json" in npz:
            meta = safe_json_loads(npz["meta_json"].item())
    return TR, E, dim, meta


def resolve_stem_and_token2tr(
    npz_path: str,
    sub_id: str,
    tr_dir_sub: str,
    *,
    meta: Optional[Dict[str, Any]],
    logger: logging.Logger,
) -> Tuple[str, str]:
    """Resolve normalized stem and corresponding token→TR CSV path."""
    # 1) Preferred: derive from 'source_csv' in meta (robust to renames)
    stem_norm: Optional[str] = None
    if meta:
        src = meta.get("source_csv") or meta.get("source_csv_rel") or meta.get("source")
        if isinstance(src, str) and src:
            base = os.path.basename(src)
            stem_norm = normalize_output_stem(os.path.splitext(base)[0], sub_id)

    # 2) Fallback: use NPZ basename
    if not stem_norm:
        base = os.path.basename(npz_path)
        # e.g., 'sub-001_run-01_xxx_embeddings.npz' → 'sub-001_run-01_xxx'
        if base.endswith("_embeddings.npz"):
            base = base[: -len("_embeddings.npz")]
        elif base.endswith(".npz"):
            base = base[: -len(".npz")]
        stem_norm = normalize_output_stem(base, sub_id)

    cand = os.path.join(tr_dir_sub, f"{stem_norm}_token2tr.csv")
    if os.path.exists(cand):
        return stem_norm, cand

    # 3) Last resort: scan for a unique approximate match (same run label)
    run_lbl = infer_run_label_from_stem(stem_norm) or ""
    best: Optional[str] = None
    if os.path.isdir(tr_dir_sub):
        for name in sorted(os.listdir(tr_dir_sub)):
            if not name.endswith("_token2tr.csv"):
                continue
            if name.startswith(stem_norm + "_"):
                best = os.path.join(tr_dir_sub, name)
                break
            if run_lbl and run_lbl in name:
                best = os.path.join(tr_dir_sub, name)
                # do not break; prefer first more specific match
                # but keep a candidate
    if best:
        logger.warning(
            "Token→TR CSV name did not exactly match; using approximate: %s", os.path.basename(best)
        )
        return stem_norm, best

    raise FileNotFoundError(
        f"No token→TR CSV found for NPZ stem '{stem_norm}' in {tr_dir_sub}"
    )

def get_file_path(sub_id: str) -> str | None:
    """Return path to subject CSV, checking both sub-### and legacy IDs."""
    alias_map = get_sub_id_map()
    candidates = [sub_id] + [k for k, v in alias_map.items() if v == sub_id]
    for candidate in candidates:
        path = os.path.join(EXP_CSV_DIR, f"{candidate}.csv")
        if os.path.exists(path):
            return path
    return None