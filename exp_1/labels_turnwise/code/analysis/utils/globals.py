"""
Script: globals.py

Purpose: Project-wide constants and shared configuration for the AI Perception project.

Usage:
    from utils.globals import (
        DATA_DIR, RESULTS_DIR, LOGS_DIR, ...,
    )

Author: Rachel C. Metzgar
Date: 2025-10-29
"""

from __future__ import annotations
import os

# --- Robustly determine repo root ---
# 1) If set, let the env drive everything (for jobs/scratch/other machines)
_env_root = os.environ.get("PROJECT_ROOT", "").strip()

# 2) Otherwise, derive from this file: .../ai_percep_clean/code/utils/globals.py
#    -> repo root is two levels up from 'code/utils'
_here = os.path.abspath(os.path.dirname(__file__))
_code_dir = os.path.abspath(os.path.join(_here, ".."))          # .../ai_percep_clean/code
_derived_root = os.path.abspath(os.path.join(_code_dir, ".."))  # .../ai_percep_clean

# 3) Final fallback: your historic absolute path on this cluster
_fallback_root = "/jukebox/graziano/rachel/ai_mind_rep/exp_1"

PROJECT_ROOT = _env_root or _derived_root or _fallback_root

# ==== Global Project Paths ====
DATA_DIR    = f"{PROJECT_ROOT}/data"
RESULTS_DIR = f"{PROJECT_ROOT}/results"
LOGS_DIR    = f"{PROJECT_ROOT}/logs"
ENV_FILE    = f"{PROJECT_ROOT}/environment.yml"

# Optional: expose for logging/sanity
REPO_ROOT = PROJECT_ROOT
CODE_DIR  = f"{PROJECT_ROOT}/code"
HUMAN_DIR = "data/exp_csv_human/"  
TOPICS_PATH = "data/conds/topics.csv"
HUMAN_FILE_PATTERN = "{sub_id}.csv"

def get_sub_id_map():
    """Legacy behavioral ID -> standardized MRI ID mapping."""
    return {
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