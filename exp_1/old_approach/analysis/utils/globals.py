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