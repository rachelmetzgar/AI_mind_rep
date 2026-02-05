"""
 Script name: run_logger.py
 Purpose:
    Unified logging utilities for the AI Perception project.
    - Creates a per-run logfile (and mirrors to console/Slurm).
    - Logs environment & library versions once per run.
    - Optionally snapshots the config JSON next to the log.
 Author: Rachel C. Metzgar
 Date: 2025-08-14
"""
from __future__ import annotations
from typing import Optional, Any, Dict, Tuple
import logging
import os
from datetime import datetime
import shutil
import platform
import sys, traceback
import random as _random

__all__ = ["setup_logger", "log_env_specs", "log_config_copy", "install_excepthook"]

_DEF_FMT = "%(asctime)s - %(levelname)s - %(process)d - %(message)s"  # includes PID (nice on Slurm)


def _get_logger(name: str, log_path: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or reuse) a named logger without duplicating handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # don't double-print via root

    # File handler (only add once)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "_ai_log_path", "") == log_path
               for h in logger.handlers):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh._ai_log_path = log_path  # sentinel to prevent dupes
        fh.setFormatter(logging.Formatter(_DEF_FMT))
        logger.addHandler(fh)

    # Console handler (only add once)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(_DEF_FMT))
        logger.addHandler(ch)

    return logger


def setup_logger(output_dir: str, script_name: str = "run", level: int = logging.INFO) -> logging.Logger:
    """Initialize a logger that writes to:"""
    log_path = os.path.join(output_dir, f"{script_name}_runlog.txt")
    logger = _get_logger(script_name, log_path, level=level)
    logger.info(f"===== Started {script_name}.py =====")
    return logger


def log_env_specs(logger: logging.Logger, extras: Optional[Dict[str, Any]] = None) -> None:
    """Log environment/system/library info once per run, plus any script-specific params."""
    logger.info("---- Runtime Environment ----")
    logger.info(f"Datetime (UTC/local): {datetime.utcnow().isoformat()}Z / {datetime.now().isoformat()}")
    logger.info(f"Hostname: {platform.node()}")
    logger.info(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    logger.info(f"Python: {platform.python_version()}")

    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env:
        logger.info(f"Conda env: {conda_env}")

    def _safe_ver(modname: str) -> str:
        try:
            mod = __import__(modname)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "not-installed"

    logger.info(f"numpy: {_safe_ver('numpy')}, pandas: {_safe_ver('pandas')}, torch: {_safe_ver('torch')}")
    logger.info(f"tqdm: {_safe_ver('tqdm')}, transformers: {_safe_ver('transformers')}, whisper: {_safe_ver('whisper')}")

    for v in ["SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_NODELIST"]:
        if v in os.environ:
            logger.info(f"{v}: {os.environ[v]}")

    if extras:
        logger.info("---- Run Parameters ----")
        for k, v in extras.items():
            logger.info(f"{k}: {v}")


def log_config_copy(
    config_path: str,
    output_dir: str,
    logger: "logging.Logger" = None,
    *,  # keyword-only from here down
    config_obj: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a copy of the run config into output_dir as 'config_used.json'."""
    import json, os, shutil

    os.makedirs(output_dir, exist_ok=True)
    dst = os.path.join(output_dir, "config_used.json")

    if config_obj is not None:
        # Try writing the in-memory config (best effort JSON)
        try:
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(config_obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        except TypeError:
            # Fallback if config_obj has non-JSON types: copy the source file if available,
            # else write a minimal stub (don't crash the run over this).
            if config_path and os.path.isfile(config_path):
                shutil.copy2(config_path, dst)
            else:
                with open(dst, "w", encoding="utf-8") as f:
                    f.write("{}\n")
        if logger:
            logger.info(f"Saved config JSON to: {dst}")
        return

    # Old behavior: copy the source file and raise on missing
    if not config_path:
        return
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    shutil.copy2(config_path, dst)
    if logger:
        logger.info(f"Saved config copy to: {dst}")


def install_excepthook(logger: Optional[logging.Logger] = None) -> None:
    """Install a global exception hook that logs uncaught exceptions. """
    def _hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        target = logger or logging.getLogger()
        target.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

    sys.excepthook = _hook


def init_run(
    *,
    output_dir: str,
    script_name: str,
    args,                    # argparse.Namespace (must have .config; may have .verbose/.overwrite/.dry_run)
    cfg: Dict[str, Any],
    used_alias: bool = False,
) -> Tuple[logging.Logger, int, bool, bool]:
    """
    Initialize a run:
      - create logger and write env specs
      - snapshot config JSON next to the log
      - emit CLI deprecation warnings if present
      - set seeds (random, numpy, torch if available)
      - extract overwrite/dry_run flags
    """
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logger = setup_logger(output_dir, script_name=script_name, level=level)
    install_excepthook(logger)
    log_env_specs(logger)
    log_config_copy(getattr(args, "config", ""), output_dir, logger=logger, config_obj=cfg)

    # CLI deprecation notes (centralized)
    if used_alias:
        logger.warning("CLI: Received deprecated --subjects alias; prefer --sub-id.")
    if getattr(args, "_legacy_project_root_used", False):
        logger.warning("CLI: --project-root is deprecated; prefer PROJECT_ROOT env var.")

    # Seeds
    seed = int(cfg.get("random_seed", 0))
    _random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            try:
                _torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
    except Exception:
        pass

    overwrite = bool(getattr(args, "overwrite", False))
    dry_run   = bool(getattr(args, "dry_run", False))
    return logger, seed, overwrite, dry_run

