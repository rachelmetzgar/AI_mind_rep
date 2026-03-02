"""
 Script name: cli_helpers.py
 Purpose: Standardized CLI/config parsing for all pipeline scripts.
 Author: Rachel C. Metzgar
 Date: 2025-10-29
"""

from __future__ import annotations
import argparse, json, os, logging
from typing import Callable, Dict, Any, Optional, Tuple

__all__ = [
    "std_cli",
    "parse_and_load_config",
    "load_global_config",
    "build_std_parser",
]

# --------------------------
# CLI & Config
# --------------------------

def build_std_parser(
    description: str,
    *,
    add_overwrite: bool = True,
    add_dry_run: bool = True,
    add_verbose: bool = True,
) -> "argparse.ArgumentParser":
    """Create a standard argparse parser used across pipeline scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="Path to the global JSON config.")
    if add_overwrite:
        p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    if add_dry_run:
        p.add_argument("--dry-run", action="store_true", help="Plan; do not write files.")
    if add_verbose:
        p.add_argument("--verbose", action="store_true", help="Debug logging.")
    return p


def std_cli(
    description: str,
    *,
    add_overwrite: bool = True,
    add_dry_run: bool = True,
    add_verbose: bool = True,
    extra_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> argparse.Namespace:
    """Build the standard parser, optionally inject extra flags, and parse args."""
    p = build_std_parser(
        description,
        add_overwrite=add_overwrite,
        add_dry_run=add_dry_run,
        add_verbose=add_verbose,
    )
    if extra_args is not None:
        extra_args(p)
    return p.parse_args()


def load_global_config(cfg_path: str) -> Dict[str, Any]:
    """Load the project's global JSON config."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_and_load_config(
    description: str,
    *,
    add_overwrite: bool = True,
    add_dry_run: bool = True,
    add_verbose: bool = True,
    extra_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> Tuple[argparse.Namespace, Dict[str, Any]]:
    """
    Parse CLI with the standard parser, then load the global JSON config.
    This helper is used by all analysis scripts.
    """
    args = std_cli(
        description,
        add_overwrite=add_overwrite,
        add_dry_run=add_dry_run,
        add_verbose=add_verbose,
        extra_args=extra_args,
    )
    cfg = load_global_config(args.config)
    return args, cfg
