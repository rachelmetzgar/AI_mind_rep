#!/usr/bin/env python3
"""
Experiment 4: Central Configuration

All paths, model names, and constants in one place.
Supports two model variants (chat, base) via set_model().

Usage:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import config, set_model, add_model_argument

    set_model("chat")
    data = config.data_dir("internals", "without_self")

Rachel C. Metzgar · Mar 2026
"""

import os
from pathlib import Path
from dataclasses import dataclass


# ============================================================================
# BASE PATHS
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent  # exp_4/
PROJECT_ROOT = ROOT_DIR.parent  # mind_rep/

VALID_MODELS = ("chat", "base")

MODELS = {
    "chat": {
        "path": (
            "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
            "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
            "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
        ),
        "label": "LLaMA-2-13B-Chat",
        "is_chat": True,
        "local_files_only": True,
    },
    "base": {
        "path": "meta-llama/Llama-2-13b-hf",
        "label": "LLaMA-2-13B (Base)",
        "is_chat": False,
        "local_files_only": False,
    },
}

INPUT_DIM = 5120   # LLaMA-2-13B hidden size
N_LAYERS = 41      # Embedding layer + 40 transformer layers


# ============================================================================
# ACTIVE MODEL STATE
# ============================================================================

_active_model = None


# ============================================================================
# CONFIG OBJECT
# ============================================================================

@dataclass
class Config:
    """Global configuration, updated by set_model()."""

    ROOT_DIR: Path = ROOT_DIR
    PROJECT_ROOT: Path = PROJECT_ROOT
    INPUT_DIM: int = INPUT_DIM
    N_LAYERS: int = N_LAYERS

    # Set by set_model()
    MODEL_KEY: str = None
    MODEL_PATH: str = None
    MODEL_LABEL: str = None
    IS_CHAT: bool = None
    LOCAL_FILES_ONLY: bool = None


config = Config()


# ============================================================================
# PATH HELPERS
# ============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist, return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device():
    """Get PyTorch device (cuda if available, else cpu)."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_condition_tag(include_self: bool) -> str:
    """Return condition tag string."""
    return "with_self" if include_self else "without_self"


def data_dir(phase, condition):
    """Return path to data directory: results/{phase}/{model}/{condition}/data/"""
    if _active_model is None:
        raise RuntimeError("Call set_model() before data_dir()")
    return ensure_dir(ROOT_DIR / "results" / phase / _active_model / condition / "data")


def figures_dir(phase, condition):
    """Return path to figures directory: results/{phase}/{model}/{condition}/figures/"""
    if _active_model is None:
        raise RuntimeError("Call set_model() before figures_dir()")
    return ensure_dir(ROOT_DIR / "results" / phase / _active_model / condition / "figures")


def results_phase_dir(phase, condition=None):
    """Return path to results/{phase}/{model}/ or results/{phase}/{model}/{condition}/"""
    if _active_model is None:
        raise RuntimeError("Call set_model() before results_phase_dir()")
    if condition:
        return ensure_dir(ROOT_DIR / "results" / phase / _active_model / condition)
    return ensure_dir(ROOT_DIR / "results" / phase / _active_model)


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def set_model(model):
    """
    Set the active model variant.

    Args:
        model: One of VALID_MODELS ("chat" or "base")
    """
    global _active_model

    if model not in VALID_MODELS:
        raise ValueError(f"Invalid model '{model}'. Must be one of: {VALID_MODELS}")

    _active_model = model
    info = MODELS[model]

    config.MODEL_KEY = model
    config.MODEL_PATH = info["path"]
    config.MODEL_LABEL = info["label"]
    config.IS_CHAT = info["is_chat"]
    config.LOCAL_FILES_ONLY = info["local_files_only"]


def get_active_model():
    """Return the currently active model key, or None if not set."""
    return _active_model


def add_model_argument(parser):
    """Add a required --model argument to an argparse parser."""
    parser.add_argument(
        "--model", type=str, required=True,
        choices=VALID_MODELS,
        help=f"Model variant to use. Choices: {', '.join(VALID_MODELS)}"
    )


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 4 CONFIGURATION")
    print("=" * 60)

    print(f"\nROOT DIR: {config.ROOT_DIR}")
    print(f"PROJECT ROOT: {config.PROJECT_ROOT}")
    print(f"Hidden dim: {config.INPUT_DIM}")
    print(f"Layers: {config.N_LAYERS}")

    for m in VALID_MODELS:
        set_model(m)
        print(f"\n--- {m} ---")
        print(f"  Model path: {config.MODEL_PATH}")
        print(f"  Label: {config.MODEL_LABEL}")
        print(f"  Is chat: {config.IS_CHAT}")
        print(f"  Local files only: {config.LOCAL_FILES_ONLY}")
        print(f"  data_dir('internals', 'without_self'): {data_dir('internals', 'without_self')}")
        print(f"  figures_dir('behavior', 'with_self'): {figures_dir('behavior', 'with_self')}")

    print(f"\nConfig loaded successfully!")
