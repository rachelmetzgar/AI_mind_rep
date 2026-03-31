#!/usr/bin/env python3
"""
Experiment 4: Central Configuration

All paths, model names, and constants in one place.
Supports multiple model variants via set_model().

Results are organized by branch (experimental paradigm), then modality:
    results/{model}/{branch}/{modality}/{condition}/

Branches:
    gray_replication         — 13 Gray entities, pairwise comparisons
    gray_simple              — 13 Gray entities, "Think about {entity}"
    human_ai_adaptation      — 30 AI/human characters, Gray capacities
    expanded_mental_concepts — 28 AI/human characters, Exp 3 concept dims

Usage:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import config, set_model, add_model_argument

    set_model("llama2_13b_chat")
    ddir = data_dir("gray_simple", "internals", "without_self")

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

VALID_MODELS = (
    "llama2_13b_chat", "llama2_13b_base",
    "llama3_8b_instruct", "llama3_8b_base",
    "gemma2_2b_it", "gemma2_2b",
    "gemma2_9b_it", "gemma2_9b",
    "qwen25_7b_instruct", "qwen25_7b",
    "qwen3_8b",
)

VALID_BRANCHES = (
    "gray_replication",
    "gray_simple",
    "human_ai_adaptation",
    "expanded_mental_concepts",
)

_HF_CACHE = "/mnt/cup/labs/graziano/rachel/.cache_huggingface/hub"

MODELS = {
    "llama2_13b_chat": {
        "path": (
            "/mnt/cup/labs/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
            "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
            "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
        ),
        "label": "LLaMA-2-13B-Chat",
        "family": "llama2",
        "is_chat": True,
        "local_files_only": True,
        "hidden_dim": 5120,
        "n_transformer_layers": 40,
    },
    "llama2_13b_base": {
        "path": "meta-llama/Llama-2-13b-hf",
        "label": "LLaMA-2-13B (Base)",
        "family": "llama2",
        "is_chat": False,
        "local_files_only": False,
        "hidden_dim": 5120,
        "n_transformer_layers": 40,
    },
    "llama3_8b_instruct": {
        "path": (
            f"{_HF_CACHE}/models--meta-llama--Meta-Llama-3-8B-Instruct/"
            "snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
        ),
        "label": "LLaMA-3-8B-Instruct",
        "family": "llama3",
        "is_chat": True,
        "local_files_only": True,
        "hidden_dim": 4096,
        "n_transformer_layers": 32,
    },
    "llama3_8b_base": {
        "path": (
            f"{_HF_CACHE}/models--meta-llama--Meta-Llama-3-8B/"
            "snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
        ),
        "label": "LLaMA-3-8B (Base)",
        "family": "llama3",
        "is_chat": False,
        "local_files_only": True,
        "hidden_dim": 4096,
        "n_transformer_layers": 32,
    },
    "gemma2_2b_it": {
        "path": (
            f"{_HF_CACHE}/models--google--gemma-2-2b-it/"
            "snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
        ),
        "label": "Gemma-2-2B-IT",
        "family": "gemma2",
        "is_chat": True,
        "local_files_only": True,
        "hidden_dim": 2304,
        "n_transformer_layers": 26,
    },
    "gemma2_2b": {
        "path": (
            f"{_HF_CACHE}/models--google--gemma-2-2b/"
            "snapshots/c5ebcd40d208330abc697524c919956e692655cf"
        ),
        "label": "Gemma-2-2B (Base)",
        "family": "gemma2",
        "is_chat": False,
        "local_files_only": True,
        "hidden_dim": 2304,
        "n_transformer_layers": 26,
    },
    "gemma2_9b_it": {
        "path": (
            f"{_HF_CACHE}/models--google--gemma-2-9b-it/"
            "snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
        ),
        "label": "Gemma-2-9B-IT",
        "family": "gemma2",
        "is_chat": True,
        "local_files_only": True,
        "hidden_dim": 3584,
        "n_transformer_layers": 42,
    },
    "gemma2_9b": {
        "path": (
            f"{_HF_CACHE}/models--google--gemma-2-9b/"
            "snapshots/33c193028431c2fde6c6e51f29e6f17b60cbfac6"
        ),
        "label": "Gemma-2-9B (Base)",
        "family": "gemma2",
        "is_chat": False,
        "local_files_only": True,
        "hidden_dim": 3584,
        "n_transformer_layers": 42,
    },
    "qwen25_7b_instruct": {
        "path": (
            f"{_HF_CACHE}/models--Qwen--Qwen2.5-7B-Instruct/"
            "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
        ),
        "label": "Qwen-2.5-7B-Instruct",
        "family": "qwen2",
        "is_chat": True,
        "local_files_only": True,
        "hidden_dim": 3584,
        "n_transformer_layers": 28,
    },
    "qwen25_7b": {
        "path": (
            f"{_HF_CACHE}/models--Qwen--Qwen2.5-7B/"
            "snapshots/d149729398750b98c0af14eb82c78cfe92750796"
        ),
        "label": "Qwen-2.5-7B (Base)",
        "family": "qwen2",
        "is_chat": False,
        "local_files_only": True,
        "hidden_dim": 3584,
        "n_transformer_layers": 28,
    },
    "qwen3_8b": {
        "path": (
            f"{_HF_CACHE}/models--Qwen--Qwen3-8B/"
            "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
        ),
        "label": "Qwen3-8B",
        "family": "qwen3",
        "is_chat": True,
        "local_files_only": True,
        "hidden_dim": 4096,
        "n_transformer_layers": 36,
    },
}

# Defaults — updated by set_model() to match the active model
INPUT_DIM = 5120   # LLaMA-2-13B hidden size (overwritten by set_model)
N_LAYERS = 41      # Embedding + transformer layers (overwritten by set_model)


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
    MODEL_FAMILY: str = None
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


def _build_path(*parts):
    """Build a results path from parts, filtering out None."""
    if _active_model is None:
        raise RuntimeError("Call set_model() before using path helpers")
    base = ROOT_DIR / "results" / _active_model
    for p in parts:
        if p is not None:
            base = base / p
    return base


def data_dir(branch, modality, condition=None):
    """Return path: results/{model}/{branch}/{modality}/{condition}/data/"""
    return ensure_dir(_build_path(branch, modality, condition, "data"))


def figures_dir(branch, modality, condition=None):
    """Return path: results/{model}/{branch}/{modality}/{condition}/figures/"""
    return ensure_dir(_build_path(branch, modality, condition, "figures"))


def results_dir(branch, modality=None, condition=None):
    """Return path: results/{model}/{branch}/{modality}/{condition}/"""
    return ensure_dir(_build_path(branch, modality, condition))


COMPARISONS_DIR = ROOT_DIR / "results" / "comparisons"


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def set_model(model):
    """
    Set the active model variant.

    Args:
        model: One of VALID_MODELS
    """
    global _active_model, INPUT_DIM, N_LAYERS

    if model not in VALID_MODELS:
        raise ValueError(f"Invalid model '{model}'. Must be one of: {VALID_MODELS}")

    _active_model = model
    info = MODELS[model]

    config.MODEL_KEY = model
    config.MODEL_PATH = info["path"]
    config.MODEL_LABEL = info["label"]
    config.MODEL_FAMILY = info["family"]
    config.IS_CHAT = info["is_chat"]
    config.LOCAL_FILES_ONLY = info["local_files_only"]

    # Update model-specific architecture constants
    config.INPUT_DIM = info["hidden_dim"]
    config.N_LAYERS = info["n_transformer_layers"] + 1  # +1 for embedding layer
    INPUT_DIM = config.INPUT_DIM
    N_LAYERS = config.N_LAYERS


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
    print(f"COMPARISONS_DIR: {COMPARISONS_DIR}")

    for m in VALID_MODELS:
        set_model(m)
        print(f"\n--- {m} ---")
        print(f"  Model path: {config.MODEL_PATH}")
        print(f"  Label: {config.MODEL_LABEL}")
        print(f"  Family: {config.MODEL_FAMILY}")
        print(f"  Is chat: {config.IS_CHAT}")
        print(f"  Local files only: {config.LOCAL_FILES_ONLY}")
        print(f"  Hidden dim: {config.INPUT_DIM}, Layers: {config.N_LAYERS}")
        print(f"  data_dir('gray_simple', 'internals', 'without_self'):")
        print(f"    {data_dir('gray_simple', 'internals', 'without_self')}")
        print(f"  figures_dir('gray_replication', 'behavior', 'with_self'):")
        print(f"    {figures_dir('gray_replication', 'behavior', 'with_self')}")
        print(f"  results_dir('expanded_mental_concepts', 'internals', 'rsa'):")
        print(f"    {results_dir('expanded_mental_concepts', 'internals', 'rsa')}")

    print(f"\nConfig loaded successfully!")
