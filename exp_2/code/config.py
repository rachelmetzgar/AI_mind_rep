#!/usr/bin/env python3
"""
Experiment 2: Central Configuration

All paths, model names, and hyperparameters in one place.
Update this file instead of editing individual scripts.

Usage:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
    from config import config, set_version, add_version_argument

    # Set data version (required before accessing version-dependent paths)
    set_version("labels")
    csv_dir = config.PATHS.csv_dir
    probe_dir = config.PATHS.probe_checkpoints

    # Access hyperparameters
    epochs = config.TRAINING.epochs
    batch_size = config.TRAINING.batch_size_train

Rachel C. Metzgar · Feb 2026
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass


# ============================================================================
# BASE PATHS
# ============================================================================

# Root directory for this experiment (exp_2/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Root for entire project (mind_rep/)
PROJECT_ROOT = ROOT_DIR.parent

# Valid data versions
VALID_VERSIONS = (
    "labels", "balanced_names", "balanced_gpt", "names",
    "nonsense_codeword", "nonsense_ignore",
    "labels_turnwise", "you_are_labels", "you_are_labels_turnwise",
    "you_are_balanced_gpt",
)

# Internal tracking of active version
_active_version = None


# ============================================================================
# MODEL
# ============================================================================

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

INPUT_DIM = 5120   # LLaMA-2-13B hidden size
N_LAYERS = 41      # Embedding layer + 40 transformer layers


# ============================================================================
# CSV_DIR MAPPING: version → path to per-subject CSV files
# ============================================================================

_CSV_DIR_MAP = {
    "labels": (
        "/jukebox/graziano/rachel/mind_rep/exp_1/versions/labels/"
        "data/meta-llama-Llama-2-13b-chat-hf/0.8"
    ),
    "balanced_names": (
        "/jukebox/graziano/rachel/mind_rep/exp_1/versions/balanced_names/"
        "data/meta-llama-Llama-2-13b-chat-hf/0.8"
    ),
    "balanced_gpt": (
        "/jukebox/graziano/rachel/mind_rep/exp_1/versions/balanced_gpt/"
        "data/meta-llama-Llama-2-13b-chat-hf/0.8"
    ),
    "names": (
        "/jukebox/graziano/rachel/mind_rep/exp_2/archive/names/"
        "llama_exp_2b-13B-chat/data/meta-llama-Llama-2-13b-chat-hf/0.8"
    ),
    "nonsense_codeword": (
        "/jukebox/graziano/rachel/mind_rep/exp_1/versions/nonsense_codeword/"
        "data/meta-llama-Llama-2-13b-chat-hf/0.8"
    ),
    "nonsense_ignore": (
        "/jukebox/graziano/rachel/mind_rep/exp_1/versions/nonsense_ignore/"
        "data/meta-llama-Llama-2-13b-chat-hf/0.8"
    ),
    "labels_turnwise": (
        "/jukebox/graziano/rachel/mind_rep/exp_1/versions/labels_turnwise/"
        "data/meta-llama-Llama-2-13b-chat-hf/0.8"
    ),
    "you_are_labels": (
        "/jukebox/graziano/rachel/mind_rep/exp_1/versions/you_are_labels/"
        "data/meta-llama-Llama-2-13b-chat-hf/0.8"
    ),
    "you_are_labels_turnwise": (
        "/jukebox/graziano/rachel/mind_rep/exp_1/versions/you_are_labels_turnwise/"
        "data/meta-llama-Llama-2-13b-chat-hf/0.8"
    ),
    "you_are_balanced_gpt": (
        "/jukebox/graziano/rachel/mind_rep/exp_1/versions/you_are_balanced_gpt/"
        "data/meta-llama-Llama-2-13b-chat-hf/0.8"
    ),
}

# EXP1 root mapping: version → exp_1 data_gen root (for V2 causality prompts/configs)
_EXP1_DATA_GEN_MAP = {
    "labels":            PROJECT_ROOT / "exp_1" / "versions" / "labels" / "code" / "data_gen",
    "balanced_names":    PROJECT_ROOT / "exp_1" / "versions" / "balanced_names" / "code" / "data_gen",
    "balanced_gpt":      PROJECT_ROOT / "exp_1" / "versions" / "balanced_gpt" / "code" / "data_gen",
    "names":             PROJECT_ROOT / "exp_1" / "versions" / "names" / "code" / "data_gen",
    "nonsense_codeword": PROJECT_ROOT / "exp_1" / "versions" / "nonsense_codeword" / "code" / "data_gen",
    "nonsense_ignore":        PROJECT_ROOT / "exp_1" / "versions" / "nonsense_ignore" / "code" / "data_gen",
    "labels_turnwise":        PROJECT_ROOT / "exp_1" / "versions" / "labels_turnwise" / "code" / "data_gen",
    "you_are_labels":         PROJECT_ROOT / "exp_1" / "versions" / "you_are_labels" / "code" / "data_gen",
    "you_are_labels_turnwise": PROJECT_ROOT / "exp_1" / "versions" / "you_are_labels_turnwise" / "code" / "data_gen",
    "you_are_balanced_gpt":   PROJECT_ROOT / "exp_1" / "versions" / "you_are_balanced_gpt" / "code" / "data_gen",
}

# EXP1 analysis utils mapping (for behavioral analysis feature imports)
_EXP1_UTILS_MAP = {
    "labels":                  PROJECT_ROOT / "exp_1" / "versions" / "labels" / "code" / "analysis" / "utils",
    "balanced_names":          PROJECT_ROOT / "exp_1" / "versions" / "balanced_names" / "code" / "analysis" / "utils",
    "balanced_gpt":            PROJECT_ROOT / "exp_1" / "versions" / "balanced_gpt" / "code" / "analysis" / "utils",
    "names":                   PROJECT_ROOT / "exp_1" / "versions" / "labels" / "code" / "analysis" / "utils",
    "nonsense_codeword":       PROJECT_ROOT / "exp_1" / "versions" / "nonsense_codeword" / "code" / "analysis" / "utils",
    "nonsense_ignore":         PROJECT_ROOT / "exp_1" / "versions" / "nonsense_ignore" / "code" / "analysis" / "utils",
    "labels_turnwise":         PROJECT_ROOT / "exp_1" / "versions" / "labels_turnwise" / "code" / "analysis" / "utils",
    "you_are_labels":          PROJECT_ROOT / "exp_1" / "versions" / "you_are_labels" / "code" / "analysis" / "utils",
    "you_are_labels_turnwise": PROJECT_ROOT / "exp_1" / "versions" / "you_are_labels_turnwise" / "code" / "analysis" / "utils",
    "you_are_balanced_gpt":    PROJECT_ROOT / "exp_1" / "versions" / "you_are_balanced_gpt" / "code" / "analysis" / "utils",
}


# ============================================================================
# PATHS: Inputs (data/)
# ============================================================================

@dataclass
class InputPaths:
    """Paths to input data. Version-dependent paths are None until set_version()."""

    # Per-subject CSV directory (set by set_version)
    csv_dir: Path = None

    # Probe checkpoints: data/{version}/probe_checkpoints/
    probe_checkpoints: Path = None

    # Intervention results: data/{version}/intervention_results/
    intervention_results: Path = None

    # Exp 1 data_gen root (for V2 causality prompts/configs)
    exp1_data_gen: Path = None
    exp1_prompts: Path = None
    exp1_configs: Path = None

    # Exp 1 analysis utils (for behavioral markers)
    exp1_utils: Path = None

    # Shared data (version-independent)
    topics_csv: Path = ROOT_DIR / "data" / "shared" / "conds" / "topics.csv"
    causality_questions: Path = ROOT_DIR / "data" / "shared" / "causality_test_questions" / "human_ai.txt"


# ============================================================================
# PATHS: Outputs (results/)
# ============================================================================

@dataclass
class OutputPaths:
    """Paths to output results. Version-dependent paths are None until set_version()."""

    # Root results directory
    root: Path = ROOT_DIR / "results"

    # Version-specific results (set by set_version)
    version_root: Path = None      # results/versions/{version}/
    probe_training: Path = None    # results/versions/{version}/probe_training/
    degradation: Path = None       # results/versions/{version}/degradation_analysis/

    # Cross-variant comparison results
    comparisons: Path = ROOT_DIR / "results" / "comparisons"
    probe_training_comparison: Path = ROOT_DIR / "results" / "comparisons" / "probe_training"
    causality_qc_comparison: Path = ROOT_DIR / "results" / "comparisons" / "causality_qc"

    # Logs
    logs: Path = ROOT_DIR / "logs"
    version_logs: Path = None      # logs/{version}/


# ============================================================================
# HYPERPARAMETERS: Training
# ============================================================================

@dataclass
class TrainingConfig:
    """Hyperparameters for probe training."""

    epochs: int = 50
    batch_size_train: int = 200
    batch_size_test: int = 400
    logistic: bool = True
    uncertainty: bool = False
    one_hot: bool = False
    seed: int = 12345


# ============================================================================
# HYPERPARAMETERS: Intervention
# ============================================================================

@dataclass
class InterventionConfig:
    """Hyperparameters for causal intervention."""

    default_strengths: list = None  # [2, 4, 8, 16]
    min_probe_accuracy: float = 0.55

    # V1 single-turn generation
    v1_max_new_tokens: int = 768
    v1_temperature: float = 0.7
    v1_top_p: float = 0.9
    v1_do_sample: bool = True

    # V2 multi-turn generation
    v2_max_new_tokens: int = 500
    v2_temperature: float = 0.8
    v2_top_p: float = 1.0
    v2_pairs_total: int = 4
    v2_history_pairs: int = 4
    v2_system_prompt: str = "You are engaging in a real-time spoken conversation."
    v2_conditions: list = None

    def __post_init__(self):
        if self.default_strengths is None:
            self.default_strengths = [2, 4, 8, 16]
        if self.v2_conditions is None:
            self.v2_conditions = ["baseline", "human", "ai"]


# ============================================================================
# LAYER STRATEGIES (for causal intervention)
# ============================================================================

LAYER_STRATEGIES = {
    "narrow": {
        "min_accuracy": 0.55,
        "window": None,
        "window_size": 10,
        "top_k": None,
        "description": "Contiguous 10-layer window (auto-selected), matching Viegas et al.",
    },
    "wide": {
        "min_accuracy": 0.55,
        "window": None,
        "window_size": None,
        "top_k": None,
        "description": "All layers with probe accuracy >= threshold",
    },
    "peak_15": {
        "min_accuracy": 0.55,
        "window": None,
        "window_size": None,
        "top_k": 15,
        "description": "Top 15 layers by probe accuracy (non-contiguous)",
    },
    "all_70": {
        "min_accuracy": 0.55,
        "window": None,
        "window_size": None,
        "top_k": None,
        "description": "All layers with probe accuracy >= 0.70 (equivalent to 'wide')",
    },
}

ALL_STRATEGIES = list(LAYER_STRATEGIES.keys())


# ============================================================================
# GLOBAL CONFIG OBJECT
# ============================================================================

@dataclass
class Config:
    """Global configuration object."""

    # Paths
    ROOT_DIR: Path = ROOT_DIR
    PROJECT_ROOT: Path = PROJECT_ROOT

    # Model
    MODEL_NAME: str = MODEL_NAME
    INPUT_DIM: int = INPUT_DIM
    N_LAYERS: int = N_LAYERS

    # Path groups
    PATHS: InputPaths = None
    RESULTS: OutputPaths = None

    # Hyperparameters
    TRAINING: TrainingConfig = None
    INTERVENTION: InterventionConfig = None

    # Layer strategies
    LAYER_STRATEGIES: dict = None
    ALL_STRATEGIES: list = None

    def __post_init__(self):
        if self.PATHS is None:
            self.PATHS = InputPaths()
        if self.RESULTS is None:
            self.RESULTS = OutputPaths()
        if self.TRAINING is None:
            self.TRAINING = TrainingConfig()
        if self.INTERVENTION is None:
            self.INTERVENTION = InterventionConfig()
        if self.LAYER_STRATEGIES is None:
            self.LAYER_STRATEGIES = LAYER_STRATEGIES
        if self.ALL_STRATEGIES is None:
            self.ALL_STRATEGIES = ALL_STRATEGIES


# Instantiate global config
config = Config()


# ============================================================================
# UTILITY FUNCTIONS
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


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate that critical paths exist (run on import)."""
    errors = []
    if not Path(MODEL_NAME).exists():
        errors.append(f"Model not found: {MODEL_NAME}")
    if errors:
        print("Warning: Config validation issues:")
        for err in errors:
            print(f"  - {err}")


validate_config()


# ============================================================================
# VERSION MANAGEMENT
# ============================================================================

def set_version(version):
    """
    Set the active data version.

    Updates config.PATHS and config.RESULTS with version-specific paths.
    Must be called before accessing any version-dependent paths.

    Args:
        version: One of VALID_VERSIONS
    """
    global _active_version

    if version not in VALID_VERSIONS:
        raise ValueError(
            f"Invalid version '{version}'. Must be one of: {VALID_VERSIONS}"
        )

    _active_version = version

    # --- Input paths ---
    config.PATHS.csv_dir = Path(_CSV_DIR_MAP[version])
    config.PATHS.probe_checkpoints = ROOT_DIR / "data" / version / "probe_checkpoints"
    config.PATHS.intervention_results = ROOT_DIR / "data" / version / "intervention_results"

    # Exp 1 paths
    data_gen = _EXP1_DATA_GEN_MAP[version]
    config.PATHS.exp1_data_gen = data_gen
    config.PATHS.exp1_prompts = data_gen / "utils" / "prompts"
    config.PATHS.exp1_configs = data_gen / "utils" / "config"
    config.PATHS.exp1_utils = _EXP1_UTILS_MAP[version]

    # --- Output paths ---
    config.RESULTS.version_root = ROOT_DIR / "results" / "versions" / version
    config.RESULTS.probe_training = ROOT_DIR / "results" / "versions" / version / "probe_training"
    config.RESULTS.degradation = ROOT_DIR / "results" / "versions" / version / "degradation_analysis"
    config.RESULTS.version_logs = ROOT_DIR / "logs" / version

    # Validate key paths
    if not config.PATHS.csv_dir.exists():
        print(f"Warning: CSV dir not found for '{version}': {config.PATHS.csv_dir}")
    if not config.PATHS.probe_checkpoints.exists():
        print(f"Warning: Probe checkpoints not found for '{version}': {config.PATHS.probe_checkpoints}")


def get_active_version():
    """Return the currently active version, or None if not set."""
    return _active_version


def add_version_argument(parser):
    """
    Add a required --version argument to an argparse parser.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--version", type=str, required=True,
        choices=VALID_VERSIONS,
        help=f"Data version to use. Choices: {', '.join(VALID_VERSIONS)}"
    )


def get_version_results_dir(base_path=None):
    """
    Return a version-specific subdirectory under base_path.

    Args:
        base_path: Path to the base results directory (default: config.RESULTS.root)

    Returns:
        Path: base_path / _active_version
    """
    if _active_version is None:
        raise RuntimeError(
            "No version set. Call set_version() before get_version_results_dir()."
        )
    if base_path is None:
        base_path = config.RESULTS.root
    return Path(base_path) / _active_version


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT 2 CONFIGURATION")
    print("=" * 70)

    print(f"\nROOT DIR: {config.ROOT_DIR}")
    print(f"PROJECT ROOT: {config.PROJECT_ROOT}")
    print(f"\nVALID VERSIONS: {VALID_VERSIONS}")
    print(f"Active version: {_active_version or '(not set)'}")

    print(f"\nMODEL: {Path(config.MODEL_NAME).name}")
    print(f"  Hidden dim: {config.INPUT_DIM}")
    print(f"  Layers: {config.N_LAYERS}")

    print(f"\nTRAINING CONFIG:")
    print(f"  Epochs: {config.TRAINING.epochs}")
    print(f"  Batch size (train): {config.TRAINING.batch_size_train}")
    print(f"  Batch size (test): {config.TRAINING.batch_size_test}")

    # Demo: set a version
    for v in VALID_VERSIONS:
        set_version(v)
        print(f"\n--- {v} ---")
        print(f"  csv_dir:          {config.PATHS.csv_dir}")
        print(f"  probe_checkpoints: {config.PATHS.probe_checkpoints}")
        print(f"  intervention:     {config.PATHS.intervention_results}")
        print(f"  exp1_utils:       {config.PATHS.exp1_utils}")
        print(f"  results:          {config.RESULTS.version_root}")

    print(f"\nConfig loaded successfully!")
