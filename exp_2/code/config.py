#!/usr/bin/env python3
"""
Experiment 2: Central Configuration

All paths, model names, and hyperparameters in one place.
Update this file instead of editing individual scripts.

Usage:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import config, set_version, add_version_argument

    # Set data version (required before accessing version-dependent paths)
    set_version("balanced_gpt")
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

# Valid models
VALID_MODELS = ("llama2_13b_chat",)

# Internal tracking
_active_version = None
_active_model = None


# ============================================================================
# MODEL
# ============================================================================

# Model weight paths (keyed by model short name)
_MODEL_PATHS = {
    "llama2_13b_chat": (
        "/mnt/cup/labs/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
        "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
        "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
    ),
}

_MODEL_CONFIGS = {
    "llama2_13b_chat": {"input_dim": 5120, "n_layers": 41},
}

# Default model
MODEL_NAME = _MODEL_PATHS["llama2_13b_chat"]
INPUT_DIM = 5120   # LLaMA-2-13B hidden size
N_LAYERS = 41      # Embedding layer + 40 transformer layers


# ============================================================================
# EXP 1 PATH FORMULAS
# ============================================================================

def _exp1_data_dir(version, model="llama2_13b_chat"):
    """Exp 1 per-subject CSV directory for a given version and model."""
    return PROJECT_ROOT / "exp_1" / "results" / model / version / "data"

# Exp 1 shared code paths (not per-version after exp_1 refactoring)
EXP1_DATA_GEN = PROJECT_ROOT / "exp_1" / "code" / "data_gen"
EXP1_PROMPTS = EXP1_DATA_GEN / "prompts"
EXP1_CONDITIONS = EXP1_DATA_GEN / "conditions"
EXP1_UTILS = PROJECT_ROOT / "exp_1" / "code" / "utils"


# ============================================================================
# PATHS: Inputs
# ============================================================================

@dataclass
class InputPaths:
    """Paths to input data. Version-dependent paths are None until set_version()."""

    # Per-subject CSV directory (set by set_version)
    csv_dir: Path = None

    # Probe checkpoints: results/{model}/{version}/probe_training/data/
    probe_checkpoints: Path = None

    # Intervention data: results/{model}/{version}/V{1,2}_causality/data/
    intervention_results_v1: Path = None
    intervention_results_v2: Path = None

    # Exp 1 shared paths
    exp1_data_gen: Path = EXP1_DATA_GEN
    exp1_prompts: Path = EXP1_PROMPTS
    exp1_configs: Path = EXP1_DATA_GEN / "conditions"
    exp1_utils: Path = EXP1_UTILS

    # Shared data (version-independent, tracked by git)
    topics_csv: Path = ROOT_DIR / "code" / "shared" / "conds" / "topics.csv"
    causality_questions: Path = ROOT_DIR / "code" / "shared" / "causality_test_questions" / "human_ai.txt"


# ============================================================================
# PATHS: Outputs (results/)
# ============================================================================

@dataclass
class OutputPaths:
    """Paths to output results. Version-dependent paths are None until set_version()."""

    # Root results directory
    root: Path = ROOT_DIR / "results"

    # Version-specific results (set by set_version)
    version_root: Path = None      # results/{model}/{version}/
    probe_training: Path = None    # results/{model}/{version}/probe_training/
    degradation: Path = None       # results/{model}/{version}/probe_training/degradation/

    # V1/V2 behavioral output
    v1_behavioral: Path = None
    v2_behavioral: Path = None

    # Cross-variant comparison results (set by set_version or set_model)
    comparisons: Path = None
    probe_training_comparison: Path = None
    causality_qc_comparison: Path = None

    # Logs
    logs: Path = ROOT_DIR / "logs"
    version_logs: Path = None      # logs/{model}/{version}/


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
# MODEL MANAGEMENT
# ============================================================================

def set_model(model):
    """Set the active model. Updates MODEL_NAME, INPUT_DIM, N_LAYERS."""
    global _active_model
    if model not in VALID_MODELS:
        raise ValueError(f"Invalid model '{model}'. Must be one of: {VALID_MODELS}")
    _active_model = model
    config.MODEL_NAME = _MODEL_PATHS[model]
    mc = _MODEL_CONFIGS[model]
    config.INPUT_DIM = mc["input_dim"]
    config.N_LAYERS = mc["n_layers"]


def get_model():
    """Return the active model short name (default: llama2_13b_chat)."""
    return _active_model or "llama2_13b_chat"


def add_model_argument(parser):
    """Add an optional --model argument to an argparse parser."""
    parser.add_argument(
        "--model", type=str, default="llama2_13b_chat",
        choices=VALID_MODELS,
        help=f"Model to use. Choices: {', '.join(VALID_MODELS)} (default: llama2_13b_chat)"
    )


# ============================================================================
# VERSION MANAGEMENT
# ============================================================================

def set_version(version, model=None):
    """
    Set the active data version.

    Updates config.PATHS and config.RESULTS with version-specific paths.
    Must be called before accessing any version-dependent paths.

    Args:
        version: One of VALID_VERSIONS
        model: Model short name (default: current model or llama2_13b_chat)
    """
    global _active_version

    if version not in VALID_VERSIONS:
        raise ValueError(
            f"Invalid version '{version}'. Must be one of: {VALID_VERSIONS}"
        )

    _active_version = version
    m = model or get_model()

    # --- Input paths ---
    config.PATHS.csv_dir = _exp1_data_dir(version, m)
    config.PATHS.probe_checkpoints = ROOT_DIR / "results" / m / version / "probe_training" / "data"
    config.PATHS.intervention_results_v1 = ROOT_DIR / "results" / m / version / "V1_causality" / "data"
    config.PATHS.intervention_results_v2 = ROOT_DIR / "results" / m / version / "V2_causality" / "data"

    # --- Output paths ---
    config.RESULTS.version_root = ROOT_DIR / "results" / m / version
    config.RESULTS.probe_training = ROOT_DIR / "results" / m / version / "probe_training"
    config.RESULTS.degradation = ROOT_DIR / "results" / m / version / "probe_training" / "degradation"
    config.RESULTS.v1_behavioral = ROOT_DIR / "results" / m / version / "V1_causality" / "behavioral"
    config.RESULTS.v2_behavioral = ROOT_DIR / "results" / m / version / "V2_causality" / "behavioral"

    # Comparisons
    config.RESULTS.comparisons = ROOT_DIR / "results" / "comparisons" / m
    config.RESULTS.probe_training_comparison = ROOT_DIR / "results" / "comparisons" / m / "probe_training"
    config.RESULTS.causality_qc_comparison = ROOT_DIR / "results" / "comparisons" / m / "causality_qc"

    # Logs
    config.RESULTS.version_logs = ROOT_DIR / "logs" / m / version

    # Validate key paths
    if not config.PATHS.csv_dir.exists():
        print(f"Warning: CSV dir not found for '{version}': {config.PATHS.csv_dir}")
    if not config.PATHS.probe_checkpoints.exists():
        print(f"Warning: Probe checkpoints not found for '{version}': {config.PATHS.probe_checkpoints}")


def get_active_version():
    """Return the currently active version, or None if not set."""
    return _active_version


def add_version_argument(parser):
    """Add a required --version argument to an argparse parser."""
    parser.add_argument(
        "--version", type=str, required=True,
        choices=VALID_VERSIONS,
        help=f"Data version to use. Choices: {', '.join(VALID_VERSIONS)}"
    )


def get_version_results_dir(base_path=None):
    """Return a version-specific subdirectory under base_path."""
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
    print(f"VALID MODELS: {VALID_MODELS}")
    print(f"Active version: {_active_version or '(not set)'}")
    print(f"Active model: {get_model()}")

    print(f"\nMODEL: {Path(config.MODEL_NAME).name}")
    print(f"  Hidden dim: {config.INPUT_DIM}")
    print(f"  Layers: {config.N_LAYERS}")

    print(f"\nTRAINING CONFIG:")
    print(f"  Epochs: {config.TRAINING.epochs}")
    print(f"  Batch size (train): {config.TRAINING.batch_size_train}")
    print(f"  Batch size (test): {config.TRAINING.batch_size_test}")

    # Demo: set a version
    for v in ("balanced_gpt", "nonsense_codeword"):
        set_version(v)
        print(f"\n--- {v} ---")
        print(f"  csv_dir:           {config.PATHS.csv_dir}")
        print(f"  probe_checkpoints: {config.PATHS.probe_checkpoints}")
        print(f"  intervention_v1:   {config.PATHS.intervention_results_v1}")
        print(f"  exp1_utils:        {config.PATHS.exp1_utils}")
        print(f"  results:           {config.RESULTS.version_root}")
        print(f"  comparisons:       {config.RESULTS.comparisons}")
        print(f"  logs:              {config.RESULTS.version_logs}")

    print(f"\nConfig loaded successfully!")
