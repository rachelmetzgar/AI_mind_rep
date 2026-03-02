#!/usr/bin/env python3
"""
Experiment 3: Central Configuration

All paths, model names, and hyperparameters in one place.
Update this file instead of editing individual scripts.

Usage:
    from config import config, set_version, add_version_argument

    # Access paths (version-independent)
    model = config.MODEL_NAME
    probe_dir = config.PATHS.concept_probes

    # Set Exp 2 / Exp 1 version (required before accessing exp2/exp1 paths)
    set_version("labels")
    exp2_probes = config.PATHS.exp2_probes

    # Access hyperparameters
    batch_size = config.TRAINING.batch_size
    n_bootstrap = config.ANALYSIS.n_bootstrap
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass


# ============================================================================
# BASE PATHS
# ============================================================================

# Root directory for this experiment (exp_3/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Root for entire project (mind_rep/)
PROJECT_ROOT = ROOT_DIR.parent

# Valid Exp 2 / Exp 1 data versions
VALID_VERSIONS = (
    "labels", "balanced_names", "balanced_gpt", "names",
    "nonsense_codeword", "nonsense_ignore",
    "labels_turnwise", "you_are_labels",
    "you_are_balanced_gpt", "you_are_labels_turnwise",
)

# Experiment 2 and 1 roots are set dynamically via set_version().
# None until set_version() is called.
EXP2_ROOT = None
EXP1_ROOT = None

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

INPUT_DIM = 5120  # LLaMA-2-13B hidden size
N_LAYERS = 41     # Embedding layer + 40 transformer layers


# ============================================================================
# PATHS: Inputs (data/)
# ============================================================================

@dataclass
class InputPaths:
    """Paths to input data (activations, prompts, etc.)"""

    # Concept definitions (prompts)
    concepts_root: Path = ROOT_DIR / "concepts"
    concepts_contrasts: Path = ROOT_DIR / "concepts" / "contrasts"
    concepts_standalone: Path = ROOT_DIR / "concepts" / "standalone"

    # Experiment 1 prompts and configs (for V2 multi-turn generation)
    # Set dynamically by set_version(); None until then.
    exp1_prompts: Path = None
    exp1_configs: Path = None

    # Experiment 2 conversational probes (for alignment analysis)
    # Set dynamically by set_version(); None until then.
    exp2_probes: Path = None
    exp2_control_probe: Path = None
    exp2_reading_probe: Path = None
    exp2_conversations: Path = None

    # Concept activations (extracted in Phase 1)
    concept_activations: Path = ROOT_DIR / "data" / "concept_activations"
    concept_activations_contrasts: Path = ROOT_DIR / "data" / "concept_activations" / "contrasts"
    concept_activations_standalone: Path = ROOT_DIR / "data" / "concept_activations" / "standalone"

    # Concept probes (trained in Phase 2)
    concept_probes: Path = ROOT_DIR / "data" / "concept_probes"

    # Causality test questions (for V1 single-turn generation)
    causality_questions: Path = ROOT_DIR / "data" / "causality_test_questions" / "human_ai.txt"


# ============================================================================
# PATHS: Outputs (results/)
# ============================================================================

@dataclass
class OutputPaths:
    """Paths to output results (all go in results/, not data/)"""

    # Root results directory
    root: Path = ROOT_DIR / "results"

    # Phase 1: Concept activation extraction
    concept_activations: Path = ROOT_DIR / "results" / "concept_activations"

    # Phase 1b: Alignment analysis
    alignment: Path = ROOT_DIR / "results" / "alignment"
    alignment_versions: Path = ROOT_DIR / "results" / "alignment" / "versions"
    alignment_contrasts: Path = ROOT_DIR / "results" / "alignment" / "contrasts"
    alignment_contrasts_raw: Path = ROOT_DIR / "results" / "alignment" / "contrasts" / "raw"
    alignment_contrasts_residual: Path = ROOT_DIR / "results" / "alignment" / "contrasts" / "residual"
    alignment_standalone: Path = ROOT_DIR / "results" / "alignment" / "standalone"
    alignment_layer_profiles: Path = ROOT_DIR / "results" / "alignment" / "layer_profiles"
    alignment_sysprompt: Path = ROOT_DIR / "results" / "alignment" / "sysprompt"

    # Phase 2: Concept probe training
    concept_probes: Path = ROOT_DIR / "results" / "concept_probes"
    concept_probe_summaries: Path = ROOT_DIR / "results" / "concept_probes" / "summaries"

    # Phase 3: Concept intervention (generation)
    interventions: Path = ROOT_DIR / "results" / "interventions"
    interventions_v1: Path = ROOT_DIR / "results" / "interventions" / "V1"
    interventions_v2: Path = ROOT_DIR / "results" / "interventions" / "V2"

    # Phase 4: Behavioral analysis
    behavioral: Path = ROOT_DIR / "results" / "behavioral"
    behavioral_v1: Path = ROOT_DIR / "results" / "behavioral" / "V1"
    behavioral_v2: Path = ROOT_DIR / "results" / "behavioral" / "V2"

    # Phase 5: Cross-prediction
    cross_prediction: Path = ROOT_DIR / "results" / "cross_prediction"

    # Lexical distinctiveness analysis
    lexical: Path = ROOT_DIR / "results" / "lexical"

    # Publication figures
    figures: Path = ROOT_DIR / "results" / "figures"
    figures_pub: Path = ROOT_DIR / "results" / "figures" / "publication"

    # Logs
    logs: Path = ROOT_DIR / "logs"


# ============================================================================
# HYPERPARAMETERS: Training
# ============================================================================

@dataclass
class TrainingConfig:
    """Hyperparameters for probe training"""

    # Training loop
    epochs: int = 50
    batch_size_train: int = 16
    batch_size_test: int = 16

    # Probe architecture
    logistic: bool = True  # Use sigmoid activation (vs softmax)

    # Intervention (Phase 3)
    min_probe_accuracy: float = 0.70  # Only use probes with ≥70% accuracy
    default_strengths: list = None  # [1, 2, 4, 8]

    def __post_init__(self):
        if self.default_strengths is None:
            self.default_strengths = [1, 2, 4, 8]


# ============================================================================
# HYPERPARAMETERS: Generation (Phase 3)
# ============================================================================

@dataclass
class GenerationConfigV1:
    """Hyperparameters for V1 single-turn generation"""
    max_new_tokens: int = 768
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    batch_size: int = 5


@dataclass
class GenerationConfigV2:
    """Hyperparameters for V2 multi-turn generation"""
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_p: float = 1.0
    pairs_total: int = 4
    history_pairs: int = 4
    system_prompt: str = "You are engaging in a real-time spoken conversation."
    conditions: list = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = ["baseline", "human", "ai"]


# ============================================================================
# HYPERPARAMETERS: Analysis
# ============================================================================

@dataclass
class AnalysisConfig:
    """Hyperparameters for statistical analysis"""

    # Bootstrap confidence intervals
    n_bootstrap: int = 1000
    ci_alpha: float = 0.05  # 95% CI

    # Permutation testing
    n_permutations: int = 10000

    # Split-half reliability
    n_split_half: int = 100

    # Random seed
    seed: int = 42

    # Layer filtering (exclude early layers with prompt-format confound)
    restricted_layer_start: int = 6

    # Entity baseline dimension (for residual analysis)
    entity_baseline_dim: str = "0_baseline"


# ============================================================================
# DIMENSION CATEGORIES
# ============================================================================

DIMENSION_CATEGORIES = {
    "Mental":    [1, 2, 3, 4, 5, 6, 7, 16, 17],
    "Physical":  [8, 9, 10],
    "Pragmatic": [11, 12, 13],
    "Baseline":  [0],
    "Bio Ctrl":  [14],
    "Shapes":    [15],
    "SysPrompt": [18, 19, 20, 21],
}

CATEGORY_COLORS = {
    "Mental": "#2196F3",
    "Physical": "#4CAF50",
    "Pragmatic": "#FF9800",
    "Baseline": "#9E9E9E",
    "Bio Ctrl": "#795548",
    "Shapes": "#E91E63",
    "SysPrompt": "#00BCD4",
}


# ============================================================================
# GLOBAL CONFIG OBJECT
# ============================================================================

@dataclass
class Config:
    """Global configuration object"""

    # Paths
    ROOT_DIR: Path = ROOT_DIR
    PROJECT_ROOT: Path = PROJECT_ROOT
    EXP2_ROOT: Path = None
    EXP1_ROOT: Path = None

    # Model
    MODEL_NAME: str = MODEL_NAME
    INPUT_DIM: int = INPUT_DIM
    N_LAYERS: int = N_LAYERS

    # Path groups
    PATHS: InputPaths = None
    RESULTS: OutputPaths = None

    # Hyperparameters
    TRAINING: TrainingConfig = None
    GEN_V1: GenerationConfigV1 = None
    GEN_V2: GenerationConfigV2 = None
    ANALYSIS: AnalysisConfig = None

    # Categories
    DIMENSION_CATEGORIES: dict = None
    CATEGORY_COLORS: dict = None

    def __post_init__(self):
        if self.PATHS is None:
            self.PATHS = InputPaths()
        if self.RESULTS is None:
            self.RESULTS = OutputPaths()
        if self.TRAINING is None:
            self.TRAINING = TrainingConfig()
        if self.GEN_V1 is None:
            self.GEN_V1 = GenerationConfigV1()
        if self.GEN_V2 is None:
            self.GEN_V2 = GenerationConfigV2()
        if self.ANALYSIS is None:
            self.ANALYSIS = AnalysisConfig()
        if self.DIMENSION_CATEGORIES is None:
            self.DIMENSION_CATEGORIES = DIMENSION_CATEGORIES
        if self.CATEGORY_COLORS is None:
            self.CATEGORY_COLORS = CATEGORY_COLORS


# Instantiate global config
config = Config()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_dimension_category(dim_id: int) -> str:
    """Get the category name for a dimension ID."""
    for cat, ids in DIMENSION_CATEGORIES.items():
        if dim_id in ids:
            return cat
    return "Other"


def get_category_color(category: str) -> str:
    """Get the plot color for a category."""
    return CATEGORY_COLORS.get(category, "#000000")


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device():
    """Get PyTorch device (cuda if available, else cpu)."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# VALIDATION (run on import to catch config errors early)
# ============================================================================

def validate_config():
    """Validate that critical paths exist.

    Note: Exp 2 paths are not validated here because they depend on
    set_version() which is called at runtime, not import time.
    """
    errors = []

    # Check model exists
    if not Path(MODEL_NAME).exists():
        errors.append(f"Model not found: {MODEL_NAME}")

    # Check concepts directory exists
    if not config.PATHS.concepts_root.exists():
        errors.append(f"Concepts directory not found: {config.PATHS.concepts_root}")

    if errors:
        print("⚠️  Config validation warnings:")
        for err in errors:
            print(f"  - {err}")
        print("Some features may not work until these paths are available.")


# Run validation on import
validate_config()


# ============================================================================
# VERSION MANAGEMENT (for multi-version Exp 2/Exp 1 support)
# ============================================================================

def set_version(version: str):
    """
    Set the active Exp 2 / Exp 1 data version.

    Updates config.EXP2_ROOT, config.EXP1_ROOT, and all derived input paths.
    Must be called before accessing any Exp 2 or Exp 1 paths.

    Args:
        version: One of VALID_VERSIONS
    """
    global _active_version, EXP2_ROOT, EXP1_ROOT

    if version not in VALID_VERSIONS:
        raise ValueError(
            f"Invalid version '{version}'. Must be one of: {VALID_VERSIONS}"
        )

    _active_version = version
    EXP2_ROOT = PROJECT_ROOT / "exp_2"
    EXP1_ROOT = PROJECT_ROOT / "exp_1" / "versions" / version

    # Update global config object
    config.EXP2_ROOT = EXP2_ROOT
    config.EXP1_ROOT = EXP1_ROOT

    # Update all derived input paths (new exp_2 structure: data/{version}/...)
    config.PATHS.exp2_probes = EXP2_ROOT / "data" / version / "probe_checkpoints"
    config.PATHS.exp2_control_probe = EXP2_ROOT / "data" / version / "probe_checkpoints" / "turn_5" / "control_probe"
    config.PATHS.exp2_reading_probe = EXP2_ROOT / "data" / version / "probe_checkpoints" / "turn_5" / "reading_probe"
    config.PATHS.exp2_conversations = EXP2_ROOT / "data" / version / "human_ai_conversations"
    config.PATHS.exp1_prompts = EXP1_ROOT / "code" / "data_gen" / "utils" / "prompts"
    config.PATHS.exp1_configs = EXP1_ROOT / "code" / "data_gen" / "utils" / "config"

    # Validate that key Exp 2 paths exist
    if not config.PATHS.exp2_probes.exists():
        print(f"⚠️  Exp 2 probes not found for version '{version}': {config.PATHS.exp2_probes}")


def add_version_argument(parser):
    """
    Add a required --version argument to an argparse parser.

    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--version", type=str, required=True,
        choices=VALID_VERSIONS,
        help=f"Exp 2 data version to use. Choices: {', '.join(VALID_VERSIONS)}"
    )


def get_version_results_dir(base_path):
    """
    Return a version-specific subdirectory under base_path.

    Args:
        base_path: Path to the base results directory

    Returns:
        Path: base_path / _active_version
    """
    if _active_version is None:
        raise RuntimeError(
            "No version set. Call set_version() before get_version_results_dir()."
        )
    return Path(base_path) / _active_version


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 3 CONFIGURATION")
    print("="*70)

    print(f"\n📁 ROOT DIR: {config.ROOT_DIR}")
    print(f"📁 PROJECT ROOT: {config.PROJECT_ROOT}")
    print(f"\n📦 VALID VERSIONS: {VALID_VERSIONS}")
    print(f"   Active version: {_active_version or '(not set — call set_version())'}")

    # Demo: set a version to show derived paths
    print(f"\n--- Setting version='balanced_names' for demo ---")
    set_version("balanced_names")
    print(f"   EXP2_ROOT: {config.EXP2_ROOT}")
    print(f"   EXP1_ROOT: {config.EXP1_ROOT}")
    print(f"   Exp 2 probes: {config.PATHS.exp2_probes}")
    print(f"   Exp 2 conversations: {config.PATHS.exp2_conversations}")

    print(f"\n🤖 MODEL: {Path(config.MODEL_NAME).name}")
    print(f"   Hidden dim: {config.INPUT_DIM}")
    print(f"   Layers: {config.N_LAYERS}")

    print(f"\n📂 KEY INPUT PATHS:")
    print(f"   Concepts (contrasts): {config.PATHS.concepts_contrasts}")
    print(f"   Concepts (standalone): {config.PATHS.concepts_standalone}")

    print(f"\n📂 KEY OUTPUT PATHS:")
    print(f"   Results root: {config.RESULTS.root}")
    print(f"   Concept activations: {config.RESULTS.concept_activations}")
    print(f"   Alignment: {config.RESULTS.alignment}")
    print(f"   Interventions: {config.RESULTS.interventions}")
    print(f"   Behavioral: {config.RESULTS.behavioral}")

    print(f"\n⚙️  TRAINING CONFIG:")
    print(f"   Epochs: {config.TRAINING.epochs}")
    print(f"   Batch size: {config.TRAINING.batch_size_train}")
    print(f"   Min probe accuracy: {config.TRAINING.min_probe_accuracy}")

    print(f"\n⚙️  ANALYSIS CONFIG:")
    print(f"   Bootstrap iterations: {config.ANALYSIS.n_bootstrap}")
    print(f"   Permutation tests: {config.ANALYSIS.n_permutations}")
    print(f"   Random seed: {config.ANALYSIS.seed}")

    print(f"\n📊 DIMENSION CATEGORIES:")
    for cat, dims in config.DIMENSION_CATEGORIES.items():
        print(f"   {cat}: {dims}")

    print(f"\n✅ Config loaded successfully!")
