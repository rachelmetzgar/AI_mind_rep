#!/usr/bin/env python3
"""
Experiment 3: Central Configuration

All paths, model names, and hyperparameters in one place.
Update this file instead of editing individual scripts.

Usage:
    from config import config, set_version, add_version_argument

    # Access paths (version-independent)
    model = config.MODEL_NAME
    probe_dir = config.RESULTS.concept_probes_data

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

# Valid turn numbers for Exp 2 conversational probes
VALID_TURNS = (1, 2, 3, 4, 5)

# Internal tracking of active version, turn, model, and variant
_active_version = None
_active_turn = None
_active_model = None
_active_variant = ""


# ============================================================================
# MODEL
# ============================================================================

VALID_MODELS = ("llama2_13b_chat",)

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

MODEL_NAME = _MODEL_PATHS["llama2_13b_chat"]
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

    # Experiment 1 shared utils (linguistic markers)
    exp1_utils: Path = PROJECT_ROOT / "exp_1" / "code"

    # Experiment 1 prompts and configs (for V2 multi-turn generation)
    # Set dynamically by set_version(); None until then.
    exp1_prompts: Path = None
    exp1_configs: Path = None

    # Experiment 2 conversational probes (for alignment analysis)
    # Set dynamically by set_version(); None until then.
    exp2_probes: Path = None
    exp2_operational: Path = None
    exp2_metacognitive: Path = None
    exp2_conversations: Path = None

    # Causality test questions (for V1 single-turn generation)
    causality_questions: Path = ROOT_DIR / "code" / "causality_questions.txt"


# ============================================================================
# PATHS: Outputs (results/)
# ============================================================================

def _model_dir():
    """Return the active model directory name."""
    return _active_model or "llama2_13b_chat"


@dataclass
class OutputPaths:
    """Paths to output results.

    Version-independent paths are model-scoped: results/{model}/...
    Version-dependent paths are set by set_version(): results/{model}/{version}/...
    """

    # Root results directory
    root: Path = ROOT_DIR / "results"

    # --- Version-independent, model-scoped ---
    # These are set at init and updated by set_model()
    concept_activations: Path = None
    concept_activations_contrasts: Path = None
    concept_activations_standalone: Path = None
    concept_probes_data: Path = None
    concept_overlap: Path = None
    sysprompt: Path = None
    lexical: Path = None
    comparisons: Path = None

    # --- Version-dependent ---
    # These are None until set_version() is called
    alignment: Path = None
    alignment_versions: Path = None
    alignment_contrasts: Path = None
    alignment_contrasts_raw: Path = None
    alignment_contrasts_residual: Path = None
    alignment_standalone: Path = None
    alignment_layer_profiles: Path = None
    alignment_sysprompt: Path = None
    concept_steering: Path = None
    interventions: Path = None
    interventions_v1: Path = None
    interventions_v2: Path = None
    behavioral: Path = None
    behavioral_v1: Path = None
    behavioral_v2: Path = None
    cross_prediction: Path = None

    # Publication figures (model-scoped)
    figures: Path = None
    figures_pub: Path = None

    # Logs
    logs: Path = ROOT_DIR / "logs"

    def __post_init__(self):
        self._update_model_paths()

    def _update_model_paths(self):
        """Update model-scoped paths (version-independent)."""
        model = _model_dir()
        model_root = self.root / model
        self.concept_activations = model_root / f"concept_activations{_active_variant}"
        self.concept_activations_contrasts = self.concept_activations / "contrasts"
        self.concept_activations_standalone = self.concept_activations / "standalone"
        self.concept_probes_data = model_root / "concept_probes"
        self.concept_overlap = model_root / f"concept_overlap{_active_variant}"
        self.sysprompt = model_root / "sysprompt"
        self.lexical = model_root / "lexical"
        self.comparisons = model_root / "comparisons"
        self.figures = model_root / "figures"
        self.figures_pub = model_root / "figures" / "publication"

    def _update_version_paths(self, version):
        """Update version-dependent paths under results/{model}/{version}/."""
        model = _model_dir()
        ver_root = self.root / model / version
        alignment_base = ver_root / f"alignment{_active_variant}"
        self.alignment = alignment_base
        self.alignment_versions = alignment_base  # use with f"turn_{N}" directly
        self.alignment_contrasts = alignment_base / "contrasts"
        self.alignment_contrasts_raw = alignment_base / "contrasts" / "raw"
        self.alignment_contrasts_residual = alignment_base / "contrasts" / "residual"
        self.alignment_standalone = alignment_base / "standalone"
        self.alignment_layer_profiles = alignment_base / "layer_profiles"
        self.alignment_sysprompt = alignment_base / "sysprompt"
        self.concept_steering = ver_root / f"concept_steering{_active_variant}"
        self.interventions = ver_root / "interventions"
        self.interventions_v1 = ver_root / "interventions" / "V1"
        self.interventions_v2 = ver_root / "interventions" / "V2"
        self.behavioral = ver_root / "behavioral"
        self.behavioral_v1 = ver_root / "behavioral" / "V1"
        self.behavioral_v2 = ver_root / "behavioral" / "V2"
        self.cross_prediction = ver_root / "cross_prediction"


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
    "Mental":    [1, 2, 3, 4, 5, 6, 7, 17, 25, 26, 27],
    "Physical":  [8, 9, 10],
    "Pragmatic": [11, 12, 13],
    "Baseline":  [0],
    "Bio Ctrl":  [14],
    "Shapes":    [15, 29, 30, 31, 32],  # 29=shapes_flip, 30-32=new shape controls
    "Meta":      [16],                   # Mind (holistic)
    "SysPrompt": [18, 20, 21, 22, 23],   # 18=contrasts, 20-23=standalone variants
}

CATEGORY_COLORS = {
    "Mental": "#2196F3",
    "Physical": "#4CAF50",
    "Pragmatic": "#FF9800",
    "Baseline": "#9E9E9E",
    "Bio Ctrl": "#795548",
    "Shapes": "#E91E63",
    "Meta": "#9C27B0",
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
        print("Config validation warnings:")
        for err in errors:
            print(f"  - {err}")
        print("Some features may not work until these paths are available.")


# Run validation on import
validate_config()


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def set_model(model):
    """Set the active model. Updates MODEL_NAME, INPUT_DIM, N_LAYERS, and output paths."""
    global _active_model
    if model not in VALID_MODELS:
        raise ValueError(f"Invalid model '{model}'. Must be one of: {VALID_MODELS}")
    _active_model = model
    config.MODEL_NAME = _MODEL_PATHS[model]
    mc = _MODEL_CONFIGS[model]
    config.INPUT_DIM = mc["input_dim"]
    config.N_LAYERS = mc["n_layers"]
    config.RESULTS._update_model_paths()


def get_model():
    """Return the active model name (default: llama2_13b_chat)."""
    return _active_model or "llama2_13b_chat"


def add_model_argument(parser):
    """Add an optional --model argument to an argparse parser."""
    parser.add_argument(
        "--model", type=str, default="llama2_13b_chat",
        choices=VALID_MODELS,
        help=f"Model to use. Choices: {', '.join(VALID_MODELS)} (default: llama2_13b_chat)"
    )


# ============================================================================
# VARIANT MANAGEMENT (for pipeline variants like _1 top-prompt)
# ============================================================================

def set_variant(variant):
    """Set active variant suffix (e.g., '_1'). Updates concept activation + output paths."""
    global _active_variant
    _active_variant = variant
    config.RESULTS._update_model_paths()
    if _active_version:
        config.RESULTS._update_version_paths(_active_version)


def add_variant_argument(parser):
    """Add optional --variant argument to argparse parser."""
    parser.add_argument(
        "--variant", type=str, default="",
        help="Variant suffix for concept paths (e.g., '_1' for top-1 pipeline)"
    )


# ============================================================================
# VERSION MANAGEMENT (for multi-version Exp 2/Exp 1 support)
# ============================================================================

def set_version(version: str, turn: int = 5):
    """
    Set the active Exp 2 / Exp 1 data version and conversation turn.

    Updates config.EXP2_ROOT, config.EXP1_ROOT, all derived input paths,
    and version-dependent output paths.
    Must be called before accessing any Exp 2 or Exp 1 paths.

    Args:
        version: One of VALID_VERSIONS
        turn: Conversation turn for Exp 2 probes (1-5, default=5)
    """
    global _active_version, _active_turn, EXP2_ROOT, EXP1_ROOT

    if version not in VALID_VERSIONS:
        raise ValueError(
            f"Invalid version '{version}'. Must be one of: {VALID_VERSIONS}"
        )
    if turn not in VALID_TURNS:
        raise ValueError(
            f"Invalid turn {turn}. Must be one of: {VALID_TURNS}"
        )

    _active_version = version
    _active_turn = turn
    EXP2_ROOT = PROJECT_ROOT / "exp_2"
    EXP1_ROOT = PROJECT_ROOT / "exp_1" / "versions" / version

    # Update global config object
    config.EXP2_ROOT = EXP2_ROOT
    config.EXP1_ROOT = EXP1_ROOT

    # Update Exp 2 input paths (post-refactor structure)
    model = get_model()
    turn_dir = f"turn_{turn}"
    config.PATHS.exp2_probes = EXP2_ROOT / "results" / model / version / "probe_training" / "data"
    config.PATHS.exp2_operational = config.PATHS.exp2_probes / turn_dir / "operational"
    config.PATHS.exp2_metacognitive = config.PATHS.exp2_probes / turn_dir / "metacognitive"
    config.PATHS.exp2_conversations = EXP2_ROOT / "data" / version / "human_ai_conversations"

    # Update Exp 1 input paths
    config.PATHS.exp1_prompts = EXP1_ROOT / "code" / "data_gen" / "utils" / "prompts"
    config.PATHS.exp1_configs = EXP1_ROOT / "code" / "data_gen" / "utils" / "config"

    # Update version-dependent output paths
    config.RESULTS._update_version_paths(version)

    # Validate that key Exp 2 paths exist
    if not config.PATHS.exp2_probes.exists():
        print(f"Exp 2 probes not found for version '{version}': {config.PATHS.exp2_probes}")


def add_version_argument(parser):
    """Add a required --version argument to an argparse parser."""
    parser.add_argument(
        "--version", type=str, required=True,
        choices=VALID_VERSIONS,
        help=f"Exp 2 data version to use. Choices: {', '.join(VALID_VERSIONS)}"
    )


def get_version_results_dir(base_path):
    """Return a version-specific subdirectory under base_path."""
    if _active_version is None:
        raise RuntimeError(
            "No version set. Call set_version() before get_version_results_dir()."
        )
    return Path(base_path) / _active_version


def add_turn_argument(parser):
    """Add an optional --turn argument to an argparse parser."""
    parser.add_argument(
        "--turn", type=int, default=5,
        choices=list(VALID_TURNS),
        help=f"Exp 2 conversation turn for probes (default: 5). Choices: {VALID_TURNS}"
    )


def get_turn_results_dir(base_path):
    """Return a version- and turn-specific subdirectory under base_path."""
    if _active_version is None:
        raise RuntimeError(
            "No version set. Call set_version() before get_turn_results_dir()."
        )
    if _active_turn is None:
        raise RuntimeError(
            "No turn set. Call set_version(version, turn=N) before get_turn_results_dir()."
        )
    return Path(base_path) / _active_version / f"turn_{_active_turn}"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 3 CONFIGURATION")
    print("="*70)

    print(f"\nROOT DIR: {config.ROOT_DIR}")
    print(f"PROJECT ROOT: {config.PROJECT_ROOT}")
    print(f"\nVALID VERSIONS: {VALID_VERSIONS}")
    print(f"   Active version: {_active_version or '(not set)'}")
    print(f"   Active model: {get_model()}")

    # Demo: set a version to show derived paths
    print(f"\n--- Setting version='balanced_gpt' for demo ---")
    set_version("balanced_gpt")
    print(f"   EXP2_ROOT: {config.EXP2_ROOT}")
    print(f"   EXP1_ROOT: {config.EXP1_ROOT}")
    print(f"   Exp 2 probes: {config.PATHS.exp2_probes}")
    print(f"   Exp 2 conversations: {config.PATHS.exp2_conversations}")

    print(f"\nMODEL: {Path(config.MODEL_NAME).name}")
    print(f"   Hidden dim: {config.INPUT_DIM}")
    print(f"   Layers: {config.N_LAYERS}")

    print(f"\nKEY INPUT PATHS:")
    print(f"   Concepts (contrasts): {config.PATHS.concepts_contrasts}")
    print(f"   Concepts (standalone): {config.PATHS.concepts_standalone}")
    print(f"   Exp 1 utils: {config.PATHS.exp1_utils}")
    print(f"   Causality questions: {config.PATHS.causality_questions}")

    print(f"\nKEY OUTPUT PATHS (model-scoped):")
    print(f"   Results root: {config.RESULTS.root}")
    print(f"   Concept overlap: {config.RESULTS.concept_overlap}")
    print(f"   Comparisons: {config.RESULTS.comparisons}")

    print(f"\nKEY OUTPUT PATHS (version-dependent):")
    print(f"   Alignment: {config.RESULTS.alignment}")
    print(f"   Concept steering: {config.RESULTS.concept_steering}")
    print(f"   Interventions V1: {config.RESULTS.interventions_v1}")
    print(f"   Behavioral: {config.RESULTS.behavioral}")
    print(f"   Cross-prediction: {config.RESULTS.cross_prediction}")

    print(f"\nTRAINING CONFIG:")
    print(f"   Epochs: {config.TRAINING.epochs}")
    print(f"   Batch size: {config.TRAINING.batch_size_train}")
    print(f"   Min probe accuracy: {config.TRAINING.min_probe_accuracy}")

    print(f"\nANALYSIS CONFIG:")
    print(f"   Bootstrap iterations: {config.ANALYSIS.n_bootstrap}")
    print(f"   Permutation tests: {config.ANALYSIS.n_permutations}")
    print(f"   Random seed: {config.ANALYSIS.seed}")

    print(f"\nDIMENSION CATEGORIES:")
    for cat, dims in config.DIMENSION_CATEGORIES.items():
        print(f"   {cat}: {dims}")

    print(f"\nConfig loaded successfully!")
