#!/usr/bin/env python3
"""
Experiment 3: Central Configuration

All paths, model names, and hyperparameters in one place.
Update this file instead of editing individual scripts.

Usage:
    from config import Config

    # Access paths
    model = Config.MODEL_NAME
    probe_dir = Config.PATHS.concept_probes

    # Access hyperparameters
    batch_size = Config.TRAINING.batch_size
    n_bootstrap = Config.ANALYSIS.n_bootstrap
"""

import os
from pathlib import Path
from dataclasses import dataclass


# ============================================================================
# BASE PATHS
# ============================================================================

# Root directory for this experiment (exp_3/labels/)
ROOT_DIR = Path(__file__).parent.absolute()

# Root for entire project (ai_mind_rep/)
PROJECT_ROOT = ROOT_DIR.parent.parent

# Experiment 2 location (for loading conversational probes)
# Note: Using balanced_names version (labels-only, no name confounds)
EXP2_ROOT = PROJECT_ROOT / "exp_2" / "balanced_names" / "llama_exp_2b-13B-chat"

# Experiment 1 location (for V2 multi-turn generation prompts)
# Note: Using balanced_names version
EXP1_ROOT = PROJECT_ROOT / "exp_1" / "balanced_names"


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
    exp1_prompts: Path = EXP1_ROOT / "code" / "data_gen" / "utils" / "prompts"
    exp1_configs: Path = EXP1_ROOT / "code" / "data_gen" / "utils" / "config"

    # Experiment 2 conversational probes (for alignment analysis)
    exp2_probes: Path = EXP2_ROOT / "data" / "probe_checkpoints"
    exp2_control_probe: Path = EXP2_ROOT / "data" / "probe_checkpoints" / "control_probe"
    exp2_reading_probe: Path = EXP2_ROOT / "data" / "probe_checkpoints" / "reading_probe"
    exp2_conversations: Path = EXP2_ROOT / "data" / "human_ai_conversations"

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
    EXP2_ROOT: Path = EXP2_ROOT
    EXP1_ROOT: Path = EXP1_ROOT

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
    """Validate that critical paths exist."""
    errors = []

    # Check model exists
    if not Path(MODEL_NAME).exists():
        errors.append(f"Model not found: {MODEL_NAME}")

    # Check Exp 2 probes exist
    if not config.PATHS.exp2_probes.exists():
        errors.append(f"Exp 2 probes not found: {config.PATHS.exp2_probes}")

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
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENT 3 CONFIGURATION")
    print("="*70)

    print(f"\n📁 ROOT DIR: {config.ROOT_DIR}")
    print(f"📁 PROJECT ROOT: {config.PROJECT_ROOT}")

    print(f"\n🤖 MODEL: {Path(config.MODEL_NAME).name}")
    print(f"   Hidden dim: {config.INPUT_DIM}")
    print(f"   Layers: {config.N_LAYERS}")

    print(f"\n📂 KEY INPUT PATHS:")
    print(f"   Concepts (contrasts): {config.PATHS.concepts_contrasts}")
    print(f"   Concepts (standalone): {config.PATHS.concepts_standalone}")
    print(f"   Exp 2 probes: {config.PATHS.exp2_probes}")
    print(f"   Exp 1 prompts: {config.PATHS.exp1_prompts}")

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
