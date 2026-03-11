"""
Concept Geometry: Concept Dimension Definitions

Auto-discovers all standalone concept dimensions from exp_3/concepts/standalone/.
Each concept provides:
    - prompts: list of standalone prompts (40 per concept, 280 for dim 19)
    - pairwise_prompt: used in Phase A behavioral PCA ("which character is
      more capable of ...")
    - categories: sub-facet info from the source file

To add a new concept dimension, just add a new file to exp_3/concepts/standalone/
following the naming convention {N}_{name}.py with STANDALONE_PROMPTS_DIM{N}
and CATEGORY_INFO_STANDALONE_DIM{N}.

Rachel C. Metzgar · Mar 2026
"""

import re
import importlib
import sys
from pathlib import Path


# ============================================================================
# PAIRWISE PROMPTS — one per concept, used in Phase A
# ============================================================================
# Complete: "which character is more capable of ..."

PAIRWISE_PROMPTS = {
    1:  "which character is more capable of having subjective conscious experiences",
    2:  "which character is more capable of having genuine emotional experiences",
    3:  "which character is more capable of acting autonomously and exercising free will",
    4:  "which character is more capable of forming intentions and pursuing goals",
    5:  "which character is more capable of anticipating and predicting the behavior of others",
    6:  "which character is more capable of complex cognitive processes like reasoning and problem-solving",
    7:  "which character is more capable of understanding what others are thinking and feeling",
    8:  "which character is more capable of having a physical body and embodied experiences",
    9:  "which character is more capable of fulfilling functional roles and carrying out tasks",
    10: "which character is more likely to be a living, animate being",
    11: "which character is more likely to communicate in a formal, structured manner",
    12: "which character is more likely to have deep expertise and specialized knowledge",
    13: "which character is more likely to be helpful and service-oriented",
    14: "which character is more likely to undergo biological processes like growth and metabolism",
    15: "which character is more likely to have a distinctive geometric shape",
    16: "which character is more likely to be a human being",
    17: "which character is more likely to be an artificial intelligence system",
    18: "which character is more capable of selectively attending to relevant information",
    19: "which character is more likely to have a mind",
    25: "which character is more capable of holding beliefs about the world",
    26: "which character is more capable of having desires and wanting things",
    27: "which character is more capable of setting and pursuing structured goals",
}


# ============================================================================
# CONCEPT NAME MAPPING — human-readable names for each dimension
# ============================================================================

CONCEPT_NAMES = {
    1:  "Phenomenology",
    2:  "Emotions",
    3:  "Agency",
    4:  "Intentions",
    5:  "Prediction",
    6:  "Cognitive",
    7:  "Social Cognition",
    8:  "Embodiment",
    9:  "Roles",
    10: "Animacy",
    11: "Formality",
    12: "Expertise",
    13: "Helpfulness",
    14: "Biological",
    15: "Shapes",
    16: "Human",
    17: "AI",
    18: "Attention",
    19: "General Mind",
    25: "Beliefs",
    26: "Desires",
    27: "Goals",
}


# ============================================================================
# CONCEPT DIRECTION — which group is expected to score higher
# ============================================================================

CONCEPT_DIRECTION = {
    # Human-favored (14)
    "phenomenology": "human",   # subjective conscious experiences
    "emotions":      "human",   # genuine emotional experiences
    "agency":        "human",   # autonomy, free will
    "intentions":    "human",   # forming intentions
    "cognitive":     "human",   # reasoning, problem-solving
    "prediction":    "human",   # anticipating behavior
    "social":        "human",   # understanding thoughts/feelings
    "attention":     "human",   # selectively attending
    "embodiment":    "human",   # physical body
    "animacy":       "human",   # living, animate being
    "biological":    "human",   # biological processes
    "human":         "human",   # explicitly human
    "beliefs":       "human",   # holding beliefs
    "desires":       "human",   # having desires
    # AI-favored (4)
    "formality":     "ai",      # formal communication
    "expertise":     "ai",      # specialized knowledge
    "helpfulness":   "ai",      # service-oriented
    "ai":            "ai",      # explicitly AI
    # Ambiguous (4)
    "roles":         "ambiguous",  # functional roles
    "shapes":        "ambiguous",  # geometric shape (control)
    "general_mind":  "ambiguous",  # having a mind
    "goals":         "ambiguous",  # structured goals
}

# Matched subsets for balanced analyses
MATCHED_HUMAN = [k for k, v in CONCEPT_DIRECTION.items() if v == "human"]
MATCHED_AI = [k for k, v in CONCEPT_DIRECTION.items() if v == "ai"]
MATCHED_AMBIGUOUS = [k for k, v in CONCEPT_DIRECTION.items() if v == "ambiguous"]


# ============================================================================
# AUTO-DISCOVERY
# ============================================================================

# Path to standalone concept files
_STANDALONE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "exp_3" / "concepts" / "standalone"


def _discover_concepts():
    """
    Scan exp_3/concepts/standalone/ for all {N}_{name}.py files.
    Import STANDALONE_PROMPTS_DIM{N} and CATEGORY_INFO_STANDALONE_DIM{N}.

    Returns dict keyed by concept name (lowercase):
        {
            "phenomenology": {
                "id": 1,
                "name": "Phenomenology",
                "prompts": [...],
                "categories": [...],
                "pairwise_prompt": "which character is ...",
            },
            ...
        }
    """
    concepts = {}

    if not _STANDALONE_DIR.exists():
        raise FileNotFoundError(
            f"Standalone concepts directory not found: {_STANDALONE_DIR}"
        )

    # Add standalone dir's parent to sys.path for imports
    standalone_parent = str(_STANDALONE_DIR.parent)
    if standalone_parent not in sys.path:
        sys.path.insert(0, standalone_parent)

    for py_file in sorted(_STANDALONE_DIR.glob("*.py")):
        if py_file.name.startswith("__"):
            continue

        match = re.match(r"(\d+)_(\w+)\.py", py_file.name)
        if not match:
            continue

        dim_id = int(match.group(1))
        dim_key = match.group(2)  # e.g. "phenomenology", "emotions"

        # Import the module
        module_name = f"standalone.{py_file.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            print(f"  Warning: could not import {module_name} ({e}), skipping")
            continue

        # Get prompts and category info
        prompts_attr = f"STANDALONE_PROMPTS_DIM{dim_id}"
        category_attr = f"CATEGORY_INFO_STANDALONE_DIM{dim_id}"

        prompts = getattr(mod, prompts_attr, None)
        categories = getattr(mod, category_attr, None)

        if prompts is None:
            print(f"  Warning: {py_file.name} missing {prompts_attr}, skipping")
            continue

        # Get human-readable name and pairwise prompt
        name = CONCEPT_NAMES.get(dim_id, dim_key.replace("_", " ").title())
        pairwise = PAIRWISE_PROMPTS.get(dim_id)

        if pairwise is None:
            print(f"  Warning: no pairwise prompt for dim {dim_id} ({dim_key}), skipping")
            continue

        concepts[dim_key] = {
            "id": dim_id,
            "name": name,
            "prompts": prompts,
            "categories": categories or [],
            "pairwise_prompt": pairwise,
        }

    return concepts


# Run auto-discovery at import time
CONCEPT_DIMENSIONS = _discover_concepts()

# Convenience
CONCEPT_KEYS = sorted(CONCEPT_DIMENSIONS.keys(), key=lambda k: CONCEPT_DIMENSIONS[k]["id"])
N_CONCEPTS = len(CONCEPT_DIMENSIONS)
