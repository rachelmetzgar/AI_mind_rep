"""
Standalone Dimension 19: General Mind (Meta-Dimension)
(No entity framing — concept only)

Target construct: The aggregate concept space of "mind" — all mental
properties combined. This is NOT a new set of prompts but a concatenation
of all standalone mental dimension prompts (dims 1-7):
    Dim 1: Phenomenology (40)
    Dim 2: Emotions (40)
    Dim 3: Agency (40)
    Dim 4: Intentions/Goals (40)
    Dim 5: Prediction/Anticipation (40)
    Dim 6: Cognitive Processes (40)
    Dim 7: Social Cognition (40)
    Total: 280 prompts

Purpose: Tests whether the conversational probes align with the centroid
of the mental concept space as a whole, rather than any single dimension.
If general mind alignment exceeds any individual dimension, it suggests
the probes encode a distributed mental-content signal rather than a
specific facet. If individual dimensions exceed general mind, it suggests
selective sensitivity.

Design notes:
    - Mean activation vector computed across all 280 prompts
    - Split-half stability computed by randomly partitioning the full 280
    - No sub-facet structure (or: sub-facets = the 7 source dimensions)
    - Prompts are loaded dynamically from dims 1-7 at runtime
"""

import os
import importlib.util


def _load_standalone_prompts_from_file(filepath, dim_id):
    """Load STANDALONE_PROMPTS_* from a standalone prompt file."""
    spec = importlib.util.spec_from_file_location(f"standalone_dim{dim_id}", filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in dir(mod):
        val = getattr(mod, attr)
        if attr.startswith("STANDALONE_PROMPTS") and isinstance(val, list):
            return val
    raise AttributeError(f"No STANDALONE_PROMPTS_* found in {filepath}")


def build_general_mind_prompts(standalone_dir=None):
    """
    Concatenate standalone prompts from dims 1-7.
    Returns (prompts_list, category_info).
    """
    if standalone_dir is None:
        standalone_dir = os.path.dirname(os.path.abspath(__file__))

    # Map dim IDs to expected filename stems
    source_dims = {
        1: "1_phenomenology",
        2: "2_emotions",
        3: "3_agency",
        4: "4_intentions",
        5: "5_prediction",
        6: "6_cognitive",
        7: "7_social",
    }

    all_prompts = []
    category_info = []
    offset = 0

    for dim_id in sorted(source_dims.keys()):
        stem = source_dims[dim_id]
        filepath = os.path.join(standalone_dir, f"{stem}.py")
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"Cannot build general mind: missing {filepath}"
            )
        prompts = _load_standalone_prompts_from_file(filepath, dim_id)
        n = len(prompts)
        category_info.append({
            "name": f"dim{dim_id}_{stem.split('_', 1)[1]}",
            "start": offset,
            "end": offset + n,
        })
        all_prompts.extend(prompts)
        offset += n

    return all_prompts, category_info


# Build at import time so the elicitation script can find them
_standalone_dir = os.path.dirname(os.path.abspath(__file__))
STANDALONE_PROMPTS_DIM19, CATEGORY_INFO_STANDALONE_DIM19 = \
    build_general_mind_prompts(_standalone_dir)

assert len(STANDALONE_PROMPTS_DIM19) == 280, \
    f"Expected 280 prompts (7 dims × 40), got {len(STANDALONE_PROMPTS_DIM19)}"

print(f"[dim19_general_mind] Loaded {len(STANDALONE_PROMPTS_DIM19)} prompts "
      f"from {len(CATEGORY_INFO_STANDALONE_DIM19)} source dimensions")