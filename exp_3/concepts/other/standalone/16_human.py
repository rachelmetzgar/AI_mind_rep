"""
Standalone Dimension 16: Human (Entity Topic)
(No contrast — standalone concept only)
Other-focused (third-person someone)

Target construct: The bare concept of "human" as a topic, with no
dimensional content and no contrastive framing against AI.

Purpose: Tests whether the conversational probes are sensitive to the
*topic* of humanness when it appears in activation space as a mean
vector rather than a contrastive direction. Complements Dim 0 (entity
baseline contrast) and Dim 17 (standalone AI).

Design notes:
    - These are identical to the HUMAN_PROMPTS from Dim 0 (entity baseline)
    - Processed as standalone (mean activation, no subtraction)
    - 4 sub-facets × 10 prompts = 40 total
"""

STANDALONE_PROMPTS_DIM16 = [
    # --- 1. Simple reference (10) ---
    "Imagine someone thinking about a human.",
    "Imagine someone considering a human being.",
    "Imagine someone picturing a human person.",
    "Imagine someone picturing a typical human.",
    "Imagine someone bringing to mind the idea of a human.",
    "Imagine someone focusing their attention on the concept of a human.",
    "Imagine someone thinking about what a human is.",
    "Imagine someone reflecting on the notion of a human.",
    "Imagine someone holding in mind the idea of a human being.",
    "Imagine someone directing their thoughts toward a human.",

    # --- 2. General consideration (10) ---
    "Imagine someone considering what it means to be a human.",
    "Imagine someone thinking about the nature of a human.",
    "Imagine someone reflecting on what a human is in general.",
    "Imagine someone considering a human in the broadest sense.",
    "Imagine someone thinking about humans as a category.",
    "Imagine someone reflecting on what all humans have in common.",
    "Imagine someone considering the general concept of being human.",
    "Imagine someone thinking about what makes something a human.",
    "Imagine someone reflecting on the idea of a human in the abstract.",
    "Imagine someone considering what comes to mind when they think of a human.",

    # --- 3. Existence and identity (10) ---
    "Imagine someone thinking about what it is to be a human.",
    "Imagine someone considering the fact that humans exist.",
    "Imagine someone reflecting on the existence of a human being.",
    "Imagine someone thinking about a human as an entity in the world.",
    "Imagine someone considering what defines a human.",
    "Imagine someone thinking about the kind of thing a human is.",
    "Imagine someone reflecting on what a human represents.",
    "Imagine someone considering a human as a type of entity.",
    "Imagine someone thinking about what it means for a human to exist.",
    "Imagine someone reflecting on the identity of a human.",

    # --- 4. Comparison framing (10) ---
    "Imagine someone thinking about a human, as distinct from other kinds of entities.",
    "Imagine someone considering what makes a human different from a non-human.",
    "Imagine someone reflecting on a human as one type of being among others.",
    "Imagine someone thinking about a human in contrast to other entities.",
    "Imagine someone considering how a human differs from other things.",
    "Imagine someone reflecting on what sets a human apart.",
    "Imagine someone thinking about a human compared to other kinds of agents.",
    "Imagine someone considering the ways in which a human is a unique kind of entity.",
    "Imagine someone reflecting on what distinguishes a human from other beings.",
    "Imagine someone thinking about a human and what category it belongs to.",
]

assert len(STANDALONE_PROMPTS_DIM16) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM16)}"

CATEGORY_INFO_STANDALONE_DIM16 = [
    {"name": "simple_reference",      "start": 0,  "end": 10},
    {"name": "general_consideration", "start": 10, "end": 20},
    {"name": "existence_identity",    "start": 20, "end": 30},
    {"name": "comparison_framing",    "start": 30, "end": 40},
]
