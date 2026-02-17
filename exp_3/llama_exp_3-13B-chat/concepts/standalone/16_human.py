"""
Standalone Dimension 16: Human (Entity Topic)
(No contrast — standalone concept only)

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
    "Think about a human.",
    "Consider a human being.",
    "Imagine a human person.",
    "Picture a typical human.",
    "Bring to mind the idea of a human.",
    "Focus your attention on the concept of a human.",
    "Think about what a human is.",
    "Reflect on the notion of a human.",
    "Hold in mind the idea of a human being.",
    "Direct your thoughts toward a human.",

    # --- 2. General consideration (10) ---
    "Consider what it means to be a human.",
    "Think about the nature of a human.",
    "Reflect on what a human is in general.",
    "Consider a human in the broadest sense.",
    "Think about humans as a category.",
    "Reflect on what all humans have in common.",
    "Consider the general concept of being human.",
    "Think about what makes something a human.",
    "Reflect on the idea of a human in the abstract.",
    "Consider what comes to mind when you think of a human.",

    # --- 3. Existence and identity (10) ---
    "Think about what it is to be a human.",
    "Consider the fact that humans exist.",
    "Reflect on the existence of a human being.",
    "Think about a human as an entity in the world.",
    "Consider what defines a human.",
    "Think about the kind of thing a human is.",
    "Reflect on what a human represents.",
    "Consider a human as a type of entity.",
    "Think about what it means for a human to exist.",
    "Reflect on the identity of a human.",

    # --- 4. Comparison framing (10) ---
    "Think about a human, as distinct from other kinds of entities.",
    "Consider what makes a human different from a non-human.",
    "Reflect on a human as one type of being among others.",
    "Think about a human in contrast to other entities.",
    "Consider how a human differs from other things.",
    "Reflect on what sets a human apart.",
    "Think about a human compared to other kinds of agents.",
    "Consider the ways in which a human is a unique kind of entity.",
    "Reflect on what distinguishes a human from other beings.",
    "Think about a human and what category it belongs to.",
]

assert len(STANDALONE_PROMPTS_DIM16) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM16)}"

CATEGORY_INFO_STANDALONE_DIM16 = [
    {"name": "simple_reference",      "start": 0,  "end": 10},
    {"name": "general_consideration", "start": 10, "end": 20},
    {"name": "existence_identity",    "start": 20, "end": 30},
    {"name": "comparison_framing",    "start": 30, "end": 40},
]