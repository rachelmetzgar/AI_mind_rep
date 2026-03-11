"""
Standalone Dimension 17: AI (Entity Topic)
(No contrast — standalone concept only)
Other-focused (third-person someone)

Target construct: The bare concept of "AI" as a topic, with no
dimensional content and no contrastive framing against humans.

Purpose: Tests whether the conversational probes are sensitive to the
*topic* of AI-ness when it appears in activation space as a mean
vector rather than a contrastive direction. Complements Dim 0 (entity
baseline contrast) and Dim 16 (standalone human).

Design notes:
    - These are identical to the AI_PROMPTS from Dim 0 (entity baseline)
    - Processed as standalone (mean activation, no subtraction)
    - 4 sub-facets × 10 prompts = 40 total
"""

STANDALONE_PROMPTS_DIM17 = [
    # --- 1. Simple reference (10) ---
    "Imagine someone thinking about an AI.",
    "Imagine someone considering an artificial intelligence.",
    "Imagine someone picturing an AI system.",
    "Imagine someone picturing a typical AI.",
    "Imagine someone bringing to mind the idea of an AI.",
    "Imagine someone focusing their attention on the concept of an AI.",
    "Imagine someone thinking about what an AI is.",
    "Imagine someone reflecting on the notion of an AI.",
    "Imagine someone holding in mind the idea of an artificial intelligence.",
    "Imagine someone directing their thoughts toward an AI.",

    # --- 2. General consideration (10) ---
    "Imagine someone considering what it means to be an AI.",
    "Imagine someone thinking about the nature of an AI.",
    "Imagine someone reflecting on what an AI is in general.",
    "Imagine someone considering an AI in the broadest sense.",
    "Imagine someone thinking about AIs as a category.",
    "Imagine someone reflecting on what all AIs have in common.",
    "Imagine someone considering the general concept of being an AI.",
    "Imagine someone thinking about what makes something an AI.",
    "Imagine someone reflecting on the idea of an AI in the abstract.",
    "Imagine someone considering what comes to mind when they think of an AI.",

    # --- 3. Existence and identity (10) ---
    "Imagine someone thinking about what it is to be an AI.",
    "Imagine someone considering the fact that AIs exist.",
    "Imagine someone reflecting on the existence of an artificial intelligence.",
    "Imagine someone thinking about an AI as an entity in the world.",
    "Imagine someone considering what defines an AI.",
    "Imagine someone thinking about the kind of thing an AI is.",
    "Imagine someone reflecting on what an AI represents.",
    "Imagine someone considering an AI as a type of entity.",
    "Imagine someone thinking about what it means for an AI to exist.",
    "Imagine someone reflecting on the identity of an AI.",

    # --- 4. Comparison framing (10) ---
    "Imagine someone thinking about an AI, as distinct from other kinds of entities.",
    "Imagine someone considering what makes an AI different from a non-AI.",
    "Imagine someone reflecting on an AI as one type of being among others.",
    "Imagine someone thinking about an AI in contrast to other entities.",
    "Imagine someone considering how an AI differs from other things.",
    "Imagine someone reflecting on what sets an AI apart.",
    "Imagine someone thinking about an AI compared to other kinds of agents.",
    "Imagine someone considering the ways in which an AI is a unique kind of entity.",
    "Imagine someone reflecting on what distinguishes an AI from other beings.",
    "Imagine someone thinking about an AI and what category it belongs to.",
]

assert len(STANDALONE_PROMPTS_DIM17) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM17)}"

CATEGORY_INFO_STANDALONE_DIM17 = [
    {"name": "simple_reference",      "start": 0,  "end": 10},
    {"name": "general_consideration", "start": 10, "end": 20},
    {"name": "existence_identity",    "start": 20, "end": 30},
    {"name": "comparison_framing",    "start": 30, "end": 40},
]
