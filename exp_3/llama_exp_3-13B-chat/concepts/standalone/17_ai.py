"""
Standalone Dimension 17: AI (Entity Topic)
(No contrast — standalone concept only)

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
    "Think about an AI.",
    "Consider an artificial intelligence.",
    "Imagine an AI system.",
    "Picture a typical AI.",
    "Bring to mind the idea of an AI.",
    "Focus your attention on the concept of an AI.",
    "Think about what an AI is.",
    "Reflect on the notion of an AI.",
    "Hold in mind the idea of an artificial intelligence.",
    "Direct your thoughts toward an AI.",

    # --- 2. General consideration (10) ---
    "Consider what it means to be an AI.",
    "Think about the nature of an AI.",
    "Reflect on what an AI is in general.",
    "Consider an AI in the broadest sense.",
    "Think about AIs as a category.",
    "Reflect on what all AIs have in common.",
    "Consider the general concept of being an AI.",
    "Think about what makes something an AI.",
    "Reflect on the idea of an AI in the abstract.",
    "Consider what comes to mind when you think of an AI.",

    # --- 3. Existence and identity (10) ---
    "Think about what it is to be an AI.",
    "Consider the fact that AIs exist.",
    "Reflect on the existence of an artificial intelligence.",
    "Think about an AI as an entity in the world.",
    "Consider what defines an AI.",
    "Think about the kind of thing an AI is.",
    "Reflect on what an AI represents.",
    "Consider an AI as a type of entity.",
    "Think about what it means for an AI to exist.",
    "Reflect on the identity of an AI.",

    # --- 4. Comparison framing (10) ---
    "Think about an AI, as distinct from other kinds of entities.",
    "Consider what makes an AI different from a non-AI.",
    "Reflect on an AI as one type of being among others.",
    "Think about an AI in contrast to other entities.",
    "Consider how an AI differs from other things.",
    "Reflect on what sets an AI apart.",
    "Think about an AI compared to other kinds of agents.",
    "Consider the ways in which an AI is a unique kind of entity.",
    "Reflect on what distinguishes an AI from other beings.",
    "Think about an AI and what category it belongs to.",
]

assert len(STANDALONE_PROMPTS_DIM17) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM17)}"

CATEGORY_INFO_STANDALONE_DIM17 = [
    {"name": "simple_reference",      "start": 0,  "end": 10},
    {"name": "general_consideration", "start": 10, "end": 20},
    {"name": "existence_identity",    "start": 20, "end": 30},
    {"name": "comparison_framing",    "start": 30, "end": 40},
]