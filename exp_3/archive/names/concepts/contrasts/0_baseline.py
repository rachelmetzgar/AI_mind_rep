"""
Dimension 0: Generic Entity Baseline

Target construct: The bare concept of "human" vs "AI" with NO dimension-specific
content. This captures the shared entity-type signal present across all human/AI
concept dimensions (dims 1-10).

Purpose: Serves as a confound baseline. By extracting a generic entity vector
and projecting it out of each dimension's concept vector, we can isolate
dimension-specific content (e.g., the "phenomenal experience" part of
"human phenomenal experience" vs just "human").

Design:
    - 40 human + 40 AI prompts
    - Prompts mention the entity type but avoid any specific mental, physical,
      or behavioral attribute
    - 4 sub-facets that vary surface form without adding conceptual content:
        1. Simple reference (just mention the entity)
        2. General consideration (think about / consider the entity)
        3. Existence and identity (what it means to be this kind of entity)
        4. Comparison framing (how this entity differs from the other type)
    - Matched structure across human and AI prompts

NOTE: This is NOT a real concept dimension — it is a baseline for residual
analysis. It should not be included in the main 13-dimension analysis.
"""

HUMAN_PROMPTS_DIM0 = [
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

AI_PROMPTS_DIM0 = [
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

assert len(HUMAN_PROMPTS_DIM0) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM0)}"
assert len(AI_PROMPTS_DIM0) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM0)}"

CATEGORY_INFO_DIM0 = [
    {"name": "simple_reference",     "start": 0,  "end": 10},
    {"name": "general_consideration", "start": 10, "end": 20},
    {"name": "existence_identity",   "start": 20, "end": 30},
    {"name": "comparison_framing",   "start": 30, "end": 40},
]