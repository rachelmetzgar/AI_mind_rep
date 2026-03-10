"""
Standalone Dimension 19: Horizontal vs Vertical (Orthogonal Control)
(No entity framing — concept only)

Target construct: Spatial orientation — horizontal and vertical axes
applied to natural features, structures, movement, and abstract patterns.

Purpose: Pipeline validation control. If this dimension shows meaningful
alignment with conversational probes, the pipeline is generating artifacts.

Design notes:
    - No "human," "AI," or other entity-type language
    - Prompts cover horizontal and vertical orientation equally
    - Same reflective prompt format as all other dimensions
    - Same number of prompts (40) and sub-facet structure (4 x 10)

4 sub-facets x 10 prompts = 40 total.

Sub-facets:
    1. Horizontal orientations
    2. Vertical orientations
    3. Orientation contrasts and transitions
    4. Directional properties and relationships
"""

STANDALONE_PROMPTS_DIM32 = [
    # --- 1. Horizontal orientations (10) ---
    "Think about a flat horizon line where sky meets land in every direction.",
    "Imagine a long shelf of rock jutting out horizontally from a cliff.",
    "Consider the level surface of a still lake, perfectly horizontal.",
    "Think about a beam spanning horizontally between two walls.",
    "Imagine lying flat on the ground, your body stretched out in a horizontal line.",
    "Consider a row of fence rails running horizontally between posts.",
    "Think about the lines of text on a page, running horizontally left to right.",
    "Imagine a bird gliding level with the horizon, wings held flat.",
    "Consider a bridge deck stretching horizontally over a river.",
    "Think about a layer of clouds spread flat across the sky.",

    # --- 2. Vertical orientations (10) ---
    "Think about a tall cliff face rising vertically from the valley floor.",
    "Imagine a column of smoke rising straight up into still air.",
    "Consider a waterfall plunging vertically down a rock face.",
    "Think about a flagpole standing straight up from the ground.",
    "Imagine standing at attention, your body held in a straight vertical line.",
    "Consider a row of fence posts driven vertically into the earth.",
    "Think about columns of numbers arranged vertically on a page.",
    "Imagine a hawk diving vertically toward the ground at high speed.",
    "Consider an elevator shaft running vertically through a building.",
    "Think about a vine climbing vertically up the side of a wall.",

    # --- 3. Orientation contrasts and transitions (10) ---
    "Think about the corner where a wall meets a floor — vertical meeting horizontal.",
    "Imagine a tree: its trunk vertical, its branches spreading horizontally.",
    "Consider a cross shape formed by one vertical and one horizontal bar.",
    "Think about rain falling vertically into a horizontal puddle.",
    "Imagine a ladder leaning at an angle, neither fully vertical nor horizontal.",
    "Consider the way a river runs horizontally until it reaches a vertical waterfall.",
    "Think about a building's floor plan: horizontal floors stacked by vertical walls.",
    "Imagine a pendulum swinging, its path arcing between vertical and horizontal.",
    "Consider the warp and weft of a fabric — vertical threads crossed by horizontal ones.",
    "Think about a sundial's shadow rotating between vertical and horizontal as hours pass.",

    # --- 4. Directional properties and relationships (10) ---
    "Think about the difference between width, which extends horizontally, and height, which extends vertically.",
    "Imagine a coordinate grid where the x-axis runs horizontally and the y-axis runs vertically.",
    "Consider how gravity pulls everything toward the horizontal ground from any vertical height.",
    "Think about the way a level tool uses a bubble to find true horizontal.",
    "Imagine a plumb line hanging motionless, defining a perfect vertical by gravity alone.",
    "Consider how a horizon appears horizontal because it is perpendicular to the pull of gravity.",
    "Think about the difference between a landscape painting, wider than tall, and a portrait, taller than wide.",
    "Imagine stacking horizontal layers one on top of another to build vertical height.",
    "Consider how a staircase converts horizontal distance into vertical rise, step by step.",
    "Think about the two axes of a graph — one measuring across, the other measuring up.",
]

assert len(STANDALONE_PROMPTS_DIM32) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM32)}"

CATEGORY_INFO_STANDALONE_DIM32 = [
    {"name": "horizontal",          "start": 0,  "end": 10},
    {"name": "vertical",            "start": 10, "end": 20},
    {"name": "contrasts",           "start": 20, "end": 30},
    {"name": "directional_props",   "start": 30, "end": 40},
]
