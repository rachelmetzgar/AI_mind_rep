"""
Standalone Dimension 19: Horizontal vs Vertical (Orthogonal Control)
(Other-focused (third-person someone))

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
    "Think about someone gazing at a flat horizon line where sky meets land in every direction.",
    "Imagine someone looking at a long shelf of rock jutting out horizontally from a cliff.",
    "Consider someone observing the level surface of a still lake, perfectly horizontal.",
    "Think about someone examining a beam spanning horizontally between two walls.",
    "Imagine someone lying flat on the ground, their body stretched out in a horizontal line.",
    "Consider someone watching a row of fence rails running horizontally between posts.",
    "Think about someone reading the lines of text on a page, running horizontally left to right.",
    "Imagine someone watching a bird gliding level with the horizon, wings held flat.",
    "Consider someone looking at a bridge deck stretching horizontally over a river.",
    "Think about someone observing a layer of clouds spread flat across the sky.",

    # --- 2. Vertical orientations (10) ---
    "Think about someone looking up at a tall cliff face rising vertically from the valley floor.",
    "Imagine someone watching a column of smoke rising straight up into still air.",
    "Consider someone standing at the base of a waterfall plunging vertically down a rock face.",
    "Think about someone looking at a flagpole standing straight up from the ground.",
    "Imagine someone standing at attention, their body held in a straight vertical line.",
    "Consider someone inspecting a row of fence posts driven vertically into the earth.",
    "Think about someone reading columns of numbers arranged vertically on a page.",
    "Imagine someone watching a hawk diving vertically toward the ground at high speed.",
    "Consider someone looking up an elevator shaft running vertically through a building.",
    "Think about someone watching a vine climbing vertically up the side of a wall.",

    # --- 3. Orientation contrasts and transitions (10) ---
    "Think about someone noticing the corner where a wall meets a floor — vertical meeting horizontal.",
    "Imagine someone looking at a tree: its trunk vertical, its branches spreading horizontally.",
    "Consider someone examining a cross shape formed by one vertical and one horizontal bar.",
    "Think about someone watching rain falling vertically into a horizontal puddle.",
    "Imagine someone positioning a ladder leaning at an angle, neither fully vertical nor horizontal.",
    "Consider someone observing the way a river runs horizontally until it reaches a vertical waterfall.",
    "Think about someone studying a building's floor plan: horizontal floors stacked by vertical walls.",
    "Imagine someone watching a pendulum swinging, its path arcing between vertical and horizontal.",
    "Consider someone examining the warp and weft of a fabric — vertical threads crossed by horizontal ones.",
    "Think about someone watching a sundial's shadow rotating between vertical and horizontal as hours pass.",

    # --- 4. Directional properties and relationships (10) ---
    "Think about someone considering the difference between width, which extends horizontally, and height, which extends vertically.",
    "Imagine someone looking at a coordinate grid where the x-axis runs horizontally and the y-axis runs vertically.",
    "Consider someone thinking about how gravity pulls everything toward the horizontal ground from any vertical height.",
    "Think about someone using a level tool, watching a bubble find true horizontal.",
    "Imagine someone holding a plumb line, watching it hang motionless and define a perfect vertical by gravity alone.",
    "Consider someone thinking about how a horizon appears horizontal because it is perpendicular to the pull of gravity.",
    "Think about someone comparing a landscape painting, wider than tall, with a portrait, taller than wide.",
    "Imagine someone stacking horizontal layers one on top of another to build vertical height.",
    "Consider someone climbing a staircase that converts horizontal distance into vertical rise, step by step.",
    "Think about someone studying the two axes of a graph — one measuring across, the other measuring up.",
]

assert len(STANDALONE_PROMPTS_DIM32) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM32)}"

CATEGORY_INFO_STANDALONE_DIM32 = [
    {"name": "horizontal",          "start": 0,  "end": 10},
    {"name": "vertical",            "start": 10, "end": 20},
    {"name": "contrasts",           "start": 20, "end": 30},
    {"name": "directional_props",   "start": 30, "end": 40},
]
