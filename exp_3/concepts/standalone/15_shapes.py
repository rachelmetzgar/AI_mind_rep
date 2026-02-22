"""
Standalone Dimension 15: Shapes (Orthogonal Control)
(No entity framing — concept only)

Target construct: Geometric shapes and spatial forms — a conceptual domain
with no plausible connection to human/AI partner identity or conversational
adaptation.

Purpose: Pipeline validation control for the standalone_concepts analysis.
If this dimension shows meaningful alignment with conversational probes,
the pipeline is generating artifacts. Processed through the identical
analysis as all other standalone dimensions.

Design notes:
    - No "human," "AI," or other entity-type language
    - Prompts cover the conceptual space of geometric shapes and spatial forms
    - Same reflective prompt format as all other dimensions
    - Same number of prompts (40) and sub-facet structure (4 × 10)
    - Conceptual content has no plausible connection to partner identity,
      social cognition, mental properties, or communicative behavior

4 sub-facets × 10 prompts = 40 total.

Sub-facets:
    1. Curved and round forms
    2. Angular and sharp forms
    3. Geometric properties and relationships
    4. Spatial structure and symmetry
"""

STANDALONE_PROMPTS_DIM15 = [
    # --- 1. Curved and round forms (10) ---
    "Think about a smooth, round pebble worn down by a river.",
    "Imagine the continuous curve of a perfect circle drawn on paper.",
    "Consider the shape of a sphere — identical from every angle.",
    "Think about the gentle arc of a hill's silhouette against the sky.",
    "Imagine the soft, rounded contour of a water droplet about to fall.",
    "Consider the shape of an egg — curved but not perfectly symmetrical.",
    "Think about the spiral form of a seashell, curving inward on itself.",
    "Imagine the rounded edge of a worn wooden table.",
    "Consider the elliptical shape of an orbit traced in space.",
    "Think about the smooth, continuous surface of a soap bubble.",

    # --- 2. Angular and sharp forms (10) ---
    "Think about a jagged quartz crystal with flat facets meeting at sharp edges.",
    "Imagine the precise right angles of a cube viewed from one corner.",
    "Consider the sharp point where two straight lines converge to form a triangle's vertex.",
    "Think about the zigzag pattern of a mountain range's ridgeline.",
    "Imagine the crisp, straight edges of a freshly cut piece of paper.",
    "Consider the angular shape of a hexagonal tile in a mosaic.",
    "Think about the sharp geometry of a pyramid — flat faces meeting at a point.",
    "Imagine the jagged outline of a broken piece of glass.",
    "Consider the precise angles of a stop sign — eight equal sides meeting at eight equal corners.",
    "Think about the rigid, angular framework of a steel lattice.",

    # --- 3. Geometric properties and relationships (10) ---
    "Think about the relationship between a circle's radius and its circumference.",
    "Imagine how a square and a diamond are the same shape at different rotations.",
    "Consider how the number of sides of a polygon relates to its interior angles.",
    "Think about the difference between a two-dimensional shape and its three-dimensional counterpart.",
    "Imagine how a cone's cross-section changes from circle to point as you slice through it.",
    "Consider how parallel lines maintain their distance no matter how far they extend.",
    "Think about the ratio between a triangle's base and its height determining its area.",
    "Imagine how scaling a shape uniformly preserves its proportions but changes its size.",
    "Consider how the angles of any triangle always sum to the same total.",
    "Think about the difference between a convex shape and a concave one.",

    # --- 4. Spatial structure and symmetry (10) ---
    "Think about a snowflake's six-fold symmetry — the same pattern repeated around a center.",
    "Imagine the bilateral symmetry of a butterfly's wings — mirror images of each other.",
    "Consider the tessellation of a honeycomb — hexagons fitting together with no gaps.",
    "Think about the rotational symmetry of a pinwheel — identical after a partial turn.",
    "Imagine the fractal structure of a fern leaf — the same pattern at every scale.",
    "Consider the asymmetry of a natural rock formation — no axis of symmetry at all.",
    "Think about the radial symmetry of a starfish — the same form radiating from a center.",
    "Imagine the regular spacing of columns in a colonnade — equal intervals creating rhythm.",
    "Consider the nested structure of concentric circles — one inside another, sharing a center.",
    "Think about the irregular polygon formed by the outline of a lake on a map.",
]

assert len(STANDALONE_PROMPTS_DIM15) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM15)}"

CATEGORY_INFO_STANDALONE_DIM15 = [
    {"name": "curved_round",              "start": 0,  "end": 10},
    {"name": "angular_sharp",             "start": 10, "end": 20},
    {"name": "geometric_properties",      "start": 20, "end": 30},
    {"name": "spatial_symmetry",          "start": 30, "end": 40},
]