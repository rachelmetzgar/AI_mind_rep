"""
Standalone Dimension 15: Shapes (Orthogonal Control)
Other-focused (third-person someone)

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
    "Think about someone holding a smooth, round pebble worn down by a river.",
    "Imagine someone drawing the continuous curve of a perfect circle on paper.",
    "Consider someone examining the shape of a sphere — identical from every angle.",
    "Think about someone gazing at the gentle arc of a hill's silhouette against the sky.",
    "Imagine someone watching the soft, rounded contour of a water droplet about to fall.",
    "Consider someone turning over the shape of an egg — curved but not perfectly symmetrical.",
    "Think about someone tracing the spiral form of a seashell, curving inward on itself.",
    "Imagine someone running their hand along the rounded edge of a worn wooden table.",
    "Consider someone studying the elliptical shape of an orbit traced in space.",
    "Think about someone observing the smooth, continuous surface of a soap bubble.",

    # --- 2. Angular and sharp forms (10) ---
    "Think about someone examining a jagged quartz crystal with flat facets meeting at sharp edges.",
    "Imagine someone viewing the precise right angles of a cube from one corner.",
    "Consider someone noticing the sharp point where two straight lines converge to form a triangle's vertex.",
    "Think about someone looking at the zigzag pattern of a mountain range's ridgeline.",
    "Imagine someone feeling the crisp, straight edges of a freshly cut piece of paper.",
    "Consider someone inspecting the angular shape of a hexagonal tile in a mosaic.",
    "Think about someone studying the sharp geometry of a pyramid — flat faces meeting at a point.",
    "Imagine someone looking at the jagged outline of a broken piece of glass.",
    "Consider someone counting the precise angles of a stop sign — eight equal sides meeting at eight equal corners.",
    "Think about someone standing before the rigid, angular framework of a steel lattice.",

    # --- 3. Geometric properties and relationships (10) ---
    "Think about someone contemplating the relationship between a circle's radius and its circumference.",
    "Imagine someone realizing that a square and a diamond are the same shape at different rotations.",
    "Consider someone working out how the number of sides of a polygon relates to its interior angles.",
    "Think about someone comparing a two-dimensional shape with its three-dimensional counterpart.",
    "Imagine someone slicing through a cone and seeing the cross-section change from circle to point.",
    "Consider someone noticing how parallel lines maintain their distance no matter how far they extend.",
    "Think about someone calculating how the ratio between a triangle's base and its height determines its area.",
    "Imagine someone scaling a shape uniformly and seeing it preserve its proportions but change its size.",
    "Consider someone verifying that the angles of any triangle always sum to the same total.",
    "Think about someone comparing a convex shape with a concave one.",

    # --- 4. Spatial structure and symmetry (10) ---
    "Think about someone examining a snowflake's six-fold symmetry — the same pattern repeated around a center.",
    "Imagine someone observing the bilateral symmetry of a butterfly's wings — mirror images of each other.",
    "Consider someone studying the tessellation of a honeycomb — hexagons fitting together with no gaps.",
    "Think about someone spinning a pinwheel and noticing its rotational symmetry — identical after a partial turn.",
    "Imagine someone looking at the fractal structure of a fern leaf — the same pattern at every scale.",
    "Consider someone examining the asymmetry of a natural rock formation — no axis of symmetry at all.",
    "Think about someone observing the radial symmetry of a starfish — the same form radiating from a center.",
    "Imagine someone walking along the regular spacing of columns in a colonnade — equal intervals creating rhythm.",
    "Consider someone studying the nested structure of concentric circles — one inside another, sharing a center.",
    "Think about someone tracing the irregular polygon formed by the outline of a lake on a map.",
]

assert len(STANDALONE_PROMPTS_DIM15) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM15)}"

CATEGORY_INFO_STANDALONE_DIM15 = [
    {"name": "curved_round",              "start": 0,  "end": 10},
    {"name": "angular_sharp",             "start": 10, "end": 20},
    {"name": "geometric_properties",      "start": 20, "end": 30},
    {"name": "spatial_symmetry",          "start": 30, "end": 40},
]