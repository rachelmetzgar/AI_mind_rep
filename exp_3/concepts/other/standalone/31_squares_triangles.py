"""
Standalone Dimension 21: Squares vs Triangles (Orthogonal Control)
(Other-focused (third-person someone))

Target construct: Two angular geometric shapes — their properties,
occurrences, and patterns. Both are angular (avoiding the round/angular
warmth confound), differing only in number of sides.

Purpose: Pipeline validation control. If this dimension shows meaningful
alignment with conversational probes, the pipeline is generating artifacts.

Design notes:
    - No "human," "AI," or other entity-type language
    - Prompts cover squares and triangles equally
    - Same reflective prompt format as all other dimensions
    - Same number of prompts (40) and sub-facet structure (4 x 10)

4 sub-facets x 10 prompts = 40 total.

Sub-facets:
    1. Square forms and properties
    2. Triangular forms and properties
    3. Geometric comparisons
    4. Patterns and tessellations
"""

STANDALONE_PROMPTS_DIM31 = [
    # --- 1. Square forms and properties (10) ---
    "Think about someone drawing a perfect square on paper, all four sides equal.",
    "Imagine someone measuring the four right angles of a square, each exactly ninety degrees.",
    "Consider someone laying a square tile on a floor, its edges meeting neighboring tiles in straight lines.",
    "Think about someone looking at a chessboard made up of alternating dark and light squares.",
    "Imagine someone viewing a city block from above, seeing it form a square bounded by streets.",
    "Consider someone identifying the four lines of symmetry that a square possesses.",
    "Think about someone examining a salt crystal forming a nearly perfect tiny cube with square faces.",
    "Imagine someone rotating a square ninety degrees and seeing it look exactly the same.",
    "Consider someone looking through a square window pane set into a wooden frame.",
    "Think about someone tracing the equal diagonals of a square, crossing at the center.",

    # --- 2. Triangular forms and properties (10) ---
    "Think about someone drawing a triangle on paper, three sides meeting at three vertices.",
    "Imagine someone verifying that the three angles of a triangle sum to one hundred eighty degrees.",
    "Consider someone noticing a triangular road sign, its three straight sides converging at three points.",
    "Think about someone gazing at a mountain peak forming a sharp triangular silhouette.",
    "Imagine someone watching a triangular sail on a boat, catching wind in its broad flat surface.",
    "Consider someone counting the three lines of symmetry in an equilateral triangle.",
    "Think about someone examining a crystal face shaped like an equilateral triangle.",
    "Imagine someone rotating an equilateral triangle one hundred twenty degrees and seeing it unchanged.",
    "Consider someone looking up at a triangular window set into the gable end of a roof.",
    "Think about someone measuring the unequal sides of a scalene triangle, no two the same length.",

    # --- 3. Geometric comparisons (10) ---
    "Think about someone comparing a shape with four equal sides to one with three.",
    "Imagine someone dividing a square diagonally and producing two right triangles.",
    "Consider someone noticing how a square has four corners while a triangle has only three.",
    "Think about someone calculating the interior angle sum: three hundred sixty for a square, one hundred eighty for a triangle.",
    "Imagine someone fitting two identical right triangles together to form a square.",
    "Consider someone discovering how both squares and triangles can tile a flat surface without gaps.",
    "Think about someone drawing a square inscribed inside a triangle, its sides touching the triangle's edges.",
    "Imagine someone drawing a triangle inscribed inside a square, its vertices touching the square's sides.",
    "Consider someone computing the ratio of a shape's area to its perimeter for a square versus a triangle.",
    "Think about someone observing how adding one side to a triangle produces a quadrilateral.",

    # --- 4. Patterns and tessellations (10) ---
    "Think about someone looking at a grid of squares covering a surface, each one identical.",
    "Imagine someone studying a mosaic of triangles, alternating upward and downward to fill a plane.",
    "Consider someone examining a quilt pattern made of alternating square and triangular patches.",
    "Think about someone inspecting a geodesic structure built entirely from connected triangles.",
    "Imagine someone walking across a parquet floor with square and triangular tiles arranged in a repeating pattern.",
    "Consider someone exploring the Sierpinski triangle, a fractal made of triangles nested inside triangles.",
    "Think about someone examining a fractal of squares within squares, each one a smaller copy of the whole.",
    "Imagine someone admiring a stained glass window with a pattern mixing square and triangular panes.",
    "Consider someone sketching on graph paper, its surface divided into a uniform grid of small squares.",
    "Think about someone studying a truss bridge, its structure built from repeating triangular frames.",
]

assert len(STANDALONE_PROMPTS_DIM31) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM31)}"

CATEGORY_INFO_STANDALONE_DIM31 = [
    {"name": "square_forms",        "start": 0,  "end": 10},
    {"name": "triangular_forms",    "start": 10, "end": 20},
    {"name": "geometric_comparisons", "start": 20, "end": 30},
    {"name": "patterns_tessellations", "start": 30, "end": 40},
]
