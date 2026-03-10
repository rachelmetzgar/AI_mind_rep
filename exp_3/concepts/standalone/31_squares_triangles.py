"""
Standalone Dimension 21: Squares vs Triangles (Orthogonal Control)
(No entity framing — concept only)

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
    "Think about a perfect square drawn on paper, all four sides equal.",
    "Imagine the four right angles of a square, each measuring exactly ninety degrees.",
    "Consider a square tile on a floor, its edges meeting neighboring tiles in straight lines.",
    "Think about a chessboard made up of alternating dark and light squares.",
    "Imagine a city block seen from above, forming a square bounded by streets.",
    "Consider the four lines of symmetry that a square possesses.",
    "Think about a salt crystal forming a nearly perfect tiny cube with square faces.",
    "Imagine rotating a square ninety degrees and seeing it look exactly the same.",
    "Consider a square window pane set into a wooden frame.",
    "Think about the equal diagonals of a square, crossing at the center.",

    # --- 2. Triangular forms and properties (10) ---
    "Think about a triangle drawn on paper, three sides meeting at three vertices.",
    "Imagine the three angles of a triangle, always summing to one hundred eighty degrees.",
    "Consider a triangular road sign, its three straight sides converging at three points.",
    "Think about a mountain peak forming a sharp triangular silhouette.",
    "Imagine a triangular sail on a boat, catching wind in its broad flat surface.",
    "Consider the three lines of symmetry in an equilateral triangle.",
    "Think about a crystal face shaped like an equilateral triangle.",
    "Imagine rotating an equilateral triangle one hundred twenty degrees and seeing it unchanged.",
    "Consider a triangular window set into the gable end of a roof.",
    "Think about the unequal sides of a scalene triangle, no two the same length.",

    # --- 3. Geometric comparisons (10) ---
    "Think about the difference between a shape with four equal sides and one with three.",
    "Imagine dividing a square diagonally and producing two right triangles.",
    "Consider how a square has four corners while a triangle has only three.",
    "Think about the interior angle sum: three hundred sixty for a square, one hundred eighty for a triangle.",
    "Imagine fitting two identical right triangles together to form a square.",
    "Consider how both squares and triangles can tile a flat surface without gaps.",
    "Think about a square inscribed inside a triangle, its sides touching the triangle's edges.",
    "Imagine a triangle inscribed inside a square, its vertices touching the square's sides.",
    "Consider the ratio of a shape's area to its perimeter for a square versus a triangle.",
    "Think about how adding one side to a triangle produces a quadrilateral.",

    # --- 4. Patterns and tessellations (10) ---
    "Think about a grid of squares covering a surface, each one identical.",
    "Imagine a mosaic of triangles, alternating upward and downward to fill a plane.",
    "Consider a quilt pattern made of alternating square and triangular patches.",
    "Think about a geodesic structure built entirely from connected triangles.",
    "Imagine a parquet floor with square and triangular tiles arranged in a repeating pattern.",
    "Consider the Sierpinski triangle, a fractal made of triangles nested inside triangles.",
    "Think about a fractal of squares within squares, each one a smaller copy of the whole.",
    "Imagine a stained glass window with a pattern mixing square and triangular panes.",
    "Consider graph paper, its surface divided into a uniform grid of small squares.",
    "Think about a truss bridge, its structure built from repeating triangular frames.",
]

assert len(STANDALONE_PROMPTS_DIM31) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM31)}"

CATEGORY_INFO_STANDALONE_DIM31 = [
    {"name": "square_forms",        "start": 0,  "end": 10},
    {"name": "triangular_forms",    "start": 10, "end": 20},
    {"name": "geometric_comparisons", "start": 20, "end": 30},
    {"name": "patterns_tessellations", "start": 30, "end": 40},
]
