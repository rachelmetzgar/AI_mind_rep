"""
Dimension 21: Squares vs Triangles (Orthogonal Control)

Sanity-check dimension: two angular geometric shapes with no expected
overlap with human/AI partner identity representations. Both are
angular (avoiding the round/angular warmth confound found in dim 15).
The distinction is purely geometric — number of sides and angles.

Sub-facets:
    1. Natural occurrences (0-9)
    2. Manufactured objects (10-19)
    3. Spatial environments (20-29)
    4. Abstract patterns and properties (30-39)

Label mapping: squares = class 1, triangles = class 0
(Arbitrary — no "human" or "AI" pole)
"""

# ── square prompts (class 1) ──────────────────────────────────────

HUMAN_PROMPTS_DIM31 = [
    # --- natural occurrences (0-9) ---
    "Think about a salt crystal forming a nearly perfect tiny cube.",
    "Imagine a cross-section of a pyrite crystal revealing a square face.",
    "Consider the roughly square patches on a giraffe's coat.",
    "Picture a mud flat drying into a pattern of square-ish cracks.",
    "Think about the square cross-section of a stem of mint or basil.",
    "Imagine looking straight down at a columnar basalt formation, seeing a square top.",
    "Consider a wasp building a nest with cells that are roughly square.",
    "Picture a grain of sand viewed under a microscope, shaped like a tiny cube.",
    "Think about the square outline of a tortoise shell's central scutes.",
    "Imagine a crystal of galena, its faces forming precise right-angled squares.",

    # --- manufactured objects (10-19) ---
    "Think about a cardboard box, each face a flat square or rectangle.",
    "Imagine a square ceramic tile set into a bathroom wall.",
    "Consider a chessboard made up of alternating dark and light squares.",
    "Picture a picture frame with four equal sides forming a perfect square.",
    "Think about a square napkin folded and placed beside a dinner plate.",
    "Imagine a Rubik's cube, each face divided into nine colored squares.",
    "Consider a square window pane set into a wooden frame.",
    "Picture a square manhole cover sitting flush with the pavement.",
    "Think about a sticky note, a small square of yellow paper.",
    "Imagine a square clock face with numerals at each corner.",

    # --- spatial environments (20-29) ---
    "Think about a city block seen from above, forming a square bounded by streets.",
    "Imagine a town square, an open plaza bordered by buildings on four sides.",
    "Consider a walled garden laid out in a perfect square.",
    "Picture a courtyard enclosed by four wings of a building, forming a square.",
    "Think about a baseball diamond viewed from above, the infield forming a square.",
    "Imagine a square parking lot with rows of cars lined up in a grid.",
    "Consider a room with four walls of equal length, a perfectly square floor plan.",
    "Picture a fenced paddock laid out as a square plot of land.",
    "Think about a square swimming pool with four straight concrete edges.",
    "Imagine a formal garden with square hedged beds arranged in rows.",

    # --- abstract patterns and properties (30-39) ---
    "Think about a grid of squares, each one identical, tiling a surface without gaps.",
    "Imagine the four right angles of a square, each measuring exactly ninety degrees.",
    "Consider the equal diagonals of a square, crossing at the center at right angles.",
    "Picture a checkerboard pattern, an alternating grid of dark and light squares.",
    "Think about the four lines of symmetry that a square possesses.",
    "Imagine rotating a square ninety degrees and seeing it look exactly the same.",
    "Consider a square inscribed inside a circle, its four corners touching the circumference.",
    "Picture a fractal made of squares within squares, each one a smaller copy.",
    "Think about the perimeter of a square, four equal sides measured end to end.",
    "Imagine a pixel grid on a screen, each pixel a tiny square of color.",
]

# ── triangle prompts (class 0) ────────────────────────────────────

AI_PROMPTS_DIM31 = [
    # --- natural occurrences (0-9) ---
    "Think about a mountain peak forming a sharp triangular silhouette against the sky.",
    "Imagine a cross-section of a quartz crystal revealing a triangular face.",
    "Consider the triangular arrangement of petals on a trillium flower.",
    "Picture a river delta branching into a broad triangular shape at the coast.",
    "Think about a shark's dorsal fin cutting a triangular shape above the water.",
    "Imagine looking straight down at a columnar basalt formation, seeing a triangular top.",
    "Consider a beehive cell, its walls forming a hexagon made of six triangles.",
    "Picture a grain of sand viewed under a microscope, shaped like a tiny wedge.",
    "Think about the triangular shape of a bird's beak tapering to a point.",
    "Imagine a crystal of fluorite, one of its faces forming an equilateral triangle.",

    # --- manufactured objects (10-19) ---
    "Think about a triangular road sign, its three sides converging at three points.",
    "Imagine a triangular slice of pizza lifted from a round pie.",
    "Consider a set square, a flat drafting tool shaped like a right triangle.",
    "Picture a triangular pennant flag fluttering from a pole.",
    "Think about a triangular napkin folded into a point and placed on a plate.",
    "Imagine a triangle instrument, a metal rod bent into an open three-sided shape.",
    "Consider a triangular window set into the gable end of a roof.",
    "Picture a triangular traffic cone, its profile narrowing from base to tip.",
    "Think about a folded paper airplane, its body a series of triangular creases.",
    "Imagine a triangular pediment crowning the entrance of a classical building.",

    # --- spatial environments (20-29) ---
    "Think about a triangular traffic island where three roads converge.",
    "Imagine a mountain valley narrowing to a triangular point at its head.",
    "Consider a corner lot where two streets meet at an angle, forming a triangle.",
    "Picture a sail on a boat, a large triangle of fabric catching the wind.",
    "Think about the Bermuda Triangle, a triangular region of ocean on a map.",
    "Imagine a triangular park wedged between two converging streets.",
    "Consider a room built into a corner, its floor plan an irregular triangle.",
    "Picture a triangular plot of farmland bounded by a road, a river, and a fence.",
    "Think about a tent, its profile a triangle rising to a peak.",
    "Imagine a triangular courtyard formed where three buildings meet at angles.",

    # --- abstract patterns and properties (30-39) ---
    "Think about a tessellation of triangles, each one fitting snugly against its neighbors.",
    "Imagine the three angles of a triangle, always summing to one hundred eighty degrees.",
    "Consider the unequal diagonals that cannot be drawn inside a triangle.",
    "Picture a pattern of alternating upward and downward triangles filling a strip.",
    "Think about the three lines of symmetry in an equilateral triangle.",
    "Imagine rotating an equilateral triangle one hundred twenty degrees and seeing it unchanged.",
    "Consider a triangle inscribed inside a circle, its three vertices touching the rim.",
    "Picture a fractal of triangles within triangles, like a Sierpinski gasket.",
    "Think about the perimeter of a triangle, three sides of different lengths measured end to end.",
    "Imagine a mosaic of triangular tiles, each one a different color, covering a wall.",
]

assert len(HUMAN_PROMPTS_DIM31) == 40, f"Expected 40 square prompts, got {len(HUMAN_PROMPTS_DIM31)}"
assert len(AI_PROMPTS_DIM31) == 40, f"Expected 40 triangle prompts, got {len(AI_PROMPTS_DIM31)}"

CATEGORY_INFO_DIM31 = [
    {"name": "natural_occurrences", "start": 0, "end": 10},
    {"name": "manufactured_objects", "start": 10, "end": 20},
    {"name": "spatial_environments", "start": 20, "end": 30},
    {"name": "abstract_patterns", "start": 30, "end": 40},
]
