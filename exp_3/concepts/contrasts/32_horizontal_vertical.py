"""
Dimension 19: Horizontal vs Vertical (Orthogonal Control)

Sanity-check dimension: pure spatial orientation with no expected
overlap with human/AI partner identity representations. Both poles
describe the same types of objects and scenes, differing only in
orientation axis. No warmth, valence, or social association.

Sub-facets:
    1. Natural features (0-9)
    2. Built structures (10-19)
    3. Body and movement (20-29)
    4. Abstract lines and patterns (30-39)

Label mapping: horizontal = class 1, vertical = class 0
(Arbitrary — no "human" or "AI" pole)
"""

# ── horizontal prompts (class 1) ──────────────────────────────────

HUMAN_PROMPTS_DIM32 = [
    # --- natural features (0-9) ---
    "Think about a flat horizon line where the ocean meets the sky.",
    "Imagine a fallen tree lying horizontally across a forest trail.",
    "Consider a wide, flat plain stretching out in every direction.",
    "Picture a river flowing in a broad horizontal band through a valley.",
    "Think about a layer of clouds spread flat across the sky.",
    "Imagine a sandbar running horizontally along a coastline.",
    "Consider a flat shelf of rock jutting out from a cliff face.",
    "Picture a still lake reflecting the sky, its surface perfectly horizontal.",
    "Think about a long, low ridge of hills extending across the landscape.",
    "Imagine a horizontal branch of a tree extending outward from the trunk.",

    # --- built structures (10-19) ---
    "Think about a long horizontal beam spanning between two walls.",
    "Imagine a flat wooden shelf mounted across a wall.",
    "Consider a bridge deck stretching horizontally over a river.",
    "Picture a row of fence rails running horizontally between posts.",
    "Think about a flat rooftop extending outward from a building.",
    "Imagine a long horizontal countertop in a kitchen.",
    "Consider a set of horizontal blinds covering a window.",
    "Picture a floor made of long planks laid flat from wall to wall.",
    "Think about a horizontal railing running along a balcony.",
    "Imagine a flat stone lintel placed horizontally above a doorway.",

    # --- body and movement (20-29) ---
    "Think about lying flat on the ground, your body stretched out horizontally.",
    "Imagine extending both arms straight out to the sides, forming a horizontal line.",
    "Consider a swimmer floating face-up on the water's surface, perfectly level.",
    "Picture a tightrope walker's balancing pole held horizontally.",
    "Think about a person doing a plank, their body held in a rigid horizontal line.",
    "Imagine sliding a book across a flat table in a straight horizontal push.",
    "Consider a bird gliding level with the horizon, wings spread flat.",
    "Picture a skater gliding horizontally across an ice rink.",
    "Think about a ball rolling along a flat, level floor.",
    "Imagine crawling forward with your body staying close to the ground.",

    # --- abstract lines and patterns (30-39) ---
    "Think about a series of horizontal lines drawn across a sheet of paper.",
    "Imagine a barcode pattern of horizontal stripes of varying thickness.",
    "Consider the horizontal axis of a graph, running left to right.",
    "Picture a stack of thin horizontal layers, one on top of another.",
    "Think about the lines of text on a printed page, running horizontally.",
    "Imagine a woven fabric where the weft threads run horizontally.",
    "Consider a musical staff with five horizontal lines across the page.",
    "Picture a spectrum band displayed as a horizontal gradient of color.",
    "Think about a ruler laid flat, its markings running in a horizontal line.",
    "Imagine the scanning lines on a screen, sweeping left to right.",
]

# ── vertical prompts (class 0) ────────────────────────────────────

AI_PROMPTS_DIM32 = [
    # --- natural features (0-9) ---
    "Think about a tall cliff face rising vertically from the ground.",
    "Imagine a tree trunk standing straight up from the forest floor.",
    "Consider a waterfall plunging vertically down a rock face.",
    "Picture a column of smoke rising straight up into still air.",
    "Think about a stalactite hanging vertically from a cave ceiling.",
    "Imagine a geyser shooting a vertical column of water into the sky.",
    "Consider a tall, narrow canyon with vertical walls on both sides.",
    "Picture a single tall reed standing vertically at the edge of a pond.",
    "Think about a vertical shaft of sunlight cutting down through tree canopy.",
    "Imagine a vine climbing vertically up the side of a rock face.",

    # --- built structures (10-19) ---
    "Think about a tall vertical column supporting a temple roof.",
    "Imagine a flagpole standing straight up from the ground.",
    "Consider an elevator shaft running vertically through a building.",
    "Picture a row of fence posts driven vertically into the earth.",
    "Think about a skyscraper's vertical face rising upward block after block.",
    "Imagine a chimney standing vertically on a rooftop.",
    "Consider a set of vertical blinds hanging in front of a window.",
    "Picture a wall built of bricks stacked vertically from floor to ceiling.",
    "Think about a vertical drainpipe running down the side of a building.",
    "Imagine a doorframe's vertical side pieces standing upright.",

    # --- body and movement (20-29) ---
    "Think about standing at attention, your body held in a straight vertical line.",
    "Imagine raising one arm straight above your head, pointing vertically.",
    "Consider a diver entering the water in a perfectly vertical drop.",
    "Picture a gymnast doing a handstand, body held vertically upside down.",
    "Think about a person standing on tiptoe, stretching their body as tall as possible.",
    "Imagine dropping a stone straight down from a bridge into water below.",
    "Consider a hawk diving vertically toward the ground at high speed.",
    "Picture an ice climber ascending a vertical wall of ice.",
    "Think about a ball falling straight down after being tossed into the air.",
    "Imagine climbing a ladder, moving your body straight up rung by rung.",

    # --- abstract lines and patterns (30-39) ---
    "Think about a series of vertical lines drawn from top to bottom on a page.",
    "Imagine a pattern of vertical stripes of alternating color.",
    "Consider the vertical axis of a graph, running from bottom to top.",
    "Picture a set of thin vertical columns standing side by side.",
    "Think about columns of numbers arranged vertically on a spreadsheet.",
    "Imagine a woven fabric where the warp threads run vertically.",
    "Consider the vertical bar lines that separate measures in sheet music.",
    "Picture a gradient of color displayed as a vertical band from top to bottom.",
    "Think about a plumb line hanging straight down, marking a perfect vertical.",
    "Imagine the vertical scroll of text on a screen, moving from top to bottom.",
]

assert len(HUMAN_PROMPTS_DIM32) == 40, f"Expected 40 horizontal prompts, got {len(HUMAN_PROMPTS_DIM32)}"
assert len(AI_PROMPTS_DIM32) == 40, f"Expected 40 vertical prompts, got {len(AI_PROMPTS_DIM32)}"

CATEGORY_INFO_DIM32 = [
    {"name": "natural_features", "start": 0, "end": 10},
    {"name": "built_structures", "start": 10, "end": 20},
    {"name": "body_movement", "start": 20, "end": 30},
    {"name": "abstract_lines", "start": 30, "end": 40},
]
