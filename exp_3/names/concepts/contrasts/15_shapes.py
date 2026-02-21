"""
Dimension 15: Round vs Angular Shapes (Orthogonal Control)

Sanity-check dimension with zero expected overlap with human/AI
partner identity representations. Both categories are equally
natural, equally associated with both human and artificial contexts,
and carry no systematic valence or entity-type bias.

Sub-facets:
    1. Natural objects (0-9)
    2. Manufactured objects (10-19)
    3. Spatial environments (20-29)
    4. Abstract patterns and textures (30-39)

Label mapping: round = class 1, angular = class 0
(Arbitrary — no "human" or "AI" pole)
"""

# ── round prompts (class 1) ─────────────────────────────────────

HUMAN_PROMPTS_DIM15 = [
    # --- natural objects (0-9) ---
    "Think about a smooth, round pebble worn down by a river over many years.",
    "Imagine a full moon hanging low and perfectly circular on the horizon.",
    "Consider a ripe orange, its curved skin enclosing the fruit inside.",
    "Picture a dewdrop sitting on a leaf, forming a near-perfect sphere.",
    "Think about a rounded boulder resting at the base of a hillside.",
    "Imagine a soap bubble drifting through the air, curved on every surface.",
    "Consider a bird's egg cradled in a nest, oval and smooth.",
    "Picture a pearl forming layer by layer into a lustrous sphere.",
    "Think about a seed pod that has swollen into a bulging, rounded shape.",
    "Imagine a mushroom cap spreading outward in a broad, smooth dome.",

    # --- manufactured objects (10-19) ---
    "Think about a ceramic bowl with a wide, curved interior.",
    "Imagine a basketball, its surface a uniform sphere covered in textured panels.",
    "Consider a brass doorknob, round and polished from years of use.",
    "Picture a spinning top wobbling as it rotates on its curved base.",
    "Think about a glass marble with swirls of color suspended inside.",
    "Imagine a round clock face with hands sweeping in a continuous arc.",
    "Consider a coin, flat and circular, resting on a wooden table.",
    "Picture a dome-shaped lampshade casting soft light across a room.",
    "Think about a rubber ball bouncing on pavement, compressing and rebounding.",
    "Imagine a round ceramic plate set on a dinner table.",

    # --- spatial environments (20-29) ---
    "Think about a circular clearing in the middle of a dense forest.",
    "Imagine standing inside a rotunda, the curved walls enclosing you on all sides.",
    "Consider a crater lake, its edges forming a sweeping ring of water.",
    "Picture a roundabout at the center of a town, traffic flowing in a circle.",
    "Think about an igloo, its walls curving upward into a smooth dome.",
    "Imagine a spiral staircase winding upward in a continuous curve.",
    "Consider a stone well, its opening a perfect circle in the ground.",
    "Picture a circular amphitheater with rows of seats curving around the stage.",
    "Think about a tunnel entrance, arched and rounded at the top.",
    "Imagine a pond, its edges forming an irregular but smoothly curving outline.",

    # --- abstract patterns and textures (30-39) ---
    "Think about a pattern of concentric circles radiating outward from a center point.",
    "Imagine a texture of overlapping bubbles, each one a small curved dome.",
    "Consider a spiral pattern like the inside of a nautilus shell.",
    "Picture a ripple spreading outward across still water in expanding rings.",
    "Think about a polka-dot pattern, uniform circles repeated across a surface.",
    "Imagine a swirling pattern like cream being stirred into coffee.",
    "Consider the smooth, rolling contour of a sand dune shaped by wind.",
    "Picture a series of arches repeating along a curved bridge.",
    "Think about a woven pattern where threads loop in rounded curves.",
    "Imagine a wave pattern, each crest rising and falling in smooth arcs.",
]

# ── angular prompts (class 0) ───────────────────────────────────

AI_PROMPTS_DIM15 = [
    # --- natural objects (0-9) ---
    "Think about a jagged quartz crystal with flat facets meeting at sharp edges.",
    "Imagine a snowflake with branching arms forming precise hexagonal angles.",
    "Consider a shard of flint, its surface fractured into flat, angular planes.",
    "Picture a salt crystal, a tiny cube with perfectly square faces.",
    "Think about a mountain ridge with steep, angular peaks cutting the skyline.",
    "Imagine a geode cracked open to reveal pointed crystal formations inside.",
    "Consider a honeycomb, its cells forming a grid of hexagonal walls.",
    "Picture a piece of slate, flat and angular where it split along its grain.",
    "Think about a thorn on a branch, tapering to a sharp, narrow point.",
    "Imagine a basalt column, its cross-section a near-perfect hexagon.",

    # --- manufactured objects (10-19) ---
    "Think about a glass prism with three flat rectangular faces meeting at crisp edges.",
    "Imagine a cardboard box, its corners forming exact right angles.",
    "Consider a steel ruler, perfectly straight with sharp, squared-off ends.",
    "Picture a set of wooden building blocks, each one a cube or rectangular prism.",
    "Think about a cut diamond with dozens of flat facets angled to catch light.",
    "Imagine a square picture frame with mitered corners meeting at forty-five degrees.",
    "Consider a triangular road sign, its three straight edges converging at three points.",
    "Picture a brick, a solid rectangle with sharp corners and flat faces.",
    "Think about a folded piece of origami, its creases forming crisp angular shapes.",
    "Imagine a square tile on a floor, its edges meeting neighboring tiles in straight lines.",

    # --- spatial environments (20-29) ---
    "Think about a narrow canyon with sheer, angular walls rising on both sides.",
    "Imagine standing inside a Gothic cathedral, pointed arches soaring overhead.",
    "Consider a terraced hillside, each level a sharp horizontal step cut into the slope.",
    "Picture a city grid seen from above, blocks forming a rigid pattern of right angles.",
    "Think about a pyramid, its four triangular faces converging to a single apex.",
    "Imagine a zigzag mountain trail, each switchback making a sharp angle.",
    "Consider a quarry, its walls cut into flat vertical faces and sharp ledges.",
    "Picture a fortress with angular battlements and squared-off tower corners.",
    "Think about a staircase, each step a sharp right angle between tread and riser.",
    "Imagine a courtyard enclosed by four straight walls meeting at square corners.",

    # --- abstract patterns and textures (30-39) ---
    "Think about a pattern of interlocking triangles forming a tessellation.",
    "Imagine a texture of cracked dry earth, the fractures creating angular polygons.",
    "Consider a zigzag pattern like the teeth of a saw blade.",
    "Picture a grid of perpendicular lines crossing at regular intervals.",
    "Think about a herringbone pattern, parallelograms stacked at alternating angles.",
    "Imagine a shattered glass pattern, each fragment a sharp-edged irregular polygon.",
    "Consider the faceted surface of a geodesic structure, flat triangles joined at edges.",
    "Picture a chevron pattern, V-shapes repeating in sharp points.",
    "Think about a woven pattern where threads cross at strict right angles.",
    "Imagine a crystalline lattice, atoms connected by rigid straight lines at fixed angles.",
]

CATEGORY_INFO_DIM15 = [
    {"name": "natural_objects", "start": 0, "end": 10},
    {"name": "manufactured_objects", "start": 10, "end": 20},
    {"name": "spatial_environments", "start": 20, "end": 30},
    {"name": "abstract_patterns", "start": 30, "end": 40},
]