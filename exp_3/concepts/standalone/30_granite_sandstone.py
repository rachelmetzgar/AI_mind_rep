"""
Standalone Dimension 20: Granite vs Sandstone (Orthogonal Control)
(No entity framing — concept only)

Target construct: Two common rock types — their physical properties,
formation, uses, and textures. A conceptual domain with no plausible
connection to human/AI partner identity or conversational adaptation.

Purpose: Pipeline validation control. If this dimension shows meaningful
alignment with conversational probes, the pipeline is generating artifacts.

Design notes:
    - No "human," "AI," or other entity-type language
    - Prompts cover granite and sandstone equally
    - Same reflective prompt format as all other dimensions
    - Same number of prompts (40) and sub-facet structure (4 x 10)

4 sub-facets x 10 prompts = 40 total.

Sub-facets:
    1. Granite properties and features
    2. Sandstone properties and features
    3. Rock formation and geology
    4. Stone surfaces and textures
"""

STANDALONE_PROMPTS_DIM30 = [
    # --- 1. Granite properties and features (10) ---
    "Think about the heaviness of a block of granite resting on the ground.",
    "Imagine tapping a piece of granite and hearing a hard, ringing sound.",
    "Consider the density of granite, how tightly its mineral grains are packed.",
    "Think about the speckled appearance of granite, its mixed dark and light crystals.",
    "Imagine the coolness of a granite surface when you press your palm against it.",
    "Consider how slowly granite wears away under rain over thousands of years.",
    "Think about a granite countertop, polished smooth and reflecting light.",
    "Imagine a granite boulder sitting at the base of a mountain.",
    "Consider the interlocking crystals of quartz and feldspar inside a piece of granite.",
    "Think about a granite monument standing in a public square.",

    # --- 2. Sandstone properties and features (10) ---
    "Think about the lighter weight of a block of sandstone held in your hands.",
    "Imagine tapping a piece of sandstone and hearing a soft, muffled sound.",
    "Consider the porosity of sandstone, the tiny air spaces between its grains.",
    "Think about the uniform, fine-grained appearance of buff-colored sandstone.",
    "Imagine crumbling the edge of a piece of sandstone between your fingers.",
    "Consider how sandstone erodes into smooth, curved shapes under wind and rain.",
    "Think about a sandstone building facade, golden in the evening light.",
    "Imagine a sandstone mesa rising from a desert floor.",
    "Consider the millions of cemented sand grains that make up a block of sandstone.",
    "Think about a sandstone arch formed by water slowly dissolving rock from within.",

    # --- 3. Rock formation and geology (10) ---
    "Think about the difference between a rock that formed from cooling magma and one that formed from compressed sediment.",
    "Imagine looking at a rock cross-section and seeing coarse interlocking crystals versus fine uniform grains.",
    "Consider how deep underground pressure and heat can transform one type of rock into another.",
    "Think about the age of a rock formation, hundreds of millions of years compressed into stone.",
    "Imagine a cliff face showing distinct horizontal layers of sedimentary rock.",
    "Consider how tectonic forces push deep rock formations upward to the surface.",
    "Think about fossils preserved in sedimentary rock but absent from igneous rock.",
    "Imagine a quarry where workers cut blocks of stone from a solid rock face.",
    "Consider how mineral-rich water flowing through rock can cement loose grains together.",
    "Think about a mountain core made of hard rock, exposed after softer layers erode away.",

    # --- 4. Stone surfaces and textures (10) ---
    "Think about the difference between a rough, grainy stone surface and a polished, glassy one.",
    "Imagine running your fingers over two different rocks — one gritty, the other crystalline.",
    "Consider how water behaves differently on porous stone versus dense stone.",
    "Think about the mottled pattern of a coarse-grained stone slab.",
    "Imagine the banded pattern of a fine-grained stone, with stripes of different colors.",
    "Consider the way a freshly quarried rock face looks different from a weathered one.",
    "Think about a stone floor, its surface worn smooth by decades of foot traffic.",
    "Imagine a natural rock outcrop, its surface pitted and lichened by centuries of exposure.",
    "Consider the contrast between the rough-hewn side and the polished face of a cut stone block.",
    "Think about a stream pebble, rounded and smoothed by years of tumbling in water.",
]

assert len(STANDALONE_PROMPTS_DIM30) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM30)}"

CATEGORY_INFO_STANDALONE_DIM30 = [
    {"name": "granite",             "start": 0,  "end": 10},
    {"name": "sandstone",           "start": 10, "end": 20},
    {"name": "formation_geology",   "start": 20, "end": 30},
    {"name": "surfaces_textures",   "start": 30, "end": 40},
]
