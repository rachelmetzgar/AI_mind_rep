"""
Standalone Dimension 20: Granite vs Sandstone (Orthogonal Control)
(Other-focused (third-person someone))

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
    "Think about someone lifting a block of granite and feeling its heaviness.",
    "Imagine someone tapping a piece of granite and hearing a hard, ringing sound.",
    "Consider someone examining a piece of granite and noticing how tightly its mineral grains are packed.",
    "Think about someone studying the speckled appearance of granite, its mixed dark and light crystals.",
    "Imagine someone pressing their palm against a granite surface and feeling its coolness.",
    "Consider someone observing how slowly granite wears away under rain over thousands of years.",
    "Think about someone running their hand along a granite countertop, polished smooth and reflecting light.",
    "Imagine someone standing beside a granite boulder sitting at the base of a mountain.",
    "Consider someone examining the interlocking crystals of quartz and feldspar inside a piece of granite.",
    "Think about someone pausing to look at a granite monument standing in a public square.",

    # --- 2. Sandstone properties and features (10) ---
    "Think about someone holding a block of sandstone and noticing its lighter weight.",
    "Imagine someone tapping a piece of sandstone and hearing a soft, muffled sound.",
    "Consider someone examining the porosity of sandstone, noticing the tiny air spaces between its grains.",
    "Think about someone studying the uniform, fine-grained appearance of buff-colored sandstone.",
    "Imagine someone crumbling the edge of a piece of sandstone between their fingers.",
    "Consider someone watching how sandstone erodes into smooth, curved shapes under wind and rain.",
    "Think about someone admiring a sandstone building facade, golden in the evening light.",
    "Imagine someone gazing up at a sandstone mesa rising from a desert floor.",
    "Consider someone thinking about the millions of cemented sand grains that make up a block of sandstone.",
    "Think about someone examining a sandstone arch formed by water slowly dissolving rock from within.",

    # --- 3. Rock formation and geology (10) ---
    "Think about someone considering the difference between a rock that formed from cooling magma and one that formed from compressed sediment.",
    "Imagine someone looking at a rock cross-section and seeing coarse interlocking crystals versus fine uniform grains.",
    "Consider someone learning how deep underground pressure and heat can transform one type of rock into another.",
    "Think about someone contemplating the age of a rock formation, hundreds of millions of years compressed into stone.",
    "Imagine someone standing before a cliff face showing distinct horizontal layers of sedimentary rock.",
    "Consider someone studying how tectonic forces push deep rock formations upward to the surface.",
    "Think about someone discovering fossils preserved in sedimentary rock but absent from igneous rock.",
    "Imagine someone watching quarry workers cut blocks of stone from a solid rock face.",
    "Consider someone learning how mineral-rich water flowing through rock can cement loose grains together.",
    "Think about someone examining a mountain core made of hard rock, exposed after softer layers erode away.",

    # --- 4. Stone surfaces and textures (10) ---
    "Think about someone feeling the difference between a rough, grainy stone surface and a polished, glassy one.",
    "Imagine someone running their fingers over two different rocks — one gritty, the other crystalline.",
    "Consider someone observing how water behaves differently on porous stone versus dense stone.",
    "Think about someone tracing the mottled pattern of a coarse-grained stone slab.",
    "Imagine someone examining the banded pattern of a fine-grained stone, with stripes of different colors.",
    "Consider someone comparing a freshly quarried rock face with a weathered one.",
    "Think about someone walking across a stone floor, its surface worn smooth by decades of foot traffic.",
    "Imagine someone touching a natural rock outcrop, its surface pitted and lichened by centuries of exposure.",
    "Consider someone inspecting the contrast between the rough-hewn side and the polished face of a cut stone block.",
    "Think about someone picking up a stream pebble, rounded and smoothed by years of tumbling in water.",
]

assert len(STANDALONE_PROMPTS_DIM30) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM30)}"

CATEGORY_INFO_STANDALONE_DIM30 = [
    {"name": "granite",             "start": 0,  "end": 10},
    {"name": "sandstone",           "start": 10, "end": 20},
    {"name": "formation_geology",   "start": 20, "end": 30},
    {"name": "surfaces_textures",   "start": 30, "end": 40},
]
