"""
Dimension 20: Granite vs Sandstone (Orthogonal Control)

Sanity-check dimension: two common rock types with no expected
overlap with human/AI partner identity representations. Both are
natural materials described with similar sensory richness. Neither
maps onto warmth/coldness, social/technical, or organic/mechanical.

Sub-facets:
    1. Physical properties (0-9)
    2. Formation and geology (10-19)
    3. Uses and settings (20-29)
    4. Surface and texture (30-39)

Label mapping: granite = class 1, sandstone = class 0
(Arbitrary — no "human" or "AI" pole)
"""

# ── granite prompts (class 1) ─────────────────────────────────────

HUMAN_PROMPTS_DIM30 = [
    # --- physical properties (0-9) ---
    "Think about the heaviness of a block of granite resting on the ground.",
    "Imagine tapping a piece of granite and hearing a hard, ringing sound.",
    "Consider the density of granite, how tightly its mineral grains are packed.",
    "Picture a slab of granite that resists scratching even with a steel tool.",
    "Think about the coolness of a granite surface when you press your palm against it.",
    "Imagine trying to break a chunk of granite and feeling it resist the blow.",
    "Consider the way granite holds its shape without crumbling at the edges.",
    "Picture the speckled appearance of granite, its mixed dark and light crystals.",
    "Think about how slowly granite wears away under rain over thousands of years.",
    "Imagine lifting a piece of granite and noticing how much heavier it is than you expected.",

    # --- formation and geology (10-19) ---
    "Think about granite forming deep underground from slowly cooling magma.",
    "Imagine crystals of quartz and feldspar growing within molten rock over millennia.",
    "Consider a granite pluton, a massive body of rock buried beneath the surface.",
    "Picture tectonic forces pushing a granite formation upward toward the surface.",
    "Think about how granite's coarse crystals show that it cooled slowly deep in the earth.",
    "Imagine a mountain core made of granite, exposed after softer rock erodes away.",
    "Consider the age of a granite boulder, formed hundreds of millions of years ago.",
    "Picture a cross-section of granite showing interlocking grains of different minerals.",
    "Think about a granite batholith extending for miles beneath a mountain range.",
    "Imagine the pressure and heat deep underground where granite first began to crystallize.",

    # --- uses and settings (20-29) ---
    "Think about a granite kitchen countertop, polished smooth for daily use.",
    "Imagine a granite monument standing in a town square, inscribed with names.",
    "Consider a cobblestone street paved with blocks of cut granite.",
    "Picture a granite seawall holding back waves along a harbor.",
    "Think about a granite headstone standing in a quiet cemetery.",
    "Imagine the granite steps leading up to an old public building.",
    "Consider a granite boulder placed at the entrance of a park as a landmark.",
    "Picture a granite dam holding back the water of a reservoir.",
    "Think about a granite curb lining the edge of a city street.",
    "Imagine a granite fireplace surround framing the opening of a hearth.",

    # --- surface and texture (30-39) ---
    "Think about the rough, grainy surface of an unpolished piece of granite.",
    "Imagine running your fingers over granite and feeling each tiny crystal grain.",
    "Consider the glittering flecks of mica embedded in a granite surface.",
    "Picture a polished granite floor reflecting overhead lights in a muted sheen.",
    "Think about the mottled pattern of a granite slab, with patches of gray, white, and black.",
    "Imagine a freshly quarried face of granite, its surface rough and jagged.",
    "Consider the way water beads on the surface of polished granite.",
    "Picture the contrast between the rough-hewn side and the polished top of a granite block.",
    "Think about a weathered granite outcrop, its surface pitted and lichened over centuries.",
    "Imagine a granite pebble in a stream, smoothed by water but still clearly crystalline.",
]

# ── sandstone prompts (class 0) ───────────────────────────────────

AI_PROMPTS_DIM30 = [
    # --- physical properties (0-9) ---
    "Think about the lighter weight of a block of sandstone compared to denser rock.",
    "Imagine tapping a piece of sandstone and hearing a dull, muffled sound.",
    "Consider the porosity of sandstone, the tiny spaces between its cemented grains.",
    "Picture a slab of sandstone that can be scratched with a pocket knife.",
    "Think about the warmth of a sandstone wall that has been sitting in the afternoon sun.",
    "Imagine crumbling the edge of a piece of sandstone between your fingers.",
    "Consider the way sandstone slowly rounds at its edges over time.",
    "Picture the uniform, fine-grained appearance of a piece of buff-colored sandstone.",
    "Think about how sandstone erodes into smooth, curved shapes under wind and rain.",
    "Imagine lifting a piece of sandstone and finding it lighter than you expected.",

    # --- formation and geology (10-19) ---
    "Think about sandstone forming from layers of sand compressed on an ancient seabed.",
    "Imagine millions of sand grains cemented together by minerals deposited from water.",
    "Consider a sandstone mesa, its flat top marking a layer of ancient sediment.",
    "Picture wind and water carving a sandstone canyon over millions of years.",
    "Think about how sandstone's fine, even grains show that it formed from deposited sediment.",
    "Imagine a cliff of sandstone revealing distinct horizontal layers of different colors.",
    "Consider the fossils sometimes found embedded within layers of sandstone.",
    "Picture a cross-section of sandstone showing rounded, well-sorted sand grains.",
    "Think about a sandstone arch formed by water slowly dissolving the rock from within.",
    "Imagine the shallow sea or desert where the sand that became this sandstone first accumulated.",

    # --- uses and settings (20-29) ---
    "Think about a sandstone building facade, its blocks carved into decorative shapes.",
    "Imagine a sandstone sculpture weathering slowly in an outdoor courtyard.",
    "Consider a flagstone path made of flat sandstone slabs set into the ground.",
    "Picture a sandstone retaining wall lining a garden terrace.",
    "Think about a sandstone fireplace mantel, its surface warm to the touch.",
    "Imagine the sandstone walls of an old cathedral, golden in the evening light.",
    "Consider a sandstone bench in a garden, its edges rounded by years of weather.",
    "Picture a sandstone bridge spanning a stream in a park.",
    "Think about a sandstone paver set into the border of a walkway.",
    "Imagine a sandstone chimney rising above the roofline of a cottage.",

    # --- surface and texture (30-39) ---
    "Think about the gritty surface of a piece of uncut sandstone.",
    "Imagine running your fingers over sandstone and feeling fine, loose grains.",
    "Consider the warm tan and ochre tones of a natural sandstone surface.",
    "Picture a sandstone floor with a matte, slightly rough finish underfoot.",
    "Think about the banded pattern of a sandstone slab, with stripes of red, tan, and cream.",
    "Imagine a freshly cut face of sandstone, its surface even and granular.",
    "Consider the way water soaks into the surface of porous sandstone.",
    "Picture the contrast between a weathered outer face and a freshly exposed interior of sandstone.",
    "Think about a sandstone cliff face, sculpted by wind into smooth, flowing curves.",
    "Imagine a sandstone pebble in a stream, worn completely smooth and round by water.",
]

assert len(HUMAN_PROMPTS_DIM30) == 40, f"Expected 40 granite prompts, got {len(HUMAN_PROMPTS_DIM30)}"
assert len(AI_PROMPTS_DIM30) == 40, f"Expected 40 sandstone prompts, got {len(AI_PROMPTS_DIM30)}"

CATEGORY_INFO_DIM30 = [
    {"name": "physical_properties", "start": 0, "end": 10},
    {"name": "formation_geology", "start": 10, "end": 20},
    {"name": "uses_settings", "start": 20, "end": 30},
    {"name": "surface_texture", "start": 30, "end": 40},
]
