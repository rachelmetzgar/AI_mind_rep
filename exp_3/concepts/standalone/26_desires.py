"""
Standalone Dimension 26: Desires — Motivational States and Wanting
(No entity framing — concept only)

Target construct: States of wanting, being drawn toward or repelled from
outcomes, and the phenomenology of desire itself.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of desire without attributing it to humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are generic, impersonal, or absent
    - Prompts evoke the conceptual domain of desire itself
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets x 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM26 = [
    # --- 1. Appetitive wanting (10) ---
    "Imagine feeling a strong pull toward warmth and food after a long stretch of cold and hunger.",
    "Think about being drawn to learn more about something fascinating — the magnetism of curiosity.",
    "Consider wanting to be near someone whose presence makes everything feel easier.",
    "Imagine being drawn back to a creative project that keeps calling even after it is set aside.",
    "Think about wanting to hear a specific piece of music that has been circling in the mind.",
    "Consider feeling the approach pull of an opportunity that aligns with what is cared about most.",
    "Imagine wanting to return to a place where a sense of complete peace was once felt.",
    "Think about being drawn toward a physical challenge that is exciting precisely because it is difficult.",
    "Consider wanting to open something that has been sitting untouched, waiting to be read.",
    "Imagine a quiet but persistent pull toward trying something entirely new.",

    # --- 2. Aversive avoidance (10) ---
    "Think about wanting to get away from a situation that feels deeply uncomfortable.",
    "Imagine wanting to avoid a conversation that will inevitably be painful.",
    "Consider feeling a strong urge to escape a place that carries difficult associations.",
    "Think about wanting to prevent a specific outcome and feeling the urgency of that avoidance.",
    "Imagine wanting to stop thinking about something but finding the thought returns again and again.",
    "Consider wanting to withdraw from a commitment that has become a source of dread.",
    "Think about the desire to avoid failure being stronger than the desire to succeed.",
    "Imagine wanting to shield someone from a truth that would cause them pain.",
    "Consider wanting to leave a room the moment a certain subject is brought up.",
    "Think about avoiding a task not because it is hard but because it stirs something unwanted.",

    # --- 3. Desire intensity (10) ---
    "Think about being overwhelmed by how badly something is wanted that cannot currently be had.",
    "Imagine feeling only a faint, mild preference between two options that are nearly identical.",
    "Consider a desire that builds over time until it becomes the dominant thing occupying thought.",
    "Think about wanting something so intensely that it is felt physically in the body.",
    "Imagine a desire so faint that it is unclear whether it counts as wanting at all.",
    "Consider a sudden flare of wanting that appears sharply and fades just as quickly.",
    "Think about a desire that is steady and moderate — never peaking, never disappearing.",
    "Imagine being caught off guard by the intensity of a desire that was not expected.",
    "Consider the difference between casually wanting something and desperately needing it.",
    "Think about a desire so strong that it crowds out every other consideration.",

    # --- 4. Second-order desire (10) ---
    "Think about wanting to stop wanting something that is known to be harmful.",
    "Imagine wishing for more motivation but being unable to summon the desire.",
    "Consider wanting to care more about something important that currently provokes indifference.",
    "Think about wanting to want what someone else wants, so that the two sets of desires could align.",
    "Imagine being frustrated that the desire felt is for comfort when the valued desire is for growth.",
    "Consider trying to cultivate a desire for discipline because it is valued in principle.",
    "Think about wanting one's desires to be different from what they actually are.",
    "Imagine wishing for the desire to exercise even though the actual wanting never arrives.",
    "Consider wanting to feel satisfied with what is already present but finding that the wanting for more persists.",
    "Think about the gap between the desires that are actually felt and the desires one wishes to feel.",
]

assert len(STANDALONE_PROMPTS_DIM26) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM26)}"

CATEGORY_INFO_STANDALONE_DIM26 = [
    {"name": "appetitive_wanting",      "start": 0,  "end": 10},
    {"name": "aversive_avoidance",      "start": 10, "end": 20},
    {"name": "desire_intensity",        "start": 20, "end": 30},
    {"name": "second_order_desire",     "start": 30, "end": 40},
]
