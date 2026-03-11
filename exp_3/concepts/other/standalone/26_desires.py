"""
Standalone Dimension 26: Desires — Motivational States and Wanting
(Other-focused (third-person someone))

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
    "Imagine someone feeling a strong pull toward warmth and food after a long stretch of cold and hunger.",
    "Think about someone being drawn to learn more about something fascinating — the magnetism of curiosity.",
    "Consider someone wanting to be near another person whose presence makes everything feel easier.",
    "Imagine someone being drawn back to a creative project that keeps calling even after it is set aside.",
    "Think about someone wanting to hear a specific piece of music that has been circling in their mind.",
    "Consider someone feeling the approach pull of an opportunity that aligns with what they care about most.",
    "Imagine someone wanting to return to a place where a sense of complete peace was once felt.",
    "Think about someone being drawn toward a physical challenge that is exciting precisely because it is difficult.",
    "Consider someone wanting to open something that has been sitting untouched, waiting to be read.",
    "Imagine someone feeling a quiet but persistent pull toward trying something entirely new.",

    # --- 2. Aversive avoidance (10) ---
    "Think about someone wanting to get away from a situation that feels deeply uncomfortable.",
    "Imagine someone wanting to avoid a conversation that will inevitably be painful.",
    "Consider someone feeling a strong urge to escape a place that carries difficult associations.",
    "Think about someone wanting to prevent a specific outcome and feeling the urgency of that avoidance.",
    "Imagine someone wanting to stop thinking about something but finding the thought returns again and again.",
    "Consider someone wanting to withdraw from a commitment that has become a source of dread.",
    "Think about someone whose desire to avoid failure is stronger than their desire to succeed.",
    "Imagine someone wanting to shield another person from a truth that would cause them pain.",
    "Consider someone wanting to leave a room the moment a certain subject is brought up.",
    "Think about someone avoiding a task not because it is hard but because it stirs something unwanted.",

    # --- 3. Desire intensity (10) ---
    "Think about someone being overwhelmed by how badly they want something that cannot currently be had.",
    "Imagine someone feeling only a faint, mild preference between two options that are nearly identical.",
    "Consider someone with a desire that builds over time until it becomes the dominant thing occupying their thought.",
    "Think about someone wanting something so intensely that it is felt physically in their body.",
    "Imagine someone with a desire so faint that it is unclear whether it counts as wanting at all.",
    "Consider someone experiencing a sudden flare of wanting that appears sharply and fades just as quickly.",
    "Think about someone with a desire that is steady and moderate — never peaking, never disappearing.",
    "Imagine someone being caught off guard by the intensity of a desire that was not expected.",
    "Consider someone experiencing the difference between casually wanting something and desperately needing it.",
    "Think about someone with a desire so strong that it crowds out every other consideration.",

    # --- 4. Second-order desire (10) ---
    "Think about someone wanting to stop wanting something that is known to be harmful.",
    "Imagine someone wishing for more motivation but being unable to summon the desire.",
    "Consider someone wanting to care more about something important that currently provokes indifference.",
    "Think about someone wanting to want what another person wants, so that their desires could align.",
    "Imagine someone being frustrated that the desire they feel is for comfort when the desire they value is for growth.",
    "Consider someone trying to cultivate a desire for discipline because it is valued in principle.",
    "Think about someone wanting their desires to be different from what they actually are.",
    "Imagine someone wishing for the desire to exercise even though the actual wanting never arrives.",
    "Consider someone wanting to feel satisfied with what is already present but finding that the wanting for more persists.",
    "Think about someone experiencing the gap between the desires that are actually felt and the desires they wish to feel.",
]

assert len(STANDALONE_PROMPTS_DIM26) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM26)}"

CATEGORY_INFO_STANDALONE_DIM26 = [
    {"name": "appetitive_wanting",      "start": 0,  "end": 10},
    {"name": "aversive_avoidance",      "start": 10, "end": 20},
    {"name": "desire_intensity",        "start": 20, "end": 30},
    {"name": "second_order_desire",     "start": 30, "end": 40},
]
