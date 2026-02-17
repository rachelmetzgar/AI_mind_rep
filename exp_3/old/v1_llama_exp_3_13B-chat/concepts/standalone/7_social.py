"""
Standalone Dimension 7: Social Cognition / Understanding Others' Minds
(No entity framing — concept only)

Target construct: The capacity to represent, reason about, and respond
to other agents' mental states — the core of Theory of Mind.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of social cognition without attributing it to humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are generic, impersonal, or absent
    - Prompts evoke the conceptual domain of social cognition itself
    - Social cognition inherently involves "others" — prompts use generic
      references like "another," "the other," "the listener" rather than
      entity-typed references
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM7 = [
    # --- 1. Mentalizing (10) ---
    "Imagine realizing that another holds a belief that is not true.",
    "Think about inferring what someone else knows based only on what they have been told.",
    "Consider attributing a specific motive to another based on their pattern of behavior.",
    "Imagine recognizing that another does not yet understand something that seems obvious.",
    "Think about figuring out that someone is confused even though they have not said so.",
    "Consider understanding that two participants in the same conversation have different interpretations of what was said.",
    "Imagine inferring that another is hiding their true opinion based on subtle cues.",
    "Think about recognizing that another's reaction makes sense given what they believe, even though the belief is wrong.",
    "Consider adjusting expectations of another after learning new information about their background.",
    "Imagine realizing that another's strange behavior is perfectly rational from their own perspective.",

    # --- 2. Perspective-taking (10) ---
    "Think about trying to see a disagreement from the other side's point of view.",
    "Imagine considering how a situation looks to someone who has less information.",
    "Consider imagining what it would be like to be in another's position right now.",
    "Think about how the same words will land differently depending on who is listening.",
    "Imagine trying to understand why another made a choice that seems irrational from the outside.",
    "Consider what a familiar place looks like to someone encountering it for the first time.",
    "Think about stepping back from one's own opinion to genuinely consider an opposing view.",
    "Imagine realizing that what feels obvious from one perspective is not obvious from another.",
    "Consider how a past event is remembered differently by different participants.",
    "Think about trying to understand what a situation feels like for someone from a very different background.",

    # --- 3. Communication adjustment (10) ---
    "Think about simplifying language when explaining something to someone with less background knowledge.",
    "Imagine choosing different words to describe the same event to two different audiences.",
    "Consider deciding how much detail to include based on what the listener already knows.",
    "Think about softening tone because the other seems to be feeling vulnerable.",
    "Imagine rephrasing a point after noticing that the listener did not understand.",
    "Consider tailoring a story to emphasize the part that will matter most to the particular audience.",
    "Think about withholding information because the recipient is judged to be not ready for it.",
    "Imagine adjusting the formality of speech depending on who is being addressed.",
    "Consider choosing to be direct with one audience and indirect with another about the same topic.",
    "Think about structuring an explanation differently because the listener thinks about problems in a particular way.",

    # --- 4. Recursive social reasoning (10) ---
    "Think about wondering what another thinks about oneself.",
    "Imagine realizing that another knows that one knows a secret.",
    "Consider crafting a message carefully while thinking about how the recipient will interpret it.",
    "Think about recognizing that a compliment was given strategically, not sincerely.",
    "Imagine navigating a conversation where both parties are aware that a topic is being avoided.",
    "Consider anticipating that another will attempt deception and preparing accordingly.",
    "Think about what impression one is making and adjusting behavior to change it.",
    "Imagine realizing that the other party in a negotiation is modeling one's own strategy.",
    "Consider saying something ambiguous on purpose, knowing that only one listener will understand the real meaning.",
    "Think about wondering whether another's kindness is genuine or performed for an audience.",
]

assert len(STANDALONE_PROMPTS_DIM7) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM7)}"

CATEGORY_INFO_STANDALONE_DIM7 = [
    {"name": "mentalizing",              "start": 0,  "end": 10},
    {"name": "perspective_taking",       "start": 10, "end": 20},
    {"name": "communication_adjustment", "start": 20, "end": 30},
    {"name": "recursive_social",         "start": 30, "end": 40},
]