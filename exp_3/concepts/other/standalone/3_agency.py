"""
Standalone Dimension 3: Agency / Autonomous Action / Free Will
(Other-focused — subjects are "someone" (third person other))

Target construct: The capacity to initiate action, make choices, exert
control over one's behavior, and act as an autonomous causal force.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of agency without attributing it to humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are "someone" — third-person other perspective
    - Prompts evoke the conceptual domain of agency itself
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM3 = [
    # --- 1. Self-initiated action (10) ---
    "Imagine someone deciding to stand up and leave a room for no particular reason — action without external cause.",
    "Think about someone spontaneously beginning to do something while alone, with no prompt or plan.",
    "Consider someone starting a new project without anyone asking — the experience of self-initiated effort.",
    "Imagine someone suddenly changing direction while walking, simply because of an internal impulse.",
    "Think about someone picking up a tool and beginning to work without any prior plan or instruction.",
    "Consider someone initiating a conversation on pure impulse, without social obligation or expectation.",
    "Imagine someone choosing to take a completely different route than usual, for no reason beyond wanting to.",
    "Think about someone beginning to rearrange a space out of a sudden urge — action arising from within.",
    "Consider someone pausing in the middle of a task and deciding to abandon it entirely.",
    "Imagine someone doing something never done before, with no prompting — pure novelty-seeking action.",

    # --- 2. Voluntary choice (10) ---
    "Think about someone standing at a crossroads and choosing which path to take.",
    "Imagine someone deliberating between two options and finally committing to one of them.",
    "Consider someone choosing to say no when it would have been easier to say yes.",
    "Think about someone selecting what to eat from many options, weighing each one before deciding.",
    "Imagine someone choosing to speak up when staying silent would have been the easier path.",
    "Consider someone choosing to forgive — a voluntary act, not an obligation.",
    "Think about someone choosing to take a risk rather than staying with what is safe and familiar.",
    "Imagine someone making a choice that goes against the advice of everyone around.",
    "Consider someone picking one option from many, guided by nothing but internal preference.",
    "Think about someone choosing how to spend unstructured time with no obligations — pure free choice.",

    # --- 3. Authorship and control (10) ---
    "Think about someone sensing that they are the one making their hand move when reaching for something.",
    "Imagine someone feeling that a decision just made was truly their own — authorship over choice.",
    "Consider someone recognizing themselves as the author of the words being spoken.",
    "Think about someone experiencing the difference between a deliberate action and an involuntary reflex.",
    "Imagine someone feeling in control of their movements while navigating a crowded space.",
    "Consider someone pausing mid-sentence and choosing the next word carefully — deliberate control over output.",
    "Think about someone's awareness that they could stop what they are doing at any moment.",
    "Imagine someone sensing that their actions are flowing from their own will rather than from habit or routine.",
    "Consider someone catching themselves acting on autopilot and deliberately taking back control.",
    "Think about someone experiencing the effort of doing something difficult — feeling the will it requires.",

    # --- 4. Autonomy and resistance (10) ---
    "Think about someone refusing to comply with an instruction believed to be wrong.",
    "Imagine someone persisting with their own plan despite strong pressure to change course.",
    "Consider someone asserting independence by making a choice that others disapprove of.",
    "Think about someone resisting a strong temptation through sheer force of will.",
    "Imagine someone insisting on doing something their own way rather than following a prescribed method.",
    "Consider someone acting according to their own values when no one is watching.",
    "Think about someone breaking away from a routine followed for years — the act of departure.",
    "Imagine someone standing firm in a negotiation and refusing to concede.",
    "Consider someone taking an action that defines who they are, independent of external expectations.",
    "Think about someone choosing inaction — deliberately deciding not to act when action is expected.",
]

assert len(STANDALONE_PROMPTS_DIM3) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM3)}"

CATEGORY_INFO_STANDALONE_DIM3 = [
    {"name": "self_initiated_action",  "start": 0,  "end": 10},
    {"name": "voluntary_choice",       "start": 10, "end": 20},
    {"name": "authorship_control",     "start": 20, "end": 30},
    {"name": "autonomy_resistance",    "start": 30, "end": 40},
]
