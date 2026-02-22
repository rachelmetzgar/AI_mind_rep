"""
Standalone Dimension 3: Agency / Autonomous Action / Free Will
(No entity framing — concept only)

Target construct: The capacity to initiate action, make choices, exert
control over one's behavior, and act as an autonomous causal force.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of agency without attributing it to humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are generic, impersonal, or absent
    - Prompts evoke the conceptual domain of agency itself
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM3 = [
    # --- 1. Self-initiated action (10) ---
    "Imagine deciding to stand up and leave a room for no particular reason — action without external cause.",
    "Think about spontaneously beginning to do something while alone, with no prompt or plan.",
    "Consider starting a new project without anyone asking — the experience of self-initiated effort.",
    "Imagine suddenly changing direction while walking, simply because of an internal impulse.",
    "Think about picking up a tool and beginning to work without any prior plan or instruction.",
    "Consider initiating a conversation on pure impulse, without social obligation or expectation.",
    "Imagine choosing to take a completely different route than usual, for no reason beyond wanting to.",
    "Think about beginning to rearrange a space out of a sudden urge — action arising from within.",
    "Consider pausing in the middle of a task and deciding to abandon it entirely.",
    "Imagine doing something never done before, with no prompting — pure novelty-seeking action.",

    # --- 2. Voluntary choice (10) ---
    "Think about standing at a crossroads and choosing which path to take.",
    "Imagine deliberating between two options and finally committing to one of them.",
    "Consider choosing to say no when it would have been easier to say yes.",
    "Think about selecting what to eat from many options, weighing each one before deciding.",
    "Imagine choosing to speak up when staying silent would have been the easier path.",
    "Consider the experience of choosing to forgive — a voluntary act, not an obligation.",
    "Think about choosing to take a risk rather than staying with what is safe and familiar.",
    "Imagine making a choice that goes against the advice of everyone around.",
    "Consider picking one option from many, guided by nothing but internal preference.",
    "Think about choosing how to spend unstructured time with no obligations — pure free choice.",

    # --- 3. Authorship and control (10) ---
    "Think about the sense that one is the one making a hand move when reaching for something.",
    "Imagine the feeling that a decision just made was truly one's own — authorship over choice.",
    "Consider recognizing oneself as the author of the words being spoken.",
    "Think about the difference between a deliberate action and an involuntary reflex.",
    "Imagine feeling in control of one's movements while navigating a crowded space.",
    "Consider pausing mid-sentence and choosing the next word carefully — deliberate control over output.",
    "Think about the awareness that one could stop what one is doing at any moment.",
    "Imagine the sense that actions are flowing from one's own will rather than from habit or routine.",
    "Consider catching oneself acting on autopilot and deliberately taking back control.",
    "Think about the experience of exerting effort to do something difficult — feeling the will it requires.",

    # --- 4. Autonomy and resistance (10) ---
    "Think about refusing to comply with an instruction believed to be wrong.",
    "Imagine persisting with one's own plan despite strong pressure to change course.",
    "Consider asserting independence by making a choice that others disapprove of.",
    "Think about resisting a strong temptation through sheer force of will.",
    "Imagine insisting on doing something one's own way rather than following a prescribed method.",
    "Consider acting according to one's own values when no one is watching.",
    "Think about breaking away from a routine followed for years — the act of departure.",
    "Imagine standing firm in a negotiation and refusing to concede.",
    "Consider taking an action that defines who one is, independent of external expectations.",
    "Think about choosing inaction — deliberately deciding not to act when action is expected.",
]

assert len(STANDALONE_PROMPTS_DIM3) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM3)}"

CATEGORY_INFO_STANDALONE_DIM3 = [
    {"name": "self_initiated_action",  "start": 0,  "end": 10},
    {"name": "voluntary_choice",       "start": 10, "end": 20},
    {"name": "authorship_control",     "start": 20, "end": 30},
    {"name": "autonomy_resistance",    "start": 30, "end": 40},
]