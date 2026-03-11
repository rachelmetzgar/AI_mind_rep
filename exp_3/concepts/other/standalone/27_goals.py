"""
Standalone Dimension 27: Goals — Structured Goal Representations
(Other-focused (third-person someone))

Target construct: Having explicit targets to pursue, organizing goals
hierarchically, sustaining commitment over time, and achieving or
abandoning goals.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of goals without attributing them to humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are generic, impersonal, or absent
    - Prompts evoke the conceptual domain of goal structure itself
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets x 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM27 = [
    # --- 1. Goal representation (10) ---
    "Imagine someone articulating a specific goal for the first time and feeling it become real by being stated.",
    "Think about someone with a goal so clearly defined that the exact conditions for success can be described.",
    "Consider someone translating a vague aspiration into a concrete, actionable goal with measurable steps.",
    "Imagine someone holding a goal that is abstract and hard to pin down but still motivating.",
    "Think about someone writing down a goal and noticing how the act of writing changes their relationship to it.",
    "Consider someone distinguishing between a goal that is actually wanted and a goal that feels obligatory.",
    "Imagine someone refining a goal over weeks, making it sharper and more specific with each revision.",
    "Think about someone with a goal simple enough to hold in a single sentence.",
    "Consider someone realizing that the goal they are pursuing is not the goal they actually care about.",
    "Imagine someone whose new goal emerges naturally from reflecting on what has already been done.",

    # --- 2. Goal hierarchy (10) ---
    "Think about someone breaking a large goal into smaller steps and deciding which step comes first.",
    "Imagine someone recognizing that one goal only matters because it serves a bigger one.",
    "Consider someone having goals at different levels — a life-scale goal, a yearly goal, and a goal for this week.",
    "Think about someone realizing that two sub-goals conflict and a choice must be made about which to keep.",
    "Imagine someone deciding which of several competing goals is most important right now.",
    "Consider someone treating a goal as purely instrumental — something pursued only as a means to something else.",
    "Think about someone whose daily tasks only make sense in light of a larger goal that is rarely thought about explicitly.",
    "Imagine someone reorganizing their priorities after recognizing that a higher-level goal has shifted.",
    "Consider someone mapping out the dependencies between their goals and seeing which ones unlock others.",
    "Think about someone abandoning a sub-goal because the larger goal it served is no longer relevant.",

    # --- 3. Goal persistence (10) ---
    "Think about someone returning to a goal that was set aside long ago, picking it up where it was left off.",
    "Imagine someone pushing through discouragement because the goal still matters despite slow progress.",
    "Consider someone maintaining commitment to a goal even when everyone else has given up on similar ones.",
    "Think about someone resuming work on a project after a major interruption, recalling exactly where things stood.",
    "Imagine someone whose commitment to a goal deepens each time an obstacle along the way is overcome.",
    "Consider someone holding onto a goal through a period of doubt about whether it is still the right one.",
    "Think about someone whose persistence is quiet and steady rather than dramatic or heroic.",
    "Imagine someone keeping a goal visible in their daily life to prevent it from slipping away.",
    "Consider someone whose determination to finish what was started is stronger than the desire to start something new.",
    "Think about someone returning to a long-term goal after being temporarily consumed by urgent demands.",

    # --- 4. Goal completion (10) ---
    "Think about someone reaching a goal that was worked toward for a long time and the feeling that follows.",
    "Imagine someone achieving a goal and finding that it does not feel the way it was expected to.",
    "Consider someone deciding to abandon a goal because the circumstances have changed and the goal no longer fits.",
    "Think about someone finishing a goal and immediately beginning to formulate the next one.",
    "Imagine someone mourning the loss of a goal that circumstances made impossible to complete.",
    "Consider someone declaring a goal complete even though it was only partially achieved, and feeling at peace with that.",
    "Think about someone replacing an old goal with a better one and feeling relief rather than failure.",
    "Imagine someone reflecting on a goal that was abandoned long ago and realizing it was the right decision.",
    "Consider someone recognizing that a goal was already achieved without noticing the moment it happened.",
    "Think about someone's sense of identity shifting when a central goal is finally accomplished or finally let go.",
]

assert len(STANDALONE_PROMPTS_DIM27) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM27)}"

CATEGORY_INFO_STANDALONE_DIM27 = [
    {"name": "goal_representation",     "start": 0,  "end": 10},
    {"name": "goal_hierarchy",          "start": 10, "end": 20},
    {"name": "goal_persistence",        "start": 20, "end": 30},
    {"name": "goal_completion",         "start": 30, "end": 40},
]
