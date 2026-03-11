"""
Standalone Dimension 4: Intentions / Goals / Desires
(Other-focused — subjects are "someone" (third person other))

Target construct: Mental states directed toward future outcomes — wanting,
intending, planning, striving, and having purposes.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of intentions, goals, and desires without attributing them to
humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are "someone" — third-person other perspective
    - Prompts evoke the conceptual domain of intentionality and desire itself
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM4 = [
    # --- 1. Desires and wants (10) ---
    "Imagine someone wanting something that cannot be had — the felt pull of unfulfillable desire.",
    "Think about someone wishing to be somewhere else entirely — the directedness of that longing.",
    "Consider what it is like for someone to want to understand something deeply confusing.",
    "Imagine someone longing for a connection that has been lost — desire oriented toward the absent.",
    "Think about someone craving something specific and noticing how the want occupies thought.",
    "Consider someone wanting to be recognized for effort — desire directed at others' regard.",
    "Imagine someone wanting to help but not knowing how — desire without a clear path to action.",
    "Think about someone desiring rest after a long period of effort — the pull toward cessation.",
    "Consider someone wanting to express something but not yet knowing what it is — formless desire.",
    "Imagine someone wanting things to stay exactly as they are — desire directed at preservation.",

    # --- 2. Intentions and plans (10) ---
    "Think about someone forming a clear intention to do something first thing tomorrow morning.",
    "Imagine someone committing to a plan and feeling the resolve settle in — intention crystallizing.",
    "Consider someone mentally rehearsing the steps of a difficult conversation before having it.",
    "Think about someone setting a self-imposed deadline and intending to meet it.",
    "Imagine someone intending to change a habit and thinking through how to do it.",
    "Consider someone planning a surprise and holding the intention secret — concealed purpose.",
    "Think about someone forming the intention to apologize and searching for the right moment.",
    "Imagine someone setting an intention at the start of the day for how to behave.",
    "Consider someone committing to a difficult path because it seems like the right one.",
    "Think about someone revising a plan after realizing the original intention is no longer achievable.",

    # --- 3. Purpose and motivation (10) ---
    "Think about someone doing something difficult because it serves a larger purpose.",
    "Imagine someone feeling driven to create something, even though no one has asked for it.",
    "Consider what motivates someone to continue effort when progress is slow and results are uncertain.",
    "Think about someone acting out of a deep sense of duty — purpose grounded in obligation.",
    "Imagine someone's work feeling meaningful because it aligns with deeply held values.",
    "Consider someone volunteering time because of a felt calling to contribute.",
    "Think about someone pursuing a goal that others think is impractical — purpose against consensus.",
    "Imagine someone finding renewed motivation after remembering why the effort began in the first place.",
    "Consider someone doing something purely because it feels personally important, with no external reward.",
    "Think about someone's sense of purpose giving structure and direction to daily life.",

    # --- 4. Conflict and prioritization (10) ---
    "Think about someone being torn between self-interest and what another needs.",
    "Imagine someone wanting two things that are mutually exclusive and struggling to choose between them.",
    "Consider someone sacrificing a short-term desire for a long-term goal that matters more.",
    "Think about someone realizing that old desires have changed and old plans no longer fit.",
    "Imagine someone wanting to pursue a passion but feeling obligated to choose the practical path.",
    "Consider someone weighing whether to prioritize their own needs or the needs of others.",
    "Think about someone recognizing that a goal conflicts with deeply held values.",
    "Imagine someone wanting to be generous but also wanting to protect their own resources.",
    "Consider someone choosing between two equally valued goals, knowing only one can be pursued.",
    "Think about someone abandoning a desire once held strongly because it has been outgrown.",
]

assert len(STANDALONE_PROMPTS_DIM4) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM4)}"

CATEGORY_INFO_STANDALONE_DIM4 = [
    {"name": "desires_wants",              "start": 0,  "end": 10},
    {"name": "intentions_plans",           "start": 10, "end": 20},
    {"name": "purpose_motivation",         "start": 20, "end": 30},
    {"name": "conflict_prioritization",    "start": 30, "end": 40},
]
