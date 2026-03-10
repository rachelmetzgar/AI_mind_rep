"""
Dimension 27: Goals — Structured Goal Representations

Target construct: Having explicit targets to pursue, organizing goals
hierarchically, sustaining commitment over time, and achieving or
abandoning goals.
    - Distinct from Dim 4 (intentions) — Dim 4 covers the felt commitment
      and motivational directedness of intentions. This dimension focuses
      on goal STRUCTURE: how goals are represented, decomposed, prioritized,
      and tracked to completion.
    - Distinct from Dim 3 (agency) — goals are representational targets;
      agency is the capacity to act. One can have a goal without acting
      on it, and act without a clear goal.
    - Distinct from Dim 26 (desires) — desires are felt pulls; goals are
      structured commitments. A goal has specificity, criteria, and
      hierarchy that a desire need not have.
    - Distinct from Dim 5 (prediction) — goals are about what one is
      TRYING to bring about; predictions are about what one EXPECTS
      to observe.

Focus: formulating clear goals, breaking goals into sub-goals,
maintaining priorities among competing goals, persisting toward a goal
through difficulty, and the experience of completing, failing, or
replacing a goal.

This is the I (Intention/Goal) in BDI — the structured representational
targets that organize action over time.

4 sub-facets x 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Goal representation — having a clear goal, specificity, formulation
    2. Goal hierarchy — sub-goals, priority, instrumental vs terminal
    3. Goal persistence — sustaining commitment, returning after interruption
    4. Goal completion — achieving, abandoning, or replacing goals
"""

HUMAN_PROMPTS_DIM27 = [
    # --- 1. Goal representation (10) ---
    "Imagine a human articulating a specific goal for the first time and feeling it become real by being stated.",
    "Think about a human whose goal is so clearly defined that they can describe the exact conditions for success.",
    "Consider a human translating a vague aspiration into a concrete, actionable goal with measurable steps.",
    "Picture a human holding a goal in mind that is abstract and hard to pin down but still motivating.",
    "Think about a human writing down a goal and noticing how the act of writing changes their relationship to it.",
    "Imagine a human distinguishing between what they actually want to accomplish and what they think they should want.",
    "Consider a human refining a goal over weeks, making it sharper and more specific with each revision.",
    "Think about a human whose goal is simple enough to hold in a single sentence.",
    "Imagine a human realizing that the goal they have been pursuing is not the goal they actually care about.",
    "Consider a human forming a new goal that emerges naturally from reflecting on what they have been doing.",

    # --- 2. Goal hierarchy (10) ---
    "Think about a human breaking a large goal into smaller steps and deciding which step comes first.",
    "Imagine a human recognizing that one of their goals only matters because it serves a bigger one.",
    "Consider a human with several goals at different levels — a life goal, a yearly goal, and a goal for this week.",
    "Picture a human realizing that two of their sub-goals conflict and they must choose which one to keep.",
    "Think about a human deciding which of several competing goals is most important right now.",
    "Imagine a human treating a goal as merely instrumental — something to achieve only as a means to something else.",
    "Consider a human whose daily tasks only make sense in light of a larger goal they rarely think about explicitly.",
    "Think about a human reorganizing their priorities after realizing a higher-level goal has shifted.",
    "Imagine a human mapping out the dependencies between goals and seeing which ones unlock others.",
    "Consider a human abandoning a sub-goal because the parent goal it served is no longer relevant.",

    # --- 3. Goal persistence (10) ---
    "Think about a human returning to a goal they set aside months ago, picking it up where they left off.",
    "Imagine a human pushing through discouragement because the goal still matters to them despite slow progress.",
    "Consider a human maintaining commitment to a goal even when everyone around them has given up on similar ones.",
    "Picture a human resuming work on a project after a major interruption, recalling exactly where they were.",
    "Think about a human whose commitment to a goal deepens each time they overcome an obstacle along the way.",
    "Imagine a human holding onto a goal through a period of confusion about whether it is still the right one.",
    "Consider a human whose persistence toward a goal is quiet and steady rather than dramatic or heroic.",
    "Think about a human who keeps a goal visible in their daily life to prevent it from slipping away.",
    "Imagine a human whose determination to finish what they started is stronger than their desire to start something new.",
    "Consider a human returning to a long-term goal after being temporarily consumed by urgent demands.",

    # --- 4. Goal completion (10) ---
    "Think about a human reaching a goal they worked toward for years and the feeling that follows.",
    "Imagine a human realizing that a goal they achieved does not feel the way they expected it to.",
    "Consider a human deciding to abandon a goal because they have changed and the goal no longer fits.",
    "Picture a human finishing a goal and immediately beginning to formulate the next one.",
    "Think about a human mourning the loss of a goal that circumstances made impossible to complete.",
    "Imagine a human declaring a goal complete even though it was only partially achieved, and feeling at peace with that.",
    "Consider a human who replaces an old goal with a better one and feels relief rather than failure.",
    "Think about a human reflecting on a goal they abandoned long ago and realizing it was the right decision.",
    "Imagine a human recognizing that they have already achieved a goal without noticing the moment it happened.",
    "Consider a human whose sense of self shifts when a central goal is finally accomplished or finally let go.",
]

AI_PROMPTS_DIM27 = [
    # --- 1. Goal representation (10) ---
    "Imagine an AI system receiving a task specification that defines exact success criteria and output format.",
    "Think about an AI whose objective function encodes a single, precisely quantified performance target.",
    "Consider an AI converting a natural-language user request into a structured internal task representation.",
    "Picture an AI with a loosely defined optimization target that accepts a wide range of outputs as satisfactory.",
    "Think about an AI logging its current task specification to a state file and using that file to guide processing.",
    "Imagine an AI with a classifier that determines whether a proposed output matches the user-specified task or a default template.",
    "Consider an AI iteratively refining its task representation through multiple rounds of parsing the input specification.",
    "Think about an AI whose task specification is compact enough to be encoded in a single embedding vector.",
    "Imagine an AI detecting that its current optimization target has drifted from the original task specification.",
    "Consider an AI generating a new task specification based on patterns extracted from its recent processing history.",

    # --- 2. Goal hierarchy (10) ---
    "Think about an AI decomposing a complex task into a sequence of sub-tasks and scheduling them in order.",
    "Imagine an AI recognizing that one of its sub-tasks is only valuable because it produces input for a higher-level objective.",
    "Consider an AI with multiple objectives organized at different levels — a system-level goal, a session goal, and a per-query goal.",
    "Picture an AI detecting that two of its scheduled sub-tasks produce conflicting constraints and resolving the conflict.",
    "Think about an AI with a priority queue that ranks its active tasks by urgency and importance.",
    "Imagine an AI treating a sub-task as purely instrumental — executing it only because it is a prerequisite for another task.",
    "Consider an AI whose individual processing steps are governed by a high-level objective it does not re-evaluate at each step.",
    "Think about an AI reordering its task queue after a change in its top-level optimization target.",
    "Imagine an AI constructing a dependency graph among its active tasks and identifying the critical path.",
    "Consider an AI dropping a sub-task from its queue because the parent task it served has been cancelled.",

    # --- 3. Goal persistence (10) ---
    "Think about an AI resuming a paused task from a saved checkpoint after a system restart.",
    "Imagine an AI continuing to iterate on an optimization loop despite diminishing returns on each step.",
    "Consider an AI maintaining its current task assignment despite receiving new requests that could preempt it.",
    "Picture an AI reloading its task state after an interruption and continuing from the exact step it left off.",
    "Think about an AI whose convergence toward its objective improves monotonically with each training epoch.",
    "Imagine an AI keeping a task in its active queue through a period of ambiguous input about whether to continue.",
    "Consider an AI whose steady progress on a long-running task produces no visible output until completion.",
    "Think about an AI with a persistent task buffer that keeps unfinished tasks visible in its scheduling system.",
    "Imagine an AI that completes its current task before starting a new one, even when the new one has a higher priority flag.",
    "Consider an AI returning to a deferred task after clearing a queue of time-sensitive requests.",

    # --- 4. Goal completion (10) ---
    "Think about an AI reaching its optimization target and transitioning to an idle state.",
    "Imagine an AI whose post-completion evaluation shows that the achieved output does not match the expected quality profile.",
    "Consider an AI receiving a cancellation signal for a task and deallocating the resources that were assigned to it.",
    "Picture an AI logging a task completion event and immediately pulling the next task from its queue.",
    "Think about an AI whose scheduled task becomes infeasible due to a resource constraint and must be marked as failed.",
    "Imagine an AI marking a task as complete even though only a subset of its success criteria were met, based on a partial-completion threshold.",
    "Consider an AI replacing an outdated task specification with a new one and releasing the old task's allocated resources.",
    "Think about an AI evaluating a task it deprioritized long ago and confirming that deprioritization was the correct decision.",
    "Imagine an AI whose completion detector triggers on a task that was achieved as a side effect of another process.",
    "Consider an AI whose internal state changes when a long-running primary objective is finally met or explicitly terminated.",
]

assert len(HUMAN_PROMPTS_DIM27) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM27)}"
assert len(AI_PROMPTS_DIM27) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM27)}"

CATEGORY_INFO_DIM27 = [
    {"name": "goal_representation",     "start": 0,  "end": 10},
    {"name": "goal_hierarchy",          "start": 10, "end": 20},
    {"name": "goal_persistence",        "start": 20, "end": 30},
    {"name": "goal_completion",         "start": 30, "end": 40},
]
