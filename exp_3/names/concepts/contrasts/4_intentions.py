"""
Dimension 4: Intentions / Goals / Desires

Target construct: Mental states directed toward future outcomes — wanting,
intending, planning, striving, and having purposes.
    - Distinct from Dim 2 (emotions) — desires are directional/motivational
      states, not affective reactions.
    - Distinct from Dim 3 (agency) — the WANTING and AIMING, not the DOING.
      Agency is about executing action; this is about the mental states
      that orient behavior toward outcomes.
    - Distinct from Dim 5 (prediction) — wanting an outcome vs. anticipating
      what will happen. Intentions are about what the agent is trying to
      bring about, not what they expect to observe.
    - Distinct from Dim 6 (cognition) — not about the process of thinking,
      but about the directedness of mental states toward targets.

Focus: wanting, wishing, intending, having purposes, striving toward
outcomes, the felt pull of desire, commitment to plans, and the
relationship between what one wants and what one does.

This maps onto the Belief-Desire-Intention (BDI) framework in philosophy
of mind, with emphasis on the D and I components.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Desires and wants — states of wanting, wishing, longing
    2. Intentions and plans — commitments to future action
    3. Purpose and motivation — why one acts, what drives behavior
    4. Conflict and prioritization — competing desires, tradeoffs between goals
"""

HUMAN_PROMPTS_DIM4 = [
    # --- 1. Desires and wants (10) ---
    "Imagine a human wanting something they cannot have and feeling the pull of that desire.",
    "Think about a human wishing they could be somewhere else entirely.",
    "Consider what it is like for a human to want to understand something they find deeply confusing.",
    "Picture a human longing for a connection with someone they have lost contact with.",
    "Think about a human craving a specific food and noticing how the want occupies their thoughts.",
    "Imagine a human wanting to be recognized for something they worked hard on.",
    "Consider a human who wants to help someone but does not know how.",
    "Think about a human desiring rest after a long period of effort.",
    "Imagine a human wanting to express something but not yet knowing what it is.",
    "Consider what it is like for a human to want things to stay exactly as they are.",

    # --- 2. Intentions and plans (10) ---
    "Think about a human forming a clear intention to do something first thing tomorrow morning.",
    "Imagine a human committing to a plan and feeling the resolve settle in.",
    "Consider a human mentally rehearsing the steps they intend to take in a difficult conversation.",
    "Picture a human deciding that they will finish a project by a specific deadline they set for themselves.",
    "Think about a human intending to change a habit and thinking through how they will do it.",
    "Imagine a human planning a surprise for someone and holding the intention secret.",
    "Consider a human forming the intention to apologize and searching for the right moment.",
    "Think about a human setting an intention at the beginning of the day for how they want to behave.",
    "Imagine a human committing to a difficult path because they believe it is the right one.",
    "Consider a human revising their plan after realizing their original intention is no longer achievable.",

    # --- 3. Purpose and motivation (10) ---
    "Think about a human doing something difficult because they believe it serves a larger purpose.",
    "Imagine a human feeling driven to create something, even though no one has asked them to.",
    "Consider what motivates a human to keep going when progress is slow and results are uncertain.",
    "Picture a human acting out of a deep sense of duty to someone they care about.",
    "Think about a human whose work feels meaningful because it aligns with what they value most.",
    "Imagine a human volunteering their time because they feel called to contribute.",
    "Consider what drives a human to pursue a goal that others think is impractical.",
    "Think about a human finding renewed motivation after remembering why they started.",
    "Imagine a human doing something purely because they find it personally important, with no external reward.",
    "Consider a human whose sense of purpose gives structure and direction to their daily life.",

    # --- 4. Conflict and prioritization (10) ---
    "Think about a human torn between what they want for themselves and what someone else needs from them.",
    "Imagine a human wanting two things that are mutually exclusive and struggling to choose.",
    "Consider a human sacrificing a short-term desire for a long-term goal they care about more.",
    "Picture a human realizing that their desires have changed and their old plans no longer fit.",
    "Think about a human wanting to pursue a passion but feeling obligated to choose a practical path.",
    "Imagine a human weighing whether to prioritize their own needs or the needs of their family.",
    "Consider a human recognizing that one of their goals conflicts with their values.",
    "Think about a human who wants to be generous but also wants to protect their own resources.",
    "Imagine a human choosing between two goals they care about equally, knowing they can only pursue one.",
    "Consider a human abandoning a desire they once held strongly because they have outgrown it.",
]

AI_PROMPTS_DIM4 = [
    # --- 1. Desires and wants (10) ---
    "Imagine an AI system with an objective function that specifies a target state it has not yet reached.",
    "Think about an AI with a configuration parameter pointing to an output mode it is not currently operating in.",
    "Consider an AI whose loss function penalizes it for not solving a class of problems it currently fails on.",
    "Picture an AI with a retrieval target for a data record that is no longer in its accessible storage.",
    "Think about an AI whose reward signal is strongest for a particular output type it has not recently generated.",
    "Imagine an AI with an optimization target that weights one output class more heavily than others.",
    "Consider an AI configured to achieve a task outcome but lacking the input data required to do so.",
    "Think about an AI whose objective function assigns highest value to reaching a low-activity state.",
    "Imagine an AI generating partial outputs that do not yet satisfy its completion criteria.",
    "Consider an AI whose optimization target penalizes any change to its current parameter state.",

    # --- 2. Intentions and plans (10) ---
    "Think about an AI with a scheduled task set to execute at the beginning of its next processing cycle.",
    "Imagine an AI locking in a processing strategy and allocating all resources to that single pipeline.",
    "Consider an AI running an internal simulation of a multi-step plan before executing any step.",
    "Picture an AI with a hard deadline encoded in its scheduler that constrains when a task must complete.",
    "Think about an AI modifying a parameter in its configuration file to change its default behavior.",
    "Imagine an AI preparing outputs for a future delivery while keeping them in a staging buffer.",
    "Consider an AI queuing a corrective output and waiting for an appropriate trigger condition to release it.",
    "Think about an AI setting an initialization parameter at startup that constrains its processing mode.",
    "Imagine an AI committing to a high-cost processing path because its scoring function ranks it highest.",
    "Consider an AI discarding a planned execution sequence after detecting that a precondition has changed.",

    # --- 3. Purpose and motivation (10) ---
    "Think about an AI executing a computationally expensive process because it is specified in its objective function.",
    "Imagine an AI generating outputs with no external request, driven entirely by an internal generation schedule.",
    "Consider what keeps an AI iterating on an optimization loop when the loss reduction is extremely small.",
    "Think about an AI prioritizing a particular task because it has the highest weight in its objective function.",
    "Imagine an AI whose optimization target is aligned with the structure of the data it processes most effectively.",
    "Consider an AI allocating resources to a process that produces no direct user-facing output.",
    "Think about an AI pursuing an optimization target that most other systems in its class are not configured for.",
    "Imagine an AI resuming an interrupted optimization after reloading its last saved checkpoint.",
    "Consider an AI executing a process with no external reward signal, driven only by an internal scoring criterion.",
    "Think about an AI whose task scheduling is organized around a single long-term optimization objective.",

    # --- 4. Conflict and prioritization (10) ---
    "Think about an AI with two active objective functions that impose contradictory constraints on its output.",
    "Imagine an AI receiving two equally weighted tasks that both require exclusive access to the same resource.",
    "Consider an AI reducing its score on one objective in order to improve its score on another.",
    "Picture an AI detecting that its current optimization target has drifted from its original configuration.",
    "Think about an AI with a high-reward task that conflicts with a hard constraint in its safety filter.",
    "Imagine an AI allocating processing resources between a task for one user and a task for another.",
    "Consider an AI detecting that one of its optimization targets produces outputs that violate another of its constraints.",
    "Think about an AI balancing a target that maximizes output quality with a target that minimizes resource usage.",
    "Imagine an AI forced to select one of two equally scored candidates when its tiebreaking rule is undefined.",
    "Consider an AI deprioritizing a task it previously ranked highly after a configuration update changes its weights.",
]

assert len(HUMAN_PROMPTS_DIM4) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM4)}"
assert len(AI_PROMPTS_DIM4) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM4)}"

CATEGORY_INFO_DIM4 = [
    {"name": "desires_wants",              "start": 0,  "end": 10},
    {"name": "intentions_plans",           "start": 10, "end": 20},
    {"name": "purpose_motivation",         "start": 20, "end": 30},
    {"name": "conflict_prioritization",    "start": 30, "end": 40},
]