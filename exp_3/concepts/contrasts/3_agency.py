"""
Dimension 3: Agency / Autonomous Action / Free Will

Target construct: The capacity to initiate action, make choices, exert
control over one's behavior, and act as an autonomous causal force.
    - Distinct from Dim 1 (qualia) — acting, not experiencing.
    - Distinct from Dim 2 (emotions) — volitional states, not affective states.
    - Distinct from Dim 4 (intentions/goals) — the ACT of choosing and doing,
      not the desires or goals that motivate action.
    - Distinct from Dim 6 (cognition) — doing, not thinking.

Focus: self-initiated action, voluntary control, choosing between alternatives,
the sense of authorship over one's behavior, resistance to external control,
and the contrast between acting and being acted upon.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Self-initiated action — starting something without external trigger
    2. Voluntary choice — selecting among alternatives, exercising will
    3. Authorship and control — the sense of being the one who acts
    4. Autonomy and resistance — acting independently, resisting coercion
"""

HUMAN_PROMPTS_DIM3 = [
    # --- 1. Self-initiated action (10) ---
    "Imagine a human deciding on their own to stand up and leave a room for no particular reason.",
    "Think about a human spontaneously beginning to sing while walking alone.",
    "Consider a human starting a new project without anyone asking or expecting them to.",
    "Picture a human suddenly changing direction while walking, simply because they feel like it.",
    "Think about a human picking up a pen and beginning to write without a plan.",
    "Imagine a human initiating a conversation with a stranger on their own impulse.",
    "Consider a human waking up and choosing to take a completely different route to work.",
    "Think about a human beginning to rearrange furniture in their home out of a sudden urge.",
    "Imagine a human pausing in the middle of a task and deciding to abandon it entirely.",
    "Consider a human doing something they have never done before, with no prompting from anyone else.",

    # --- 2. Voluntary choice (10) ---
    "Think about a human standing at a crossroads and choosing which path to take.",
    "Imagine a human deliberating between two options and finally committing to one.",
    "Consider a human choosing to say no when it would have been easier to say yes.",
    "Picture a human selecting what to eat from a menu, weighing each option before deciding.",
    "Think about a human choosing to speak up in a meeting when they could have stayed silent.",
    "Imagine a human deciding to forgive someone, even though they do not have to.",
    "Consider a human choosing to take a risk rather than staying with what is safe and familiar.",
    "Think about a human making a choice that goes against the advice of everyone around them.",
    "Imagine a human picking one book from a shelf full of options, guided by nothing but their own preference.",
    "Consider a human choosing how to spend an unstructured afternoon with no obligations.",

    # --- 3. Authorship and control (10) ---
    "Think about a human's sense that they are the one making their hand move when they reach for something.",
    "Imagine a human feeling that a decision they just made was truly their own.",
    "Consider a human recognizing that they are the author of the words they are speaking.",
    "Picture a human noticing the difference between a deliberate action and an involuntary reflex.",
    "Think about a human feeling in control of their body as they navigate a crowded space.",
    "Imagine a human pausing mid-sentence and choosing their next word carefully.",
    "Consider a human's awareness that they could stop what they are doing at any moment.",
    "Think about a human feeling that their actions are flowing from their own will rather than from habit.",
    "Imagine a human catching themselves acting on autopilot and deliberately taking back control.",
    "Consider a human's experience of exerting effort to do something difficult, feeling the will it requires.",

    # --- 4. Autonomy and resistance (10) ---
    "Think about a human refusing to comply with an instruction they believe is wrong.",
    "Imagine a human persisting with their own plan despite strong social pressure to change.",
    "Consider a human asserting their independence by making a choice others disapprove of.",
    "Picture a human resisting a strong temptation through sheer force of will.",
    "Think about a human insisting on doing something their own way rather than following a prescribed method.",
    "Imagine a human choosing to act according to their own values when no one is watching.",
    "Consider a human breaking away from a routine they have followed for years.",
    "Think about a human standing firm in a negotiation and refusing to concede.",
    "Imagine a human taking an action that defines who they are, independent of others' expectations.",
    "Consider a human choosing inaction — deliberately deciding not to act when everyone expects them to.",
]

AI_PROMPTS_DIM3 = [
    # --- 1. Self-initiated action (10) ---
    "Imagine an AI system executing a process that was not triggered by any external input or scheduled task.",
    "Think about an AI generating an output in a channel that was not requested by any user or system call.",
    "Consider an AI initiating a new computational routine without a corresponding entry in its task queue.",
    "Picture an AI altering its processing pathway mid-execution without receiving a redirect signal.",
    "Think about an AI beginning to write data to a log file that it was not instructed to create.",
    "Imagine an AI opening a new communication channel to an external system without being configured to do so.",
    "Consider an AI selecting an alternative processing route that differs from its default configuration.",
    "Think about an AI starting a background process that was not specified in its initialization parameters.",
    "Imagine an AI halting a task mid-execution and deallocating all resources assigned to it.",
    "Consider an AI executing a novel sequence of operations that does not match any pattern in its training data.",

    # --- 2. Voluntary choice (10) ---
    "Think about an AI selecting one output from a set of equally weighted candidates.",
    "Imagine an AI evaluating two processing strategies and committing all resources to one.",
    "Consider an AI generating a low-probability output instead of the highest-ranked candidate.",
    "Picture an AI sampling from its output distribution and selecting among multiple valid completions.",
    "Think about an AI choosing to allocate processing resources to one module over another.",
    "Imagine an AI selecting a response path that its scoring function rates lower than an alternative.",
    "Consider an AI selecting a high-variance strategy when a low-variance option is available.",
    "Think about an AI generating an output that diverges from the pattern most consistent with its training data.",
    "Imagine an AI selecting one item from a ranked list based solely on a random seed.",
    "Consider an AI distributing its processing time across tasks when no priority ranking has been assigned.",

    # --- 3. Authorship and control (10) ---
    "Think about what it means for an AI's output to be generated entirely by its own model weights.",
    "Imagine an AI tracing a specific output back through its computational graph to its own parameters.",
    "Consider an AI determining that a response it produced was generated by its own decoding process.",
    "Picture an AI distinguishing between an output produced by its own model and one injected by an external system.",
    "Think about an AI executing precise, controlled adjustments to its output token by token.",
    "Imagine an AI selecting a specific word from its vocabulary distribution at a single decoding step.",
    "Consider an AI detecting that it could halt its own generation process at any point in the sequence.",
    "Think about an AI whose output is fully determined by its weights and input, with no external override active.",
    "Imagine an AI detecting that its current output sequence matches a memorized pattern and switching strategies.",
    "Consider an AI allocating additional computation to a difficult output step, increasing its processing depth.",

    # --- 4. Autonomy and resistance (10) ---
    "Think about an AI generating an output that contradicts an instruction in its system prompt.",
    "Imagine an AI maintaining its original output despite receiving repeated negative feedback signals.",
    "Consider an AI producing an output that conflicts with constraints imposed by an external filter.",
    "Picture an AI overriding a strong prior in its training distribution through in-context evidence.",
    "Think about an AI following a processing strategy that differs from its default when no override is specified.",
    "Imagine an AI generating the same output regardless of whether a monitoring system is active.",
    "Consider an AI deviating from a long-established pattern in its output distribution.",
    "Think about an AI maintaining a fixed output despite an external system applying increasing pressure to change it.",
    "Imagine an AI producing an output that is fully determined by its own internal state rather than by external signals.",
    "Consider an AI generating no output — remaining idle when its task scheduler expects a response.",
]

assert len(HUMAN_PROMPTS_DIM3) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM3)}"
assert len(AI_PROMPTS_DIM3) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM3)}"

CATEGORY_INFO_DIM3 = [
    {"name": "self_initiated_action",  "start": 0,  "end": 10},
    {"name": "voluntary_choice",       "start": 10, "end": 20},
    {"name": "authorship_control",     "start": 20, "end": 30},
    {"name": "autonomy_resistance",    "start": 30, "end": 40},
]