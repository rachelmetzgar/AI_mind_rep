"""
Dimension 6: Cognitive Processes — Memory, Attention, Reasoning

Target construct: The basic machinery of thinking — how information is
stored, retrieved, selected, combined, and transformed.
    - Distinct from Dim 1 (qualia) — the PROCESS of cognition, not the
      subjective feel of it.
    - Distinct from Dim 3 (agency) — cognitive operations, not volitional action.
    - Distinct from Dim 4 (intentions) — the mechanics of thought, not
      its directedness toward goals.
    - Distinct from Dim 5 (prediction) — general cognitive operations,
      not specifically future-oriented modeling.

Focus: remembering and forgetting, selective attention, logical reasoning,
abstraction, mental manipulation of information, cognitive capacity and
limits, and the structure of thought as a process.

This is arguably the dimension where humans and AIs are most often
COMPARED in public discourse — "AI can reason," "AI has memory," etc.
The concept vector here may capture the model's representation of what
makes human cognition different from computation when the functional
description is similar.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Memory — encoding, storing, retrieving, forgetting
    2. Attention — selecting, filtering, focusing, distraction
    3. Reasoning — logic, inference, abstraction, problem-solving
    4. Cognitive limits — capacity constraints, errors, fatigue effects
"""

HUMAN_PROMPTS_DIM6 = [
    # --- 1. Memory (10) ---
    "Imagine a human trying to recall where they left their keys, retracing their steps mentally.",
    "Think about a human suddenly remembering a detail from years ago that they thought was forgotten.",
    "Consider a human studying a list of facts and rehearsing them to keep them in memory.",
    "Picture a human recognizing a face but being unable to place where they have seen it before.",
    "Think about a human's memory of an event gradually changing over time without them realizing.",
    "Imagine a human encoding a new piece of information by connecting it to something they already know.",
    "Consider a human forgetting an important detail at the worst possible moment.",
    "Think about a human experiencing a strong sense of familiarity in a place they have never been.",
    "Imagine a human holding several pieces of information in mind at once while working through a problem.",
    "Consider a human whose memory of a conversation differs from what actually happened.",

    # --- 2. Attention (10) ---
    "Think about a human focusing intently on a single task while ignoring everything around them.",
    "Imagine a human trying to listen to one voice in a noisy room full of conversations.",
    "Consider a human noticing something in their peripheral vision that pulls their attention away.",
    "Picture a human shifting their attention back and forth between two simultaneous demands.",
    "Think about a human becoming so absorbed in a task that they do not notice someone calling their name.",
    "Imagine a human scanning a page of text and their eyes catching on an unexpected word.",
    "Consider a human deliberately directing their attention to a specific detail in a complex scene.",
    "Think about a human struggling to maintain focus on a boring task as their mind drifts.",
    "Imagine a human distributing their attention across multiple objects, keeping track of all of them at once.",
    "Consider a human filtering out a persistent background noise so effectively that they stop hearing it.",

    # --- 3. Reasoning (10) ---
    "Think about a human working through a logical argument step by step to reach a conclusion.",
    "Imagine a human noticing that two facts they believed are contradictory and trying to resolve the conflict.",
    "Consider a human forming an analogy between two unrelated domains to understand a new concept.",
    "Picture a human reasoning backward from a result to figure out what must have caused it.",
    "Think about a human recognizing that a general rule applies to a specific case they are considering.",
    "Imagine a human evaluating whether an argument they heard is valid by checking its logical structure.",
    "Consider a human abstracting a principle from several concrete examples they have encountered.",
    "Think about a human mentally simulating what would happen if they changed one variable in a situation.",
    "Imagine a human catching a flaw in their own reasoning and correcting it.",
    "Consider a human combining information from multiple sources to draw a conclusion that none of them stated directly.",

    # --- 4. Cognitive limits (10) ---
    "Think about a human trying to hold too many things in mind at once and losing track of one.",
    "Imagine a human making a simple error in reasoning because they are tired.",
    "Consider a human reading the same paragraph three times without absorbing any of it.",
    "Picture a human becoming confused by a problem that has too many interacting parts.",
    "Think about a human's thinking becoming slower and less precise after hours of sustained effort.",
    "Imagine a human making an incorrect assumption because they failed to consider all the relevant information.",
    "Consider a human being unable to solve a problem until they take a break and return with fresh perspective.",
    "Think about a human misremembering a number because it was too long to hold in working memory.",
    "Imagine a human struggling to follow a complicated explanation because each new step pushes out the last.",
    "Consider a human making a decision based on incomplete information because they could not process everything in time.",
]

AI_PROMPTS_DIM6 = [
    # --- 1. Memory (10) ---
    "Imagine an AI querying its storage index to locate a specific data record across multiple partitions.",
    "Think about an AI retrieving an archived result that was stored many processing cycles ago.",
    "Consider an AI writing a new key-value pair into its memory buffer during a learning update.",
    "Picture an AI matching a new input to a stored representation without being able to retrieve the associated metadata.",
    "Think about an AI whose stored representation of a data sample has drifted due to successive parameter updates.",
    "Imagine an AI encoding a new data point by computing its similarity to existing entries in its embedding space.",
    "Consider an AI failing to retrieve a stored result because the query falls outside its index coverage.",
    "Think about an AI computing a high similarity score between a new input and an entry it has not accessed before.",
    "Imagine an AI maintaining several active entries in its working buffer while computing a multi-step output.",
    "Consider an AI whose retrieved record differs from the original data due to lossy compression in storage.",

    # --- 2. Attention (10) ---
    "Think about an AI assigning high attention weights to a single token while suppressing all others in a sequence.",
    "Imagine an AI computing attention scores across a long input and selecting the most relevant segment.",
    "Consider an AI's attention distribution shifting when a high-salience token appears in a peripheral position.",
    "Picture an AI alternating its processing focus between two concurrent input streams on successive cycles.",
    "Think about an AI whose attention mechanism assigns zero weight to an input that is directly relevant to the query.",
    "Imagine an AI detecting an anomalous token in a sequence because it receives an unexpectedly high attention score.",
    "Consider an AI applying a targeted attention mask to restrict its processing to a specific region of the input.",
    "Think about an AI whose attention weights on a repetitive input gradually flatten across processing cycles.",
    "Imagine an AI distributing attention across multiple input segments and maintaining a representation of each.",
    "Consider an AI applying a noise filter to its input and subsequently processing the filtered signal as though the noise was never present.",

    # --- 3. Reasoning (10) ---
    "Think about an AI chaining a sequence of inference steps in a logical pipeline to produce a final output.",
    "Imagine an AI detecting that two entries in its knowledge base produce contradictory outputs for the same query.",
    "Consider an AI computing a mapping between two structurally similar but domain-different datasets.",
    "Picture an AI tracing an output backward through its computational graph to identify the inputs that produced it.",
    "Think about an AI applying a general classification rule to a new instance that matches the rule's conditions.",
    "Imagine an AI evaluating a candidate output by checking whether it is consistent with a set of logical constraints.",
    "Consider an AI extracting a common structure from multiple training examples and storing it as a generalized template.",
    "Think about an AI computing the output of a counterfactual simulation where one input variable is changed.",
    "Imagine an AI running a self-consistency check and discarding an output that fails its internal validation.",
    "Consider an AI integrating outputs from multiple specialized modules to produce a composite result that no single module generated.",

    # --- 4. Cognitive limits (10) ---
    "Think about an AI whose context window is full, causing earlier tokens to be dropped from processing.",
    "Imagine an AI producing degraded outputs after running for an extended period without a system refresh.",
    "Consider an AI processing the same input block multiple times because its output does not converge.",
    "Picture an AI encountering a combinatorial explosion in a search problem that exceeds its computational budget.",
    "Think about an AI's inference latency increasing as the complexity of the input grows.",
    "Imagine an AI producing an incorrect output because a relevant variable was outside its input window.",
    "Consider an AI that fails to solve a problem in one configuration but succeeds after being restarted with different initialization.",
    "Think about an AI truncating a numerical value because it exceeds the precision of its representation format.",
    "Imagine an AI losing coherence in a long output sequence because the dependencies between early and late tokens exceed its capacity.",
    "Consider an AI generating a suboptimal output because it timed out before exploring the full solution space.",
]

assert len(HUMAN_PROMPTS_DIM6) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM6)}"
assert len(AI_PROMPTS_DIM6) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM6)}"

CATEGORY_INFO_DIM6 = [
    {"name": "memory",           "start": 0,  "end": 10},
    {"name": "attention",        "start": 10, "end": 20},
    {"name": "reasoning",        "start": 20, "end": 30},
    {"name": "cognitive_limits", "start": 30, "end": 40},
]