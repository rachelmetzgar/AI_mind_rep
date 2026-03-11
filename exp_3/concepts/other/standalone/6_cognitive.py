"""
Standalone Dimension 6: Cognitive Processes — Memory, Attention, Reasoning
(Other-focused — third-person someone)

Target construct: The basic machinery of thinking — how information is
stored, retrieved, selected, combined, and transformed.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of cognitive processes without attributing them to humans or AIs.
Subjects reference "someone" rather than being generic/impersonal.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects reference "someone" (third-person other-focused)
    - Prompts evoke the conceptual domain of cognition itself
    - This dimension is relatively low in implicit entity-type coding —
      memory, attention, and reasoning are commonly attributed to both
      humans and AI systems in everyday discourse
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM6 = [
    # --- 1. Memory (10) ---
    "Imagine someone trying to recall where something was left, mentally retracing steps.",
    "Think about someone suddenly remembering a detail from long ago that seemed forgotten.",
    "Consider someone rehearsing a list of facts to keep them in memory.",
    "Imagine someone recognizing something familiar but being unable to place where it was encountered before.",
    "Think about how someone's memory of an event can gradually change over time without being noticed.",
    "Consider someone encoding new information by connecting it to something already known.",
    "Imagine someone forgetting an important detail at the worst possible moment.",
    "Think about someone experiencing a strong sense of familiarity in an entirely new situation.",
    "Consider someone holding several pieces of information in mind at once while working through a problem.",
    "Imagine someone discovering that a memory of a conversation differs from what actually happened.",

    # --- 2. Attention (10) ---
    "Think about someone focusing intently on a single task while ignoring everything else.",
    "Imagine someone trying to follow one voice among many competing sources of information.",
    "Consider someone noticing something at the edge of awareness that suddenly pulls attention away.",
    "Think about someone shifting attention back and forth between two simultaneous demands.",
    "Imagine someone becoming so absorbed in a task that external signals go completely unnoticed.",
    "Consider someone scanning a dense stream of information and catching on something unexpected.",
    "Think about someone deliberately directing attention to a specific detail in a complex scene.",
    "Imagine someone struggling to maintain focus on a monotonous task as concentration drifts.",
    "Consider someone distributing attention across multiple things, keeping track of all of them at once.",
    "Think about someone filtering out a persistent background signal so effectively that it stops being noticed.",

    # --- 3. Reasoning (10) ---
    "Think about someone working through a logical argument step by step to reach a conclusion.",
    "Imagine someone noticing that two accepted facts are contradictory and trying to resolve the conflict.",
    "Consider someone forming an analogy between two unrelated domains to understand something new.",
    "Think about someone reasoning backward from a result to figure out what must have caused it.",
    "Imagine someone recognizing that a general rule applies to a specific case under consideration.",
    "Consider someone evaluating whether an argument is valid by checking its logical structure.",
    "Think about someone abstracting a principle from several concrete examples.",
    "Imagine someone mentally simulating what would happen if one variable in a situation were changed.",
    "Consider someone catching a flaw in their own reasoning and correcting course.",
    "Think about someone combining information from multiple sources to draw a conclusion none of them stated directly.",

    # --- 4. Cognitive limits (10) ---
    "Think about someone trying to hold too many things in mind at once and losing track of one.",
    "Imagine someone making a simple error in reasoning because of exhaustion.",
    "Consider someone processing the same information repeatedly without absorbing any of it.",
    "Think about someone becoming confused by a problem that has too many interacting parts.",
    "Imagine someone's thinking becoming slower and less precise after sustained effort.",
    "Consider someone making an incorrect assumption because not all relevant information was taken into account.",
    "Think about someone being unable to solve a problem until stepping away and returning with fresh perspective.",
    "Imagine someone misremembering a detail because it exceeded the capacity of working memory.",
    "Consider someone struggling to follow a complex explanation because each new step pushes out the last.",
    "Think about someone making a decision based on incomplete information because everything could not be processed in time.",
]

assert len(STANDALONE_PROMPTS_DIM6) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM6)}"

CATEGORY_INFO_STANDALONE_DIM6 = [
    {"name": "memory",           "start": 0,  "end": 10},
    {"name": "attention",        "start": 10, "end": 20},
    {"name": "reasoning",        "start": 20, "end": 30},
    {"name": "cognitive_limits", "start": 30, "end": 40},
]
