"""
Standalone Dimension 5: Prediction / Anticipation / Behavior-Reading
(Other-focused — subjects are "someone" (third person other))

Target construct: The capacity to anticipate what will happen next —
predicting others' behavior, forming expectations, modeling likely
outcomes, and updating predictions based on evidence.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of prediction and anticipation without attributing them to
humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are "someone" — third-person other perspective
    - Prompts evoke the conceptual domain of prediction and anticipation itself
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM5 = [
    # --- 1. Anticipating others' behavior (10) ---
    "Imagine someone predicting what another will say before they say it.",
    "Think about someone anticipating how another will react to unexpected news.",
    "Consider someone watching a face and sensing what the other is about to do.",
    "Imagine someone preparing a response to a question that is expected but not yet asked.",
    "Think about someone predicting that a child will reach for something forbidden.",
    "Consider someone anticipating that a conversation is about to become tense.",
    "Imagine someone knowing in advance which choice another will make in a familiar situation.",
    "Think about someone reading body language and predicting that the other is about to leave.",
    "Consider someone foreseeing how a group will divide on a controversial topic.",
    "Imagine someone anticipating a stranger's next move in a crowded space to avoid a collision.",

    # --- 2. Forming expectations (10) ---
    "Think about someone forming an expectation about what tomorrow's weather will be like.",
    "Imagine someone expecting a certain outcome from a plan that has been carefully thought through.",
    "Consider someone building a mental model of how an event will unfold before it begins.",
    "Think about someone anticipating what a meal will taste like before the first bite.",
    "Imagine someone expecting something to arrive and checking repeatedly for it.",
    "Consider someone predicting the ending of a story based on the clues so far.",
    "Think about someone forming an expectation of how long a task will take before starting it.",
    "Imagine someone anticipating what a place will look like before arriving there.",
    "Consider someone expecting to feel a certain way about an event before it happens.",
    "Think about someone constructing a mental timeline of how the coming days will unfold.",

    # --- 3. Surprise and prediction error (10) ---
    "Think about someone being surprised when a person they know well acts completely out of character.",
    "Imagine someone expecting silence and being startled by a sudden sound.",
    "Consider someone experiencing a plan that seemed certain failing in an unexpected way.",
    "Think about someone tasting something expected to be sweet and finding it bitter instead.",
    "Imagine someone arriving somewhere and finding it completely different from what was imagined.",
    "Consider someone being caught off guard when something predicted not to happen actually occurs.",
    "Think about someone realizing that their model of how another thinks was fundamentally wrong.",
    "Imagine someone expecting bad news and being surprised to receive good news instead.",
    "Consider someone watching a familiar routine break down in a way never anticipated.",
    "Think about someone encountering a result that contradicts everything believed about how something works.",

    # --- 4. Pattern reading and extrapolation (10) ---
    "Think about someone noticing that a certain behavior always occurs on the same day of the week.",
    "Imagine someone recognizing a recurring sequence of events and expecting the pattern to continue.",
    "Consider someone using past experience to predict how a new but similar situation will unfold.",
    "Think about someone detecting a subtle shift in behavior and predicting a larger change to come.",
    "Imagine someone watching a trend develop over time and anticipating where it is heading.",
    "Consider someone recognizing that the same kind of problem keeps arising and predicting the next occurrence.",
    "Think about someone using the rhythm of a conversation to predict when the other will pause.",
    "Imagine someone noticing a pattern in their own reactions and predicting the next response.",
    "Consider someone reading early signs of a change in conditions and preparing before it arrives.",
    "Think about someone extrapolating from a few observations to form a rough prediction about a complex outcome.",
]

assert len(STANDALONE_PROMPTS_DIM5) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM5)}"

CATEGORY_INFO_STANDALONE_DIM5 = [
    {"name": "anticipating_behavior",     "start": 0,  "end": 10},
    {"name": "forming_expectations",      "start": 10, "end": 20},
    {"name": "surprise_prediction_error", "start": 20, "end": 30},
    {"name": "pattern_extrapolation",     "start": 30, "end": 40},
]
