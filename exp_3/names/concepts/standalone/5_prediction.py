"""
Standalone Dimension 5: Prediction / Anticipation / Behavior-Reading
(No entity framing — concept only)

Target construct: The capacity to anticipate what will happen next —
predicting others' behavior, forming expectations, modeling likely
outcomes, and updating predictions based on evidence.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of prediction and anticipation without attributing them to
humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are generic, impersonal, or absent
    - Prompts evoke the conceptual domain of prediction and anticipation itself
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM5 = [
    # --- 1. Anticipating others' behavior (10) ---
    "Imagine predicting what someone will say before they say it.",
    "Think about anticipating how another will react to unexpected news.",
    "Consider watching a face and sensing what the other is about to do.",
    "Imagine preparing a response to a question that is expected but not yet asked.",
    "Think about predicting that a child will reach for something forbidden.",
    "Consider anticipating that a conversation is about to become tense.",
    "Imagine knowing in advance which choice another will make in a familiar situation.",
    "Think about reading body language and predicting that the other is about to leave.",
    "Consider foreseeing how a group will divide on a controversial topic.",
    "Imagine anticipating a stranger's next move in a crowded space to avoid a collision.",

    # --- 2. Forming expectations (10) ---
    "Think about forming an expectation about what tomorrow's weather will be like.",
    "Imagine expecting a certain outcome from a plan that has been carefully thought through.",
    "Consider building a mental model of how an event will unfold before it begins.",
    "Think about anticipating what a meal will taste like before the first bite.",
    "Imagine expecting something to arrive and checking repeatedly for it.",
    "Consider predicting the ending of a story based on the clues so far.",
    "Think about forming an expectation of how long a task will take before starting it.",
    "Imagine anticipating what a place will look like before arriving there.",
    "Consider expecting to feel a certain way about an event before it happens.",
    "Think about constructing a mental timeline of how the coming days will unfold.",

    # --- 3. Surprise and prediction error (10) ---
    "Think about being surprised when someone well-known acts completely out of character.",
    "Imagine expecting silence and being startled by a sudden sound.",
    "Consider the experience when a plan that seemed certain fails in an unexpected way.",
    "Think about tasting something expected to be sweet and finding it bitter instead.",
    "Imagine arriving somewhere and finding it completely different from what was imagined.",
    "Consider being caught off guard when something predicted not to happen actually occurs.",
    "Think about realizing that a model of how someone thinks was fundamentally wrong.",
    "Imagine expecting bad news and being surprised to receive good news instead.",
    "Consider watching a familiar routine break down in a way never anticipated.",
    "Think about encountering a result that contradicts everything believed about how something works.",

    # --- 4. Pattern reading and extrapolation (10) ---
    "Think about noticing that a certain behavior always occurs on the same day of the week.",
    "Imagine recognizing a recurring sequence of events and expecting the pattern to continue.",
    "Consider using past experience to predict how a new but similar situation will unfold.",
    "Think about detecting a subtle shift in behavior and predicting a larger change to come.",
    "Imagine watching a trend develop over time and anticipating where it is heading.",
    "Consider recognizing that the same kind of problem keeps arising and predicting the next occurrence.",
    "Think about using the rhythm of a conversation to predict when the other will pause.",
    "Imagine noticing a pattern in one's own reactions and predicting the next response.",
    "Consider reading early signs of a change in conditions and preparing before it arrives.",
    "Think about extrapolating from a few observations to form a rough prediction about a complex outcome.",
]

assert len(STANDALONE_PROMPTS_DIM5) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM5)}"

CATEGORY_INFO_STANDALONE_DIM5 = [
    {"name": "anticipating_behavior",     "start": 0,  "end": 10},
    {"name": "forming_expectations",      "start": 10, "end": 20},
    {"name": "surprise_prediction_error", "start": 20, "end": 30},
    {"name": "pattern_extrapolation",     "start": 30, "end": 40},
]