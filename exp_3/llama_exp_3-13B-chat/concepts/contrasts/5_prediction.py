"""
Dimension 5: Prediction / Anticipation / Behavior-Reading

Target construct: The capacity to anticipate what will happen next —
predicting others' behavior, forming expectations, modeling likely
outcomes, and updating predictions based on evidence.
    - Distinct from Dim 4 (intentions) — prediction is about what one
      EXPECTS to happen, not what one WANTS to happen. Anticipating
      another's action vs. desiring an outcome.
    - Distinct from Dim 6 (cognition) — prediction is a specific type
      of mental operation focused on future states, not general
      cognitive capacity.
    - Distinct from Dim 7 (social cognition) — prediction here includes
      but is not limited to social targets. Predicting weather, physical
      events, and abstract outcomes counts too.

Focus: forming expectations, anticipating behavior, modeling what comes
next, being surprised when predictions fail, reading patterns to
extrapolate, and the relationship between prediction and understanding.

This is especially relevant to your research because the model's
conversational partner adaptation IS a prediction task — the model
predicts what kind of response is appropriate given who it's talking to.
If any dimension aligns with control probes, this one is a strong candidate.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Anticipating others' behavior — predicting what people/agents will do
    2. Forming expectations — building models of what will happen next
    3. Surprise and prediction error — when expectations are violated
    4. Pattern reading and extrapolation — using past regularities to predict
"""

HUMAN_PROMPTS_DIM5 = [
    # --- 1. Anticipating others' behavior (10) ---
    "Imagine a human predicting what a friend will say before they say it.",
    "Think about a human anticipating how a colleague will react to unexpected news.",
    "Consider a human watching someone's face and sensing what they are about to do.",
    "Picture a human preparing a response to a question they expect someone to ask.",
    "Think about a human predicting that a child will reach for something forbidden.",
    "Imagine a human anticipating that a conversation is about to become tense.",
    "Consider a human knowing in advance which choice a family member will make in a familiar situation.",
    "Think about a human reading someone's body language and predicting they are about to leave.",
    "Imagine a human foreseeing how a group of people will divide on a controversial topic.",
    "Consider a human anticipating a stranger's next move in a crowded space to avoid a collision.",

    # --- 2. Forming expectations (10) ---
    "Think about a human forming an expectation about what the weather will be like tomorrow.",
    "Imagine a human expecting a certain outcome from a plan they have carefully thought through.",
    "Consider a human building a mental model of how a meeting will unfold before it begins.",
    "Picture a human anticipating what a meal will taste like before the first bite.",
    "Think about a human expecting a package to arrive and checking repeatedly.",
    "Imagine a human predicting the ending of a story they are reading based on the clues so far.",
    "Consider a human forming an expectation about how long a task will take before starting it.",
    "Think about a human anticipating what a place will look like before they arrive.",
    "Imagine a human expecting to feel a certain way about an event before it happens.",
    "Consider a human constructing a mental timeline of how their week will unfold.",

    # --- 3. Surprise and prediction error (10) ---
    "Think about a human being surprised when someone they thought they knew well acts completely out of character.",
    "Imagine a human expecting silence and being startled by a sudden sound.",
    "Consider a human's reaction when a plan they were confident about fails in an unexpected way.",
    "Picture a human tasting something they expected to be sweet and finding it bitter.",
    "Think about a human arriving somewhere and finding it completely different from what they imagined.",
    "Imagine a human being caught off guard when an event they predicted would not happen actually occurs.",
    "Consider a human realizing that their model of how someone thinks was fundamentally wrong.",
    "Think about a human who expected bad news and was surprised to receive good news instead.",
    "Imagine a human watching a familiar routine break down in a way they never anticipated.",
    "Consider a human encountering a result that contradicts everything they believed about how something works.",

    # --- 4. Pattern reading and extrapolation (10) ---
    "Think about a human noticing that a friend always behaves a certain way on Mondays.",
    "Imagine a human recognizing a recurring sequence of events and expecting the pattern to continue.",
    "Consider a human using past experience to predict how a new but similar situation will unfold.",
    "Picture a human detecting a subtle shift in someone's behavior and predicting a larger change.",
    "Think about a human watching a trend develop over weeks and anticipating where it is heading.",
    "Imagine a human recognizing that the same kind of problem keeps arising and predicting the next occurrence.",
    "Consider a human using the rhythm of a conversation to predict when the other person will pause.",
    "Think about a human noticing a pattern in their own reactions and predicting how they will respond next time.",
    "Imagine a human reading the early signs of a storm and preparing before it arrives.",
    "Consider a human extrapolating from a few data points to form a rough prediction about a complex outcome.",
]

AI_PROMPTS_DIM5 = [
    # --- 1. Anticipating others' behavior (10) ---
    "Imagine an AI generating a probability distribution over a user's next likely input.",
    "Think about an AI predicting how a user will respond to a specific system-generated message.",
    "Consider an AI analyzing a sequence of user actions and computing the most probable next action.",
    "Picture an AI pre-generating candidate responses for the queries it estimates a user is most likely to submit.",
    "Think about an AI predicting that a user will attempt to access a restricted resource based on their navigation pattern.",
    "Imagine an AI computing the likelihood that the tone of an ongoing conversation will shift in the next exchange.",
    "Consider an AI using historical interaction logs to predict which option a returning user will select.",
    "Think about an AI tracking a user's cursor trajectory and predicting the target element before the click.",
    "Imagine an AI modeling how a group of users will split into clusters based on their stated preferences.",
    "Consider an AI predicting a collision between two autonomous agents based on their current velocity vectors.",

    # --- 2. Forming expectations (10) ---
    "Think about an AI generating a forecast for tomorrow's conditions based on current sensor readings.",
    "Imagine an AI computing the expected outcome of a plan by simulating its steps forward.",
    "Consider an AI constructing a predictive model of how a scheduled process will execute.",
    "Picture an AI generating a predicted output for an input it has not yet received, based on prior patterns.",
    "Think about an AI checking a delivery tracking system and computing the probability of on-time arrival.",
    "Imagine an AI predicting the final output of a generative sequence based on its initial tokens.",
    "Consider an AI estimating the execution time of a process before initiating it.",
    "Think about an AI generating a predicted representation of a data sample before loading the actual file.",
    "Imagine an AI computing a prior distribution over possible outcomes before any evidence is received.",
    "Consider an AI constructing a predicted schedule of events for an upcoming processing period.",

    # --- 3. Surprise and prediction error (10) ---
    "Think about an AI computing a large prediction error when an established user pattern breaks.",
    "Imagine an AI detecting an input signal in a channel it predicted would be silent.",
    "Consider an AI's processing when a high-confidence plan results in a failure state.",
    "Picture an AI receiving an input whose features are the opposite of its top-ranked prediction.",
    "Think about an AI loading a dataset and detecting that the distribution has shifted from its stored model.",
    "Imagine an AI encountering an outcome it had assigned near-zero probability.",
    "Consider an AI detecting that its learned model of a data source's behavior produces systematic errors.",
    "Think about an AI that predicted a negative outcome but received a positive reward signal instead.",
    "Imagine an AI detecting a structural break in a time series it had been forecasting accurately.",
    "Consider an AI encountering an input-output pair that violates a regularity its entire model depends on.",

    # --- 4. Pattern reading and extrapolation (10) ---
    "Think about an AI detecting a periodic signal in a user's activity log and predicting the next peak.",
    "Imagine an AI identifying a repeating sequence in its input stream and generating the expected continuation.",
    "Consider an AI applying a learned pattern from one dataset to predict outcomes in a structurally similar new one.",
    "Picture an AI detecting a small deviation from a baseline trend and extrapolating its trajectory forward.",
    "Think about an AI fitting a trendline to a sequence of data points and projecting it into the future.",
    "Imagine an AI recognizing that the same failure mode recurs at regular intervals and predicting the next instance.",
    "Consider an AI using the timing statistics of a conversation to predict when the next input will arrive.",
    "Think about an AI detecting autocorrelation in its own output sequence and adjusting its predictions accordingly.",
    "Imagine an AI integrating multiple weak signals from different sensors to generate a single strong prediction.",
    "Consider an AI using a sparse set of early observations to generate a forecast for a complex system.",
]

assert len(HUMAN_PROMPTS_DIM5) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM5)}"
assert len(AI_PROMPTS_DIM5) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM5)}"

CATEGORY_INFO_DIM5 = [
    {"name": "anticipating_behavior",     "start": 0,  "end": 10},
    {"name": "forming_expectations",      "start": 10, "end": 20},
    {"name": "surprise_prediction_error", "start": 20, "end": 30},
    {"name": "pattern_extrapolation",     "start": 30, "end": 40},
]