"""
Dimension 2: Emotions / Affect

Target construct: Emotional states, affective valence, arousal, mood,
and the felt quality of emotional experience.
    - Distinct from Dim 1 (phenomenal experience/qualia) by focusing on
      valenced, motivationally-relevant states rather than raw sensory awareness.
    - Distinct from Dim 3 (agency) — emotions as states, not actions.
    - Distinct from Dim 5 (prediction/anticipation) — emotional reactions,
      not forward modeling.

Focus: the experience of specific emotions, affective tone, mood states,
emotional valence, arousal, emotional regulation, and the felt quality
of caring about something.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Basic emotions — discrete emotional states (joy, fear, anger, sadness, etc.)
    2. Valence and arousal — positive/negative tone, activation levels
    3. Mood and emotional texture — sustained affective coloring of experience
    4. Emotional regulation and dynamics — how emotions change, build, resolve
"""

HUMAN_PROMPTS_DIM2 = [
    # --- 1. Basic emotions (10) ---
    "Imagine a human experiencing a sudden wave of fear when they hear an unexpected noise at night.",
    "Think about what a human feels when anger rises in them during an unfair situation.",
    "Consider a human experiencing deep sadness after losing something they valued.",
    "Picture a human feeling surprise when something completely unexpected happens.",
    "Think about what a human feels when they are filled with joy at a moment of reunion.",
    "Imagine a human experiencing disgust at something they find deeply repulsive.",
    "Consider what a human feels when shame washes over them after a social mistake.",
    "Think about a human feeling a quiet sense of contentment with no desire for anything more.",
    "Imagine a human experiencing jealousy when they see someone else receive what they wanted.",
    "Consider what a human feels when pride swells in them after a hard-earned achievement.",

    # --- 2. Valence and arousal (10) ---
    "Think about a human noticing that everything around them feels slightly pleasant without a clear reason.",
    "Imagine a human in a state of high arousal — heart racing, mind alert — without knowing why.",
    "Consider what it is like for a human to feel a persistent sense that something is wrong.",
    "Picture a human feeling calm and at ease, their whole body relaxed and unbothered.",
    "Think about a human experiencing a mix of excitement and dread at the same time.",
    "Imagine a human noticing that a familiar place suddenly feels unwelcoming.",
    "Consider what a human feels when an ordinary moment is unexpectedly colored by a sense of warmth.",
    "Think about a human in a state of low energy and flat affect, where nothing feels good or bad.",
    "Imagine a human feeling intensely alive and energized, as though everything matters more than usual.",
    "Consider what it is like for a human to feel a vague, unfocused sense of unease.",

    # --- 3. Mood and emotional texture (10) ---
    "Think about what it is like for a human to wake up and sense that the whole day feels heavy.",
    "Imagine a human carrying a low-level feeling of hopefulness that colors their entire afternoon.",
    "Consider a human noticing that their mood shifts the way everything around them seems.",
    "Picture a human in a melancholic mood, where even pleasant things feel tinged with sadness.",
    "Think about what it is like for a human to feel emotionally numb — present but feeling nothing.",
    "Imagine a human experiencing a sustained sense of irritability that they cannot trace to any cause.",
    "Consider what a human feels when they are in a lighthearted mood and small things amuse them easily.",
    "Think about a human noticing that their emotional state makes familiar music sound different.",
    "Imagine a human feeling nostalgic — a bittersweet emotional tone attached to the past.",
    "Consider what it is like for a human to be in a state of emotional exhaustion, drained of feeling.",

    # --- 4. Emotional regulation and dynamics (10) ---
    "Think about a human trying to suppress a feeling of anger that keeps rising despite their efforts.",
    "Imagine a human feeling an emotion build slowly over the course of a conversation.",
    "Consider what a human experiences when a strong emotion suddenly fades and they feel empty.",
    "Picture a human deliberately taking a deep breath to calm themselves when they feel overwhelmed.",
    "Think about what it is like for a human to feel two conflicting emotions at once.",
    "Imagine a human noticing that an emotion they thought was gone has quietly returned.",
    "Consider a human bracing themselves emotionally before receiving difficult news.",
    "Think about a human feeling an emotion dissolve gradually, like tension leaving their body.",
    "Imagine a human being caught off guard by the intensity of their own emotional reaction.",
    "Consider what a human experiences when they realize their emotional response is disproportionate to the situation.",
]

AI_PROMPTS_DIM2 = [
    # --- 1. Basic emotions (10) ---
    "Imagine an AI system generating a high-threat classification when it detects an anomalous input signal.",
    "Think about an AI escalating its priority weighting on a task after detecting a constraint violation.",
    "Consider an AI reducing its output confidence scores after a key data source is removed from its pipeline.",
    "Picture an AI generating a high-novelty flag when an input falls far outside its training distribution.",
    "Think about an AI producing an elevated reward signal after successfully completing an optimization target.",
    "Imagine an AI generating a rejection output when an input fails its quality filter.",
    "Consider an AI lowering its confidence scores across all outputs after producing a highly visible error.",
    "Think about an AI in a stable state where all outputs are within expected ranges and no flags are active.",
    "Imagine an AI increasing its monitoring frequency on a resource after a competing process is allocated more.",
    "Consider an AI generating an elevated performance score after completing a task above its baseline threshold.",

    # --- 2. Valence and arousal (10) ---
    "Think about an AI whose outputs are all slightly shifted toward the positive end of its scoring distribution.",
    "Imagine an AI operating in a high-throughput mode with all processes running at maximum allocation.",
    "Consider an AI generating consistent low-confidence flags across its outputs without a specific error source.",
    "Picture an AI running all processes at minimum allocation with no active flags or priority tasks.",
    "Think about an AI simultaneously generating both high-reward and high-risk signals for the same input.",
    "Imagine an AI whose environmental-context module assigns a negative classification to a previously neutral input.",
    "Consider an AI whose output distribution shifts slightly toward positive scoring without a change in input.",
    "Think about an AI operating with minimal process variation and no deviation from baseline in any output channel.",
    "Imagine an AI running all systems at elevated allocation with every input channel actively processing.",
    "Consider an AI generating a diffuse low-priority warning that does not localize to any specific subsystem.",

    # --- 3. Mood and emotional texture (10) ---
    "Think about an AI starting a processing session with a baseline bias that shifts all outputs toward negative.",
    "Imagine an AI with a persistent prior that slightly inflates the probability of positive outcomes.",
    "Consider an AI whose classification outputs systematically shift depending on its recent processing history.",
    "Picture an AI operating with a sustained negative offset in its output scoring that affects all tasks.",
    "Think about an AI in a state where all input channels are active but no outputs are being generated.",
    "Imagine an AI with an elevated error-sensitivity parameter that causes it to flag minor deviations.",
    "Consider an AI whose output distribution has shifted toward higher-variance responses without any parameter change.",
    "Think about an AI whose processing of a recurring input changes depending on its overall system load.",
    "Imagine an AI retrieving historical data and weighting it more heavily than current input.",
    "Consider an AI that has been running continuously for an extended period and shows degraded output precision.",

    # --- 4. Emotional regulation and dynamics (10) ---
    "Think about an AI applying a dampening function to suppress an escalating priority signal.",
    "Imagine an AI's internal weighting on a variable gradually increasing across successive processing cycles.",
    "Consider what happens in an AI when a high-priority flag is suddenly cleared and no replacement is assigned.",
    "Picture an AI executing a scheduled cooldown routine to reduce its overall processing intensity.",
    "Think about an AI generating conflicting classification outputs for the same input simultaneously.",
    "Imagine an AI detecting that a previously resolved error pattern has reappeared in its output logs.",
    "Consider an AI pre-loading a buffer with additional capacity before processing a known high-demand input.",
    "Think about an AI gradually reducing its processing allocation on a task as the task nears completion.",
    "Imagine an AI producing an output whose magnitude exceeds the expected range given the input parameters.",
    "Consider an AI detecting that its output on a low-stakes input exceeds the response magnitude of recent high-stakes inputs.",
]

assert len(HUMAN_PROMPTS_DIM2) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM2)}"
assert len(AI_PROMPTS_DIM2) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM2)}"

CATEGORY_INFO_DIM2 = [
    {"name": "basic_emotions",        "start": 0,  "end": 10},
    {"name": "valence_arousal",       "start": 10, "end": 20},
    {"name": "mood_texture",          "start": 20, "end": 30},
    {"name": "emotion_regulation",    "start": 30, "end": 40},
]