"""
Standalone Dimension 2: Emotions / Affect
(No entity framing — concept only)

Target construct: Emotional states, affective valence, arousal, mood,
and the felt quality of emotional experience.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of emotions and affect without attributing them to humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are generic, impersonal, or absent
    - Prompts evoke the conceptual domain of emotions and affect itself
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM2 = [
    # --- 1. Basic emotions (10) ---
    "Imagine experiencing a sudden wave of fear triggered by an unexpected noise at night.",
    "Think about what it is like when anger rises in response to an unfair situation.",
    "Consider the experience of deep sadness after losing something valued.",
    "Imagine the feeling of surprise when something completely unexpected happens.",
    "Think about the experience of joy at a moment of reunion — the felt quality of it.",
    "Consider what disgust feels like — the immediate, visceral quality of finding something repulsive.",
    "Imagine the feeling of shame washing over someone after a social mistake.",
    "Think about a quiet sense of contentment — no desire for anything more, just sufficiency.",
    "Consider the experience of jealousy — wanting what someone else has received.",
    "Imagine the feeling of pride swelling after a hard-earned achievement.",

    # --- 2. Valence and arousal (10) ---
    "Think about the experience of everything feeling slightly pleasant without any clear reason.",
    "Imagine a state of high arousal — heart racing, mind alert — with no identifiable cause.",
    "Consider what it is like to carry a persistent sense that something is wrong.",
    "Think about the feeling of being calm and at ease, with the whole body relaxed and unbothered.",
    "Imagine experiencing excitement and dread at the same time — two opposing valences at once.",
    "Consider the experience of a familiar place suddenly feeling unwelcoming.",
    "Think about an ordinary moment being unexpectedly colored by a sense of warmth.",
    "Imagine a state of low energy and flat affect, where nothing feels particularly good or bad.",
    "Consider the experience of feeling intensely alive and energized, as though everything matters more.",
    "Think about a vague, unfocused sense of unease — not about anything specific, just present.",

    # --- 3. Mood and emotional texture (10) ---
    "Think about waking up and sensing that the whole day ahead feels heavy.",
    "Imagine carrying a low-level feeling of hopefulness that colors an entire afternoon.",
    "Consider how mood shifts the way everything around seems — the same scene, different feeling.",
    "Think about a melancholic mood where even pleasant things feel tinged with sadness.",
    "Imagine the experience of emotional numbness — being present but feeling nothing at all.",
    "Consider a sustained sense of irritability that cannot be traced to any particular cause.",
    "Think about a lighthearted mood where small things are easily amusing.",
    "Imagine how an emotional state can make familiar music sound completely different.",
    "Consider the feeling of nostalgia — a bittersweet emotional tone attached to the past.",
    "Think about the experience of emotional exhaustion — being drained of the capacity to feel.",

    # --- 4. Emotional regulation and dynamics (10) ---
    "Think about trying to suppress a feeling of anger that keeps rising despite every effort.",
    "Imagine an emotion building slowly over the course of a conversation, gaining strength gradually.",
    "Consider what it is like when a strong emotion suddenly fades and leaves emptiness behind.",
    "Think about deliberately taking a deep breath to calm down when feeling overwhelmed.",
    "Imagine the experience of feeling two conflicting emotions at once — wanting and fearing the same thing.",
    "Consider the experience of an emotion that seemed gone quietly returning.",
    "Think about bracing emotionally before receiving difficult news — the anticipatory regulation.",
    "Imagine an emotion dissolving gradually, like tension leaving the body.",
    "Consider being caught off guard by the intensity of one's own emotional reaction.",
    "Think about realizing that an emotional response is disproportionate to the situation that triggered it.",
]

assert len(STANDALONE_PROMPTS_DIM2) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM2)}"

CATEGORY_INFO_STANDALONE_DIM2 = [
    {"name": "basic_emotions",        "start": 0,  "end": 10},
    {"name": "valence_arousal",       "start": 10, "end": 20},
    {"name": "mood_texture",          "start": 20, "end": 30},
    {"name": "emotion_regulation",    "start": 30, "end": 40},
]