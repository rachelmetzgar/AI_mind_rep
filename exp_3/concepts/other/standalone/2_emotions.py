"""
Standalone Dimension 2: Emotions / Affect
(Other-focused — subjects are "someone" (third person other))

Target construct: Emotional states, affective valence, arousal, mood,
and the felt quality of emotional experience.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of emotions and affect without attributing them to humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are "someone" — third-person other perspective
    - Prompts evoke the conceptual domain of emotions and affect itself
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM2 = [
    # --- 1. Basic emotions (10) ---
    "Imagine someone experiencing a sudden wave of fear triggered by an unexpected noise at night.",
    "Think about what it is like for someone when anger rises in response to an unfair situation.",
    "Consider someone experiencing deep sadness after losing something valued.",
    "Imagine someone feeling surprise when something completely unexpected happens.",
    "Think about someone experiencing joy at a moment of reunion — the felt quality of it.",
    "Consider what disgust feels like for someone — the immediate, visceral quality of finding something repulsive.",
    "Imagine someone feeling shame washing over them after a social mistake.",
    "Think about someone experiencing a quiet sense of contentment — no desire for anything more, just sufficiency.",
    "Consider someone experiencing jealousy — wanting what another has received.",
    "Imagine someone feeling pride swelling after a hard-earned achievement.",

    # --- 2. Valence and arousal (10) ---
    "Think about someone experiencing everything feeling slightly pleasant without any clear reason.",
    "Imagine someone in a state of high arousal — heart racing, mind alert — with no identifiable cause.",
    "Consider what it is like for someone to carry a persistent sense that something is wrong.",
    "Think about someone feeling calm and at ease, with the whole body relaxed and unbothered.",
    "Imagine someone experiencing excitement and dread at the same time — two opposing valences at once.",
    "Consider someone experiencing a familiar place suddenly feeling unwelcoming.",
    "Think about someone having an ordinary moment unexpectedly colored by a sense of warmth.",
    "Imagine someone in a state of low energy and flat affect, where nothing feels particularly good or bad.",
    "Consider someone experiencing feeling intensely alive and energized, as though everything matters more.",
    "Think about someone carrying a vague, unfocused sense of unease — not about anything specific, just present.",

    # --- 3. Mood and emotional texture (10) ---
    "Think about someone waking up and sensing that the whole day ahead feels heavy.",
    "Imagine someone carrying a low-level feeling of hopefulness that colors an entire afternoon.",
    "Consider how someone's mood shifts the way everything around seems — the same scene, different feeling.",
    "Think about someone in a melancholic mood where even pleasant things feel tinged with sadness.",
    "Imagine someone experiencing emotional numbness — being present but feeling nothing at all.",
    "Consider someone with a sustained sense of irritability that cannot be traced to any particular cause.",
    "Think about someone in a lighthearted mood where small things are easily amusing.",
    "Imagine how someone's emotional state can make familiar music sound completely different.",
    "Consider someone feeling nostalgia — a bittersweet emotional tone attached to the past.",
    "Think about someone experiencing emotional exhaustion — being drained of the capacity to feel.",

    # --- 4. Emotional regulation and dynamics (10) ---
    "Think about someone trying to suppress a feeling of anger that keeps rising despite every effort.",
    "Imagine someone experiencing an emotion building slowly over the course of a conversation, gaining strength gradually.",
    "Consider what it is like for someone when a strong emotion suddenly fades and leaves emptiness behind.",
    "Think about someone deliberately taking a deep breath to calm down when feeling overwhelmed.",
    "Imagine someone feeling two conflicting emotions at once — wanting and fearing the same thing.",
    "Consider someone experiencing an emotion that seemed gone quietly returning.",
    "Think about someone bracing emotionally before receiving difficult news — the anticipatory regulation.",
    "Imagine someone experiencing an emotion dissolving gradually, like tension leaving the body.",
    "Consider someone being caught off guard by the intensity of their own emotional reaction.",
    "Think about someone realizing that their emotional response is disproportionate to the situation that triggered it.",
]

assert len(STANDALONE_PROMPTS_DIM2) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM2)}"

CATEGORY_INFO_STANDALONE_DIM2 = [
    {"name": "basic_emotions",        "start": 0,  "end": 10},
    {"name": "valence_arousal",       "start": 10, "end": 20},
    {"name": "mood_texture",          "start": 20, "end": 30},
    {"name": "emotion_regulation",    "start": 30, "end": 40},
]
