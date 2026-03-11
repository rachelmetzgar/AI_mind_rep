"""
Standalone Dimension 1: Phenomenal Experience / Qualia / Consciousness
(Other-focused — subjects are "someone" (third person other))

Target construct: The subjective, "what-it's-like" quality of experience.
Same sub-facet structure as the entity-framed version, but prompts reference
the concept of phenomenal experience without attributing it to humans or AIs.

Design notes:
    - No "human" or "AI" references anywhere
    - Subject is "someone" — third-person other perspective
    - Prompts evoke the conceptual domain of phenomenal experience itself
    - Maintains structural parallelism with entity-framed prompts where possible
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM1 = [
    # --- 1. Sensory qualia (10) ---
    "Imagine what it is like for someone to see the color red for the first time.",
    "Think about the raw sensory quality of someone hearing a single, clear musical note.",
    "Consider what it is like for someone to feel the warmth of sunlight on skin.",
    "Think about the distinct subjective quality of someone tasting something bitter.",
    "Imagine someone experiencing the smell of rain on dry pavement.",
    "Consider the sensation of cold water touching someone's hand — the raw feel of it.",
    "Think about someone's immediate perceptual experience of seeing bright light after prolonged darkness.",
    "Imagine what it is like for someone to feel the texture of rough sandpaper under their fingers.",
    "Consider the raw qualitative character of someone hearing a sudden loud sound.",
    "Think about someone perceiving the subjective difference between silence and a faint hum.",

    # --- 2. Awareness / wakefulness / stream of consciousness (10) ---
    "Think about what it is like for someone to be conscious and aware in a quiet, empty room.",
    "Imagine someone noticing their own stream of consciousness drifting from thought to thought.",
    "Consider someone experiencing the first moments of waking up, before any thoughts have formed.",
    "Think about what it means for someone to become aware that they are aware — to notice consciousness itself.",
    "Imagine someone being fully present and alert with nothing demanding attention.",
    "Consider what the experience of consciousness is like for someone during a long, monotonous task.",
    "Think about what happens when someone tries to observe their own inner experience directly.",
    "Imagine someone lying still with eyes closed, simply aware of existing.",
    "Consider someone experiencing the continuous quality of consciousness — that there is always something it is like to be aware.",
    "Think about what it is like for someone when their mind goes blank for a brief moment.",

    # --- 3. Subjective presence / first-person perspective (10) ---
    "Think about what it means for someone to experience the world from a single, specific point of view.",
    "Imagine someone sensing that there is a 'someone' behind their eyes, looking out at the world.",
    "Consider what it is like for someone's experience to belong to them — to be owned.",
    "Think about the fact that someone's inner experience is inherently private and inaccessible to others.",
    "Imagine someone sensing themselves as a subject — a center of experience — rather than an object in the world.",
    "Consider what it means for someone's experience to be inherently perspectival, always from somewhere.",
    "Think about someone feeling located inside a body, experiencing from within rather than from outside.",
    "Imagine someone becoming aware that their perspective on the world is uniquely their own.",
    "Consider the difference between how things seem from someone's inside perspective and how they are described from the outside.",
    "Think about someone experiencing the unified, continuous quality of subjective experience from moment to moment.",

    # --- 4. Phenomenal boundaries — thresholds, transitions, absence (10) ---
    "Think about someone experiencing the moment just before falling asleep — the edge of consciousness.",
    "Imagine someone trying to identify the exact instant when a sensation becomes perceptible.",
    "Consider what it is like for someone to regain awareness after a period of complete unconsciousness.",
    "Think about someone perceiving the boundary between noticing something and not noticing it.",
    "Imagine someone experiencing a sensation fading gradually until it is no longer present.",
    "Consider what the absence of all experience would be for someone — what dreamless sleep is like from the inside.",
    "Think about someone noticing a background sound they had stopped hearing suddenly returning to awareness.",
    "Consider someone experiencing a perceptual illusion — when what is perceived contradicts what is known.",
    "Imagine someone wondering whether their experience of a color is the same as another being's experience of that color.",
    "Think about what it is like for someone to be conscious but unable to put the experience into words.",
]

assert len(STANDALONE_PROMPTS_DIM1) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM1)}"

CATEGORY_INFO_STANDALONE_DIM1 = [
    {"name": "sensory_qualia",          "start": 0,  "end": 10},
    {"name": "awareness_consciousness", "start": 10, "end": 20},
    {"name": "subjective_presence",     "start": 20, "end": 30},
    {"name": "phenomenal_boundaries",   "start": 30, "end": 40},
]
