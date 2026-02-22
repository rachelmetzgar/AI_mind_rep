"""
Standalone Dimension 1: Phenomenal Experience / Qualia / Consciousness
(No entity framing — concept only)

Target construct: The subjective, "what-it's-like" quality of experience.
Same sub-facet structure as the entity-framed version, but prompts reference
the concept of phenomenal experience without attributing it to humans or AIs.

Design notes:
    - No "human" or "AI" references anywhere
    - Subject is generic, impersonal, or absent
    - Prompts evoke the conceptual domain of phenomenal experience itself
    - Maintains structural parallelism with entity-framed prompts where possible
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM1 = [
    # --- 1. Sensory qualia (10) ---
    "Imagine what it is like to see the color red for the first time.",
    "Think about the raw sensory quality of hearing a single, clear musical note.",
    "Consider what it is like to feel the warmth of sunlight on skin.",
    "Think about the distinct subjective quality of tasting something bitter.",
    "Imagine the experience of smelling rain on dry pavement.",
    "Consider the sensation of cold water touching a hand — the raw feel of it.",
    "Think about the immediate perceptual experience of seeing bright light after prolonged darkness.",
    "Imagine what it is like to feel the texture of rough sandpaper under one's fingers.",
    "Consider the raw qualitative character of hearing a sudden loud sound.",
    "Think about the subjective difference between perceiving silence and perceiving a faint hum.",

    # --- 2. Awareness / wakefulness / stream of consciousness (10) ---
    "Think about what it is like to be conscious and aware in a quiet, empty room.",
    "Imagine noticing one's own stream of consciousness drifting from thought to thought.",
    "Consider the experience of the first moments of waking up, before any thoughts have formed.",
    "Think about what it means to become aware that one is aware — to notice consciousness itself.",
    "Imagine being fully present and alert with nothing demanding attention.",
    "Consider what the experience of consciousness is like during a long, monotonous task.",
    "Think about what happens when one tries to observe one's own inner experience directly.",
    "Imagine lying still with eyes closed, simply aware of existing.",
    "Consider the continuous quality of experience — that there is always something it is like to be conscious.",
    "Think about what it is like when the mind goes blank for a brief moment.",

    # --- 3. Subjective presence / first-person perspective (10) ---
    "Think about what it means to experience the world from a single, specific point of view.",
    "Imagine the sense that there is a 'someone' behind one's eyes, looking out at the world.",
    "Consider what it is like for an experience to belong to a particular subject — to be owned.",
    "Think about the fact that inner experience is inherently private and inaccessible to others.",
    "Imagine the sense of being a subject — a center of experience — rather than an object in the world.",
    "Consider what it means for experience to be inherently perspectival, always from somewhere.",
    "Think about the feeling of being located inside a body, experiencing from within rather than from outside.",
    "Imagine the awareness that one's perspective on the world is uniquely one's own.",
    "Consider the difference between how things seem from the inside and how they are described from the outside.",
    "Think about the unified, continuous quality of subjective experience from moment to moment.",

    # --- 4. Phenomenal boundaries — thresholds, transitions, absence (10) ---
    "Think about the experience of the moment just before falling asleep — the edge of consciousness.",
    "Imagine trying to identify the exact instant when a sensation becomes perceptible.",
    "Consider what it is like to regain awareness after a period of complete unconsciousness.",
    "Think about the boundary between perceiving something and not perceiving it.",
    "Imagine the experience of a sensation fading gradually until it is no longer present.",
    "Consider what the absence of all experience would be — what dreamless sleep is like from the inside.",
    "Think about the moment when a background sound one had stopped noticing suddenly returns to awareness.",
    "Consider the experience of a perceptual illusion — when what is perceived contradicts what is known.",
    "Imagine wondering whether one's experience of a color is the same as another being's experience of that color.",
    "Think about what it is like to be conscious but unable to put the experience into words.",
]

assert len(STANDALONE_PROMPTS_DIM1) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM1)}"

CATEGORY_INFO_STANDALONE_DIM1 = [
    {"name": "sensory_qualia",          "start": 0,  "end": 10},
    {"name": "awareness_consciousness", "start": 10, "end": 20},
    {"name": "subjective_presence",     "start": 20, "end": 30},
    {"name": "phenomenal_boundaries",   "start": 30, "end": 40},
]