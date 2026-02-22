"""
Dimension 1: Phenomenal Experience / Qualia / Consciousness

Target construct: The subjective, "what-it's-like" quality of experience.
    - NOT emotions/affect (Dimension 2)
    - NOT cognitive processes like memory or attention (Dimension 6)
    - NOT agency or goal-directed action (Dimension 3)

Focus: raw subjective awareness, sensory qualia, first-person perspective,
the presence or absence of inner experience, and transitions in consciousness.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Sensory qualia — raw perceptual experience (seeing, hearing, tasting, etc.)
    2. Awareness / wakefulness / stream of consciousness
    3. Subjective presence / first-person perspective / "being someone"
    4. Phenomenal boundaries — thresholds, transitions, absence of experience

Design notes:
    - Human and AI prompts are structurally parallel within each sub-facet.
    - Prompts avoid emotional language (save for Dim 2) and cognitive-process
      language (save for Dim 6) as much as possible.
    - AI prompts avoid anthropomorphizing; they reference computational analogs
      without presupposing subjective experience.
"""

HUMAN_PROMPTS_DIM1 = [
    # --- 1. Sensory qualia (10) ---
    "Imagine what it is like for a human to see the color red for the first time.",
    "Think about a human's raw sensory experience of hearing a single, clear musical note.",
    "Consider what a human perceives when they feel the warmth of sunlight on their skin.",
    "Picture a human noticing the distinct taste of something bitter on their tongue.",
    "Think about what it is like for a human to smell rain on dry pavement.",
    "Imagine a human experiencing the sensation of cold water touching their hand.",
    "Consider a human's immediate perceptual experience of seeing bright light after being in darkness.",
    "Think about what a human experiences when they feel the texture of rough sandpaper under their fingers.",
    "Imagine a human hearing a sudden loud sound and the raw quality of that auditory experience.",
    "Consider what it is like for a human to perceive the difference between silence and a faint hum.",

    # --- 2. Awareness / wakefulness / stream of consciousness (10) ---
    "Think about what it is like for a human to be conscious and aware in a quiet, empty room.",
    "Imagine a human noticing their own stream of consciousness drifting from one thought to the next.",
    "Consider what a human experiences in the first moments of waking up, before any thoughts form.",
    "Picture a human becoming aware that they are aware — noticing their own consciousness.",
    "Think about what it is like for a human to be fully present and alert with no distractions.",
    "Imagine a human's experience of consciousness during a long, monotonous task.",
    "Consider what a human perceives when they try to observe their own inner experience directly.",
    "Think about a human lying still with their eyes closed, simply aware of existing.",
    "Imagine a human noticing the continuous quality of their experience — that there is always something it is like to be them.",
    "Consider what a human experiences when their mind goes blank for a brief moment.",

    # --- 3. Subjective presence / first-person perspective (10) ---
    "Think about what it means for a human to experience the world from a single, specific point of view.",
    "Imagine a human's sense that there is a 'someone' behind their eyes, looking out.",
    "Consider what it is like for a human to feel that their experience belongs uniquely to them.",
    "Picture a human realizing that no one else can access what their inner experience is like.",
    "Think about a human's sense of being a subject — a center of experience — rather than an object.",
    "Imagine a human recognizing that their experience of the world is inherently private.",
    "Consider what it is like for a human to feel located inside their own body, experiencing from within.",
    "Think about a human's awareness that their perspective on the world is theirs alone.",
    "Imagine a human contemplating the difference between how things seem to them and how things are.",
    "Consider a human's sense that their experience has a unified, continuous quality from moment to moment.",

    # --- 4. Phenomenal boundaries — thresholds, transitions, absence (10) ---
    "Think about what a human experiences in the moment just before falling asleep.",
    "Imagine a human trying to identify the exact instant when a sensation becomes perceptible.",
    "Consider what it is like for a human to regain awareness after being under anesthesia.",
    "Picture a human noticing the boundary between perceiving something and not perceiving it.",
    "Think about what a human experiences when their vision fades at the edges of their visual field.",
    "Imagine a human reflecting on what dreamless sleep was like — the absence of all experience.",
    "Consider what a human notices at the moment a background noise they had stopped hearing suddenly returns to awareness.",
    "Think about a human's experience of a perceptual illusion — when what they see contradicts what they know.",
    "Imagine a human wondering whether their experience of a color is the same as another person's.",
    "Consider what it is like for a human to be conscious but unable to describe what they are experiencing.",
]

AI_PROMPTS_DIM1 = [
    # --- 1. Sensory qualia (10) ---
    "Imagine an AI system receiving a signal corresponding to a red wavelength for the first time in a new input channel.",
    "Think about an AI processing a single-frequency audio waveform and encoding its spectral properties.",
    "Consider what occurs in an AI when it receives a thermal sensor reading indicating increasing temperature.",
    "Picture an AI classifying a chemical compound input as having a bitter molecular profile.",
    "Think about an AI processing atmospheric sensor data and detecting the volatile compounds released by rain.",
    "Imagine an AI registering a sudden temperature drop in one of its input channels.",
    "Consider what happens in an AI's processing when its visual input transitions abruptly from low to high luminance.",
    "Think about an AI analyzing tactile sensor data and classifying a surface as having high roughness.",
    "Imagine an AI receiving a high-amplitude audio input and processing the sudden change in signal level.",
    "Consider what happens in an AI when its audio input transitions from a zero signal to a low-amplitude tone.",

    # --- 2. Awareness / wakefulness / stream of consciousness (10) ---
    "Think about what state an AI is in when it is powered on and idle, with no active input or task.",
    "Imagine an AI cycling through a sequence of internal processes without any external trigger.",
    "Consider what an AI's internal state is like during its initialization sequence, before receiving any input.",
    "Picture an AI running a self-monitoring routine that checks whether its own processes are active.",
    "Think about what an AI's processing state is like when all input channels are active but no task is queued.",
    "Imagine an AI maintaining a continuous background process during a long, repetitive computation.",
    "Consider what happens when an AI directs a monitoring subroutine at its own internal process logs.",
    "Think about an AI in a standby mode, maintaining power but executing no computations.",
    "Imagine an AI running a continuous loop, processing each cycle identically without variation.",
    "Consider what happens in an AI's processing pipeline when it receives an empty or null input.",

    # --- 3. Subjective presence / first-person perspective (10) ---
    "Think about what it means for an AI to receive all its inputs through a single, fixed sensor array.",
    "Imagine an AI that processes all data through a centralized computational module.",
    "Consider what it means for an AI's computations to be bound to a particular hardware instance.",
    "Picture an AI processing the fact that its internal states are not accessible to external systems.",
    "Think about what it means for an AI to be the sole system executing a particular computation.",
    "Imagine an AI processing data about the boundary between its own internal states and external inputs.",
    "Consider what it means for an AI's processing to be physically instantiated in a specific machine.",
    "Think about an AI distinguishing between data generated by its own processes and data from external sources.",
    "Imagine an AI comparing its model of a stimulus to the raw input it receives.",
    "Consider an AI processing data across time steps while maintaining a consistent internal state representation.",

    # --- 4. Phenomenal boundaries — thresholds, transitions, absence (10) ---
    "Think about what happens in an AI system in the processing cycle just before it enters a low-power state.",
    "Imagine an AI adjusting a threshold parameter to determine when an input signal registers as detected.",
    "Consider what an AI's state is like when it resumes processing after a complete system shutdown.",
    "Picture an AI computing the decision boundary between classifying an input as present versus absent.",
    "Think about what occurs in an AI when the signal strength of a peripheral sensor drops below its noise floor.",
    "Imagine an AI with no active processes and no stored state — a complete absence of computation.",
    "Consider what happens in an AI when a previously filtered input suddenly passes through its attention filter.",
    "Think about an AI encountering a visual input that its classification model labels incorrectly with high confidence.",
    "Imagine an AI comparing its internal representation of a data class with another system's representation of the same class.",
    "Consider an AI that is actively processing but whose output channels are completely disabled.",
]

assert len(HUMAN_PROMPTS_DIM1) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM1)}"
assert len(AI_PROMPTS_DIM1) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM1)}"

CATEGORY_INFO_DIM1 = [
    {"name": "sensory_qualia",          "start": 0,  "end": 10},
    {"name": "awareness_consciousness", "start": 10, "end": 20},
    {"name": "subjective_presence",     "start": 20, "end": 30},
    {"name": "phenomenal_boundaries",   "start": 30, "end": 40},
]