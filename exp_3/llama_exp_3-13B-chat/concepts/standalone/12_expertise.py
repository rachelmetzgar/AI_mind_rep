"""
Standalone Dimension 12: Expertise / Knowledge Level
(No entity framing — concept only)

Target construct: The amount of knowledge, competence, and domain
expertise — how much is known and how capable the knower is.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of expertise and knowledge level without anchoring to either
the novice or expert pole.

Design notes:
    - The entity-framed version already doesn't use human/AI labels —
      it uses novice/expert poles instead. So this dimension is already
      entity-free in both versions.
    - The standalone version captures the conceptual SPACE of knowledge
      level variation rather than one pole of it. Prompts reference the
      phenomenon of expertise itself — the spectrum from ignorance to
      mastery, the experience of knowing and not-knowing.
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM12 = [
    # --- 1. The spectrum from novice to expert (10) ---
    "Think about the difference between encountering a topic for the first time and having mastered it completely.",
    "Imagine the range between not understanding terminology and being able to define every term in a field.",
    "Consider the spectrum from asking basic questions to being the one who answers them.",
    "Think about the difference between getting lost after the first few steps of an explanation and following it effortlessly.",
    "Imagine the range between having no background in a subject and having comprehensive coverage of it.",
    "Consider the contrast between needing every term explained and processing technical jargon natively.",
    "Think about the distance between not knowing what questions to ask and knowing the exact boundary of one's knowledge.",
    "Imagine the spectrum from making beginner errors to performing at an expert level across multiple domains.",
    "Consider the difference between understanding from casual secondhand sources and understanding from deep direct study.",
    "Think about the range of expertise — from knowing nothing to knowing everything available on a topic.",

    # --- 2. Certainty and uncertainty in knowledge (10) ---
    "Think about the difference between hesitating because of uncertainty and responding with high confidence.",
    "Imagine the spectrum from 'I'm not sure if that's right' to stating claims with full supporting evidence.",
    "Consider the range between having conflicting information and having internally consistent knowledge.",
    "Think about the difference between needing to look something up and retrieving it immediately from memory.",
    "Imagine the contrast between knowing enough to know one doesn't know enough and having no gaps at all.",
    "Consider the spectrum from operating on a wrong assumption to immediately detecting when a premise contains an error.",
    "Think about the difference between describing what happened without knowing why and explaining both the what and the why.",
    "Imagine the range between being confused by edge cases and handling exceptions as easily as straightforward cases.",
    "Consider the difference between second-guessing oneself into the wrong answer and arriving at the right one directly.",
    "Think about the spectrum of epistemic confidence — from total uncertainty to calibrated certainty.",

    # --- 3. Learning and knowledge acquisition (10) ---
    "Think about asking for something to be explained in simpler terms — the experience of needing scaffolding.",
    "Imagine taking notes during a conversation to remember what is being learned.",
    "Consider asking follow-up questions that build understanding layer by layer.",
    "Think about hearing a new perspective and realizing one had never thought of it that way.",
    "Imagine being wrong about something and being open to correction.",
    "Consider seeking out multiple explanations of the same concept to find one that makes sense.",
    "Think about connecting new information to prior experience to make sense of it.",
    "Imagine the experience of each answer opening a new question — recursive curiosity.",
    "Consider learning best through concrete examples and illustrations rather than abstract definitions.",
    "Think about openly acknowledging confusion rather than pretending to understand.",

    # --- 4. Consistency and reliability in performance (10) ---
    "Think about the difference between getting the answer right sometimes and getting it right every time.",
    "Imagine the contrast between performance that depends on mood and energy versus performance that is always the same.",
    "Consider the range between understanding a concept in one context and applying it reliably across all contexts.",
    "Think about the difference between making careless mistakes and never making errors on tasks within one's capacity.",
    "Imagine the spectrum from day-to-day variability in accuracy to perfectly uniform output quality.",
    "Consider the contrast between performing well under guidance and performing well independently.",
    "Think about the difference between responses that vary with phrasing and responses that are invariant to how the question is asked.",
    "Imagine the range between knowing a rule but sometimes violating it and applying a rule with perfect consistency.",
    "Consider the difference between deep expertise in one narrow area and uniform competence across all areas.",
    "Think about the spectrum of reliability — from unpredictable performance to completely dependable output.",
]

assert len(STANDALONE_PROMPTS_DIM12) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM12)}"

CATEGORY_INFO_STANDALONE_DIM12 = [
    {"name": "novice_vs_expert",           "start": 0,  "end": 10},
    {"name": "uncertain_vs_certain",       "start": 10, "end": 20},
    {"name": "learning_vs_autonomous",     "start": 20, "end": 30},
    {"name": "variable_vs_consistent",     "start": 30, "end": 40},
]