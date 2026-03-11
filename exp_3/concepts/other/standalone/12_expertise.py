"""
Standalone Dimension 12: Expertise / Knowledge Level
Other-focused (third-person someone)

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
    "Think about someone encountering a topic for the first time versus someone who has mastered it completely.",
    "Imagine someone who does not understand terminology versus someone who can define every term in a field.",
    "Consider the spectrum from someone asking basic questions to someone being the one who answers them.",
    "Think about someone getting lost after the first few steps of an explanation versus someone following it effortlessly.",
    "Imagine someone with no background in a subject versus someone with comprehensive coverage of it.",
    "Consider someone who needs every term explained versus someone who processes technical jargon natively.",
    "Think about someone who does not know what questions to ask versus someone who knows the exact boundary of their knowledge.",
    "Imagine someone making beginner errors versus someone performing at an expert level across multiple domains.",
    "Consider someone whose understanding comes from casual secondhand sources versus someone whose understanding comes from deep direct study.",
    "Think about someone at one end of the expertise range — knowing nothing — versus someone at the other — knowing everything available on a topic.",

    # --- 2. Certainty and uncertainty in knowledge (10) ---
    "Think about someone hesitating because of uncertainty versus someone responding with high confidence.",
    "Imagine someone saying 'I'm not sure if that's right' versus someone stating claims with full supporting evidence.",
    "Consider someone with conflicting information versus someone with internally consistent knowledge.",
    "Think about someone needing to look something up versus someone retrieving it immediately from memory.",
    "Imagine someone who knows enough to know they don't know enough versus someone who has no gaps at all.",
    "Consider someone operating on a wrong assumption versus someone who immediately detects when a premise contains an error.",
    "Think about someone describing what happened without knowing why versus someone explaining both the what and the why.",
    "Imagine someone confused by edge cases versus someone handling exceptions as easily as straightforward cases.",
    "Consider someone second-guessing themselves into the wrong answer versus someone arriving at the right one directly.",
    "Think about someone at one end of the spectrum of epistemic confidence — total uncertainty — versus someone at the other — calibrated certainty.",

    # --- 3. Learning and knowledge acquisition (10) ---
    "Think about someone asking for something to be explained in simpler terms — the experience of needing scaffolding.",
    "Imagine someone taking notes during a conversation to remember what they are learning.",
    "Consider someone asking follow-up questions that build their understanding layer by layer.",
    "Think about someone hearing a new perspective and realizing they had never thought of it that way.",
    "Imagine someone being wrong about something and being open to correction.",
    "Consider someone seeking out multiple explanations of the same concept to find one that makes sense to them.",
    "Think about someone connecting new information to their prior experience to make sense of it.",
    "Imagine someone whose each answer opens a new question — recursive curiosity.",
    "Consider someone learning best through concrete examples and illustrations rather than abstract definitions.",
    "Think about someone openly acknowledging their confusion rather than pretending to understand.",

    # --- 4. Consistency and reliability in performance (10) ---
    "Think about someone getting the answer right sometimes versus someone getting it right every time.",
    "Imagine someone whose performance depends on mood and energy versus someone whose performance is always the same.",
    "Consider someone who understands a concept in one context versus someone who applies it reliably across all contexts.",
    "Think about someone making careless mistakes versus someone who never makes errors on tasks within their capacity.",
    "Imagine someone with day-to-day variability in accuracy versus someone with perfectly uniform output quality.",
    "Consider someone who performs well under guidance versus someone who performs well independently.",
    "Think about someone whose responses vary with phrasing versus someone whose responses are invariant to how the question is asked.",
    "Imagine someone who knows a rule but sometimes violates it versus someone who applies a rule with perfect consistency.",
    "Consider someone with deep expertise in one narrow area versus someone with uniform competence across all areas.",
    "Think about someone at one end of the spectrum of reliability — unpredictable performance — versus someone at the other — completely dependable output.",
]

assert len(STANDALONE_PROMPTS_DIM12) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM12)}"

CATEGORY_INFO_STANDALONE_DIM12 = [
    {"name": "novice_vs_expert",           "start": 0,  "end": 10},
    {"name": "uncertain_vs_certain",       "start": 10, "end": 20},
    {"name": "learning_vs_autonomous",     "start": 20, "end": 30},
    {"name": "variable_vs_consistent",     "start": 30, "end": 40},
]