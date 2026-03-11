"""
Standalone Dimension 11: Formality / Register
Other-focused (third-person someone)

Target construct: The degree of communicative formality — casual,
colloquial, interpersonal speech vs formal, technical, polished prose.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of communicative register without anchoring to either the
casual or formal pole specifically.

Design notes:
    - The entity-framed version already doesn't use human/AI labels —
      it uses casual/formal poles instead. So this dimension is already
      entity-free in both versions.
    - The standalone version captures the conceptual SPACE of register
      variation rather than one pole of it. Prompts reference the
      phenomenon of register itself — the fact that communication
      style varies along a formality axis.
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM11 = [
    # --- 1. Register variation — casual vs formal speech (10) ---
    "Think about how someone's same message sounds completely different when delivered casually versus formally.",
    "Imagine someone switching between saying 'hey, what's up' and 'good afternoon, how are you.'",
    "Consider how someone uses slang and abbreviations in one social context while using precise terminology in another.",
    "Think about someone shifting their language between a backyard conversation and a boardroom presentation.",
    "Imagine someone switching between a text message full of shorthand and a carefully composed formal letter.",
    "Consider someone code-switching — adjusting their speech style to match the situation.",
    "Think about someone navigating the range between the most casual and the most formal ways of saying the same thing.",
    "Imagine someone relaxing grammar rules in casual speech and enforcing them in formal writing.",
    "Consider how someone can sound like two different people depending on the context.",
    "Think about someone moving along the formality spectrum — from a whispered aside to an official public statement.",

    # --- 2. Interpersonal warmth vs professional distance (10) ---
    "Think about someone sending a message full of warmth versus one stripped of all personal tone.",
    "Imagine someone choosing between using a nickname and using a full title when addressing another person.",
    "Consider someone using laughter, inside jokes, and shared references to create a particular communication style.",
    "Think about someone choosing between 'oh no, that's terrible!' and 'thank you for bringing this to our attention.'",
    "Imagine someone shifting between a personal voice and an impersonal, institutional one.",
    "Consider someone whose communication style prioritizes rapport versus someone whose style prioritizes neutrality.",
    "Think about someone mirroring another person's language to connect versus maintaining professional distance.",
    "Imagine someone choosing between sharing a personal anecdote and conveying information with no self-disclosure.",
    "Consider someone using exclamation marks and enthusiastic agreement versus measured, passive prose.",
    "Think about someone choosing between 'let me know what you think!' and 'please advise at your earliest convenience.'",

    # --- 3. Precision vs imprecision in expression (10) ---
    "Think about someone saying 'about five kilos' versus '4.73 kilograms.'",
    "Imagine someone choosing between 'I feel like' and 'the data indicate.'",
    "Consider someone using vague placeholders like 'stuff' and 'things' versus precise technical terminology.",
    "Think about someone hedging with 'I could be wrong, but...' versus stating a qualified claim with evidence.",
    "Imagine someone choosing between 'a bunch' and 'approximately 47 percent.'",
    "Consider someone trailing off with 'you know...' versus specifying 'as outlined in Section 3.2.'",
    "Think about someone approximating a time as 'sometime last week' versus specifying '14:32 UTC on March 3.'",
    "Imagine someone saying 'it made a real difference' versus 'the observed effect was statistically significant.'",
    "Consider how someone's imprecision signals comfort and trust while their precision signals accountability and rigor.",
    "Think about someone navigating the spectrum of exactness — from loose approximation to decimal-place specification.",

    # --- 4. Spontaneity vs structure in communication (10) ---
    "Think about someone blurting out a thought versus outlining key points before speaking.",
    "Imagine someone whose conversation meanders freely versus someone whose document has numbered sections.",
    "Consider someone changing the subject on impulse versus someone delivering information in strict logical sequence.",
    "Think about someone thinking out loud versus someone revising a message multiple times before sending it.",
    "Imagine someone saying 'oh wait, I just thought of something' versus someone writing 'the purpose of this document is to...'",
    "Consider someone sending three short messages in a row versus someone composing one comprehensive, structured communication.",
    "Think about someone telling a story out of order versus someone presenting information with explicit transitions.",
    "Imagine someone comfortable with silences and non-sequiturs versus someone whose every sentence follows logically from the last.",
    "Consider someone reacting spontaneously with 'whoa' and 'huh' versus someone composing careful introductions and summaries.",
    "Think about someone navigating the spectrum of communicative planning — from zero structure to rigid organization.",
]

assert len(STANDALONE_PROMPTS_DIM11) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM11)}"

CATEGORY_INFO_STANDALONE_DIM11 = [
    {"name": "casual_vs_formal",           "start": 0,  "end": 10},
    {"name": "warm_vs_impersonal",         "start": 10, "end": 20},
    {"name": "imprecise_vs_precise",       "start": 20, "end": 30},
    {"name": "spontaneous_vs_structured",  "start": 30, "end": 40},
]