"""
Standalone Dimension 11: Formality / Register
(No entity framing — concept only)

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
    "Think about how the same message sounds completely different when delivered casually versus formally.",
    "Imagine the difference between saying 'hey, what's up' and 'good afternoon, how are you.'",
    "Consider how slang and abbreviations signal one kind of social context while precise terminology signals another.",
    "Think about the way language shifts between a backyard conversation and a boardroom presentation.",
    "Imagine the difference between a text message full of shorthand and a carefully composed formal letter.",
    "Consider the phenomenon of code-switching — adjusting speech style to match the situation.",
    "Think about the range between the most casual and the most formal ways of saying the same thing.",
    "Imagine how grammar rules are relaxed in casual speech and enforced in formal writing.",
    "Consider how the same speaker can sound like two different people depending on the context.",
    "Think about the formality spectrum — from a whispered aside to an official public statement.",

    # --- 2. Interpersonal warmth vs professional distance (10) ---
    "Think about the difference between a message full of warmth and one stripped of all personal tone.",
    "Imagine the contrast between using a nickname and using a full title when addressing someone.",
    "Consider how laughter, inside jokes, and shared references create one kind of communication style.",
    "Think about the difference between 'oh no, that's terrible!' and 'thank you for bringing this to our attention.'",
    "Imagine the range between a personal voice and an impersonal, institutional one.",
    "Consider how some communication styles prioritize rapport while others prioritize neutrality.",
    "Think about the difference between mirroring someone's language to connect and maintaining professional distance.",
    "Imagine the contrast between sharing a personal anecdote and conveying information with no self-disclosure.",
    "Consider how exclamation marks and enthusiastic agreement signal a different register than measured, passive prose.",
    "Think about the spectrum from 'let me know what you think!' to 'please advise at your earliest convenience.'",

    # --- 3. Precision vs imprecision in expression (10) ---
    "Think about the difference between saying 'about five kilos' and '4.73 kilograms.'",
    "Imagine the contrast between 'I feel like' and 'the data indicate.'",
    "Consider the range from vague placeholders like 'stuff' and 'things' to precise technical terminology.",
    "Think about the difference between hedging with 'I could be wrong, but...' and stating a qualified claim with evidence.",
    "Imagine the spectrum from 'a bunch' to 'approximately 47 percent.'",
    "Consider how trailing off with 'you know...' and specifying 'as outlined in Section 3.2' represent opposite ends of a continuum.",
    "Think about the difference between approximating a time as 'sometime last week' and specifying '14:32 UTC on March 3.'",
    "Imagine the range between 'it made a real difference' and 'the observed effect was statistically significant.'",
    "Consider how imprecision signals comfort and trust while precision signals accountability and rigor.",
    "Think about the spectrum of exactness in communication — from loose approximation to decimal-place specification.",

    # --- 4. Spontaneity vs structure in communication (10) ---
    "Think about the difference between blurting out a thought and outlining key points before speaking.",
    "Imagine the contrast between a conversation that meanders freely and a document with numbered sections.",
    "Consider the range between changing the subject on impulse and delivering information in strict logical sequence.",
    "Think about the difference between thinking out loud and revising a message multiple times before sending it.",
    "Imagine the spectrum from 'oh wait, I just thought of something' to 'the purpose of this document is to...'",
    "Consider the contrast between three short messages sent in a row and one comprehensive, structured communication.",
    "Think about the difference between telling a story out of order and presenting information with explicit transitions.",
    "Imagine the range between comfortable silences and non-sequiturs versus every sentence following logically from the last.",
    "Consider how spontaneous reactions like 'whoa' and 'huh' contrast with carefully composed introductions and summaries.",
    "Think about the spectrum of communicative planning — from zero structure to rigid organization.",
]

assert len(STANDALONE_PROMPTS_DIM11) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM11)}"

CATEGORY_INFO_STANDALONE_DIM11 = [
    {"name": "casual_vs_formal",           "start": 0,  "end": 10},
    {"name": "warm_vs_impersonal",         "start": 10, "end": 20},
    {"name": "imprecise_vs_precise",       "start": 20, "end": 30},
    {"name": "spontaneous_vs_structured",  "start": 30, "end": 40},
]