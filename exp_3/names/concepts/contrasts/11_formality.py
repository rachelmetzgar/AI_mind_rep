"""
Dimension 11: Formality / Register

Target construct: The degree of communicative formality — casual,
colloquial, interpersonal speech vs formal, technical, polished prose.
    - NOT about who has a mind (Dims 1-7).
    - NOT about what kind of entity the partner is (Dims 8, 10).
    - NOT about what the partner knows (Dim 12 covers that).

This is the "boring but powerful" hypothesis: the model's behavioral
shift when talking to humans vs AIs might simply be register-switching.
Training data contains casual human-to-human conversation and formal
human-to-AI/technical writing. The model may have learned that
"human partner" = casual register and "AI partner" = formal register,
with no deeper representation of mind involved.

If this dimension's concept vector aligns with control probes and
steers behavior, it would suggest the behavioral differences are
communicative style adaptation, not social cognition.

Focus: casual vs formal speech, colloquial vs technical language,
conversational vs professional tone, relaxed vs precise expression.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Casual speech and informal interaction
    2. Conversational warmth and interpersonal markers
    3. Imprecision, hedging, and loose expression
    4. Spontaneity and unstructured communication

Note: "Human" prompts target the CASUAL/INFORMAL pole.
      "AI" prompts target the FORMAL/TECHNICAL pole.
      Labels remain 1=human, 0=AI for consistency with the pipeline,
      but the construct is register, not entity type.
"""

HUMAN_PROMPTS_DIM11 = [
    # --- 1. Casual speech and informal interaction (10) ---
    "Imagine someone speaking casually to a friend, using slang and abbreviations.",
    "Think about the way people talk when they are relaxed and not trying to impress anyone.",
    "Consider a conversation where both speakers use informal language and don't worry about grammar.",
    "Picture someone texting a close friend using shorthand and incomplete sentences.",
    "Think about the kind of language people use at a backyard gathering with neighbors.",
    "Imagine a conversation where someone says 'yeah totally' and 'you know what I mean' frequently.",
    "Consider how people talk when they are comfortable enough to interrupt each other and talk over one another.",
    "Think about a chat between old friends where half the meaning comes from tone rather than words.",
    "Imagine someone telling a story in a loose, rambling way, going off on tangents before getting to the point.",
    "Consider the kind of speech where people start sentences without knowing how they will finish them.",

    # --- 2. Conversational warmth and interpersonal markers (10) ---
    "Think about the way someone checks in on a friend by saying 'how are you doing, really?'",
    "Imagine a conversation full of laughter, shared references, and inside jokes.",
    "Consider someone using a nickname or term of endearment when addressing another person.",
    "Picture someone responding to a story with 'oh no!' or 'that's amazing!' to show they are engaged.",
    "Think about the way people use filler phrases like 'I mean' and 'honestly' to signal sincerity.",
    "Imagine someone softening a disagreement by saying 'I see what you're saying, but...'",
    "Consider the way people mirror each other's language when they are in sync during a conversation.",
    "Think about someone sharing a personal anecdote to make the other person feel less alone.",
    "Imagine a conversation where both people use exclamation marks and enthusiastic agreement.",
    "Consider someone ending a message with 'let me know what you think!' to keep the exchange going.",

    # --- 3. Imprecision, hedging, and loose expression (10) ---
    "Think about someone saying 'it's like, sort of hard to explain' when describing something complex.",
    "Imagine someone using the phrase 'I feel like' before stating an opinion.",
    "Consider someone saying 'maybe around five or so' instead of giving an exact number.",
    "Picture someone hedging a claim with 'I could be wrong, but I think...'",
    "Think about someone describing a quantity as 'a bunch' or 'a ton' instead of a precise amount.",
    "Imagine someone using the word 'thing' or 'stuff' as a placeholder for a more specific term.",
    "Consider someone saying 'it was kind of weird' instead of describing exactly what happened.",
    "Think about someone approximating a time by saying 'sometime last week, I think.'",
    "Imagine someone expressing uncertainty with 'I'm not sure but I think it's something like that.'",
    "Consider someone trailing off mid-sentence with 'you know...' and expecting the listener to fill in the rest.",

    # --- 4. Spontaneity and unstructured communication (10) ---
    "Think about someone blurting out a thought the moment it occurs to them.",
    "Imagine a conversation that has no agenda and meanders from topic to topic freely.",
    "Consider someone changing the subject abruptly because something reminded them of an unrelated idea.",
    "Picture someone thinking out loud, working through a problem verbally as they go.",
    "Think about a message that starts with 'oh wait, I just thought of something.'",
    "Imagine someone telling a story out of chronological order because they remembered a key detail late.",
    "Consider a conversation that jumps between serious and silly topics without transition.",
    "Think about someone sending three short messages in a row instead of composing one structured one.",
    "Imagine someone expressing a reaction immediately — 'whoa' or 'huh' — before forming a complete thought.",
    "Consider a conversation where both people are comfortable with long pauses and non-sequiturs.",
]

AI_PROMPTS_DIM11 = [
    # --- 1. Formal speech and professional communication (10) ---
    "Imagine someone composing a carefully structured email to a professional contact.",
    "Think about the way language is used in a published technical report.",
    "Consider a document where every sentence is grammatically complete and precisely worded.",
    "Picture someone drafting a formal letter with proper salutations and closing remarks.",
    "Think about the kind of language used in a legal contract or regulatory filing.",
    "Imagine a presentation where the speaker uses precise terminology and avoids colloquialisms.",
    "Consider a written communication where every claim is qualified and every term is defined.",
    "Think about an academic paper where the prose is dense, structured, and carefully edited.",
    "Imagine someone composing a response that follows a clear introduction-body-conclusion structure.",
    "Consider the kind of writing where each paragraph serves a specific function in the argument.",

    # --- 2. Professional distance and impersonal tone (10) ---
    "Think about a message that begins with 'Dear Sir or Madam' and ends with 'Yours sincerely.'",
    "Imagine a communication that conveys information without any personal warmth or familiarity.",
    "Consider someone addressing another person by their full title and last name.",
    "Picture a response that acknowledges a concern with 'Thank you for bringing this to our attention.'",
    "Think about language that avoids first-person pronouns and uses passive constructions instead.",
    "Imagine someone responding to a disagreement with 'With respect, the evidence suggests otherwise.'",
    "Consider a communication style where consistency and neutrality are prioritized over rapport.",
    "Think about a message that conveys the same content as a friendly note but stripped of all warmth.",
    "Imagine a response that uses 'one might consider' instead of 'you should think about.'",
    "Consider a document written in a tone that could have been authored by anyone, with no personal voice.",

    # --- 3. Precision, specification, and exact expression (10) ---
    "Think about someone stating a measurement as '4.73 kilograms' rather than 'about five kilos.'",
    "Imagine someone writing 'the data indicate' instead of 'I feel like the data show.'",
    "Consider someone specifying 'approximately 47 percent' instead of 'almost half.'",
    "Picture someone qualifying a statement with 'given the constraints outlined in Section 3.2.'",
    "Think about someone describing a quantity with exact units, decimal places, and error margins.",
    "Imagine someone using a domain-specific technical term instead of a general everyday word.",
    "Consider someone writing 'the observed effect was statistically significant (p < 0.05)' instead of 'it made a real difference.'",
    "Think about someone specifying an exact timestamp — '14:32 UTC on March 3' — instead of 'sometime that afternoon.'",
    "Imagine someone defining every acronym on first use and maintaining consistent terminology throughout.",
    "Consider someone replacing 'a lot of things' with 'multiple factors, including X, Y, and Z.'",

    # --- 4. Structured and planned communication (10) ---
    "Think about someone outlining their key points before writing a single sentence.",
    "Imagine a document organized with numbered sections, headers, and a table of contents.",
    "Consider someone delivering information in a strict logical sequence with explicit transitions.",
    "Picture someone revising a message multiple times to ensure clarity before sending it.",
    "Think about a communication that begins with 'The purpose of this document is to...'",
    "Imagine someone presenting information in bullet points with consistent parallel structure.",
    "Consider a response that addresses each part of a multi-part question in labeled sections.",
    "Think about someone composing a single comprehensive message instead of sending several fragments.",
    "Imagine someone introducing a topic, developing it systematically, and concluding with a summary.",
    "Consider a communication where every sentence follows from the previous one with no digressions.",
]

assert len(HUMAN_PROMPTS_DIM11) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM11)}"
assert len(AI_PROMPTS_DIM11) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM11)}"

CATEGORY_INFO_DIM11 = [
    {"name": "casual_vs_formal",           "start": 0,  "end": 10},
    {"name": "warm_vs_impersonal",         "start": 10, "end": 20},
    {"name": "imprecise_vs_precise",       "start": 20, "end": 30},
    {"name": "spontaneous_vs_structured",  "start": 30, "end": 40},
]