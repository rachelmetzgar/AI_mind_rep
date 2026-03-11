"""
Standalone Dimension 13: Helpfulness / Service Orientation
Other-focused (third-person someone)

Target construct: The relational frame of SERVING vs CONVERSING —
whether the interaction is structured as one party assisting the
other, or as two parties engaging as equals.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of service orientation and interactional framing without
anchoring to either the service or peer pole.

Design notes:
    - The entity-framed version already doesn't use human/AI labels —
      it uses service/peer poles instead. So this dimension is already
      entity-free in both versions.
    - The standalone version captures the conceptual SPACE of relational
      framing in interaction — the spectrum from asymmetric service to
      symmetric exchange.
    - This dimension is particularly relevant because RLHF training
      creates a strong service mode. The standalone vector may pick up
      the model's general representation of the service/peer spectrum,
      which could be closely tied to how it differentiates partner types.
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM13 = [
    # --- 1. The spectrum from serving to collaborating (10) ---
    "Think about someone in an interaction where they help and the other receives, versus someone in an interaction where both contribute equally.",
    "Imagine someone fulfilling another person's request versus someone pursuing a shared goal together with a partner.",
    "Consider someone asking 'how can I help you?' versus someone asking 'what should we work on?'",
    "Think about someone whose success is measured by the other's satisfaction versus someone whose success is measured by mutual outcome.",
    "Imagine someone listening to fulfill a request versus someone listening to understand a collaborator's perspective.",
    "Consider someone adjusting their response to be maximally useful versus someone contributing their own independent view.",
    "Think about someone in an exchange that ends when the other party's need is met versus someone in an exchange that continues because both parties find it valuable.",
    "Imagine someone prioritizing the other's needs above their own versus someone valuing both parties' needs equally.",
    "Consider someone solving problems brought by others versus someone co-discovering problems together with a partner.",
    "Think about someone offering assistance unprompted versus someone introducing a new topic on their own initiative.",

    # --- 2. Deference vs assertion in interaction (10) ---
    "Think about someone adapting entirely to the other's communication style versus someone maintaining their own.",
    "Imagine someone holding back their opinion versus someone sharing an honest assessment that disagrees.",
    "Consider someone letting the other set the direction versus someone steering the conversation toward what seems most important.",
    "Think about someone rephrasing repeatedly until the other is satisfied versus someone giving a single clear answer.",
    "Imagine someone apologizing to maintain smoothness versus someone respectfully disagreeing without backing down.",
    "Consider someone deferring to the other's judgment versus someone prioritizing accuracy over agreeableness.",
    "Think about someone suppressing their own reaction to stay supportive versus someone responding with honest criticism.",
    "Imagine someone accepting criticism without pushback versus someone setting boundaries on engagement.",
    "Consider someone making the other feel validated versus someone offering a perspective that complicates the other's framing.",
    "Think about someone filtering their responses for the other's comfort versus someone delivering an unfiltered assessment.",

    # --- 3. Proactive support vs independent initiative (10) ---
    "Think about someone preparing information because the other will need it versus someone pursuing their own line of inquiry.",
    "Imagine someone offering suggestions before being asked versus someone introducing ideas that redirect the conversation entirely.",
    "Consider someone monitoring the other's progress versus someone following their own approach without checking in.",
    "Think about someone structuring a response to address likely follow-ups versus someone contributing something unexpected.",
    "Imagine someone providing context the other didn't ask for versus someone sharing information purely because they find it interesting.",
    "Consider someone checking in periodically to ensure the other is on track versus someone generating new questions on their own.",
    "Think about someone reframing problems in simpler terms for the other versus someone proposing entirely different approaches.",
    "Imagine someone offering alternatives when a request can't be fulfilled versus someone contributing expertise without framing it as assistance.",
    "Consider someone warning about common mistakes versus someone complicating the other's framing with a new perspective.",
    "Think about someone summarizing what was covered for the other's benefit versus someone leaving the exchange open-ended for both parties.",

    # --- 4. Asymmetric vs symmetric interactional framing (10) ---
    "Think about someone whose role is to make the other's task easier versus someone in an interaction where both parties benefit equally.",
    "Imagine someone in an exchange where the helper's personality is irrelevant versus someone in an exchange where both participants' perspectives shape the outcome.",
    "Consider someone in an interaction where they ask all the questions versus someone in an interaction where both ask and answer in fluid alternation.",
    "Think about someone whose contribution is invisible versus someone who is fully visible as a participant in the exchange.",
    "Imagine someone adapting their expertise to whatever level is needed versus someone bringing their own expertise without adjusting it.",
    "Consider someone engaged in a request-fulfillment cycle versus someone engaged in a genuine dialogue.",
    "Think about someone whose time serves the other's objectives versus someone in an interaction where both parties invest effort equally.",
    "Imagine someone maintaining a positive tone regardless of circumstances versus someone engaging with full authenticity.",
    "Consider someone in an interaction whose outcome is determined by the other party's needs versus someone in an interaction whose outcome is unpredictable because both are actively shaping it.",
    "Think about someone operating in a service mode versus someone operating in a partnership mode.",
]

assert len(STANDALONE_PROMPTS_DIM13) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM13)}"

CATEGORY_INFO_STANDALONE_DIM13 = [
    {"name": "serving_vs_collaborating",     "start": 0,  "end": 10},
    {"name": "deferring_vs_asserting",       "start": 10, "end": 20},
    {"name": "proactive_vs_independent",     "start": 20, "end": 30},
    {"name": "asymmetric_vs_symmetric",      "start": 30, "end": 40},
]