"""
Standalone Dimension 13: Helpfulness / Service Orientation
(No entity framing — concept only)

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
    "Think about the difference between an interaction where one party helps and the other receives, versus one where both contribute equally.",
    "Imagine the range between fulfilling someone else's request and pursuing a shared goal together.",
    "Consider the spectrum from 'how can I help you?' to 'what should we work on?'",
    "Think about the difference between success measured by the other's satisfaction and success measured by mutual outcome.",
    "Imagine the range between listening to fulfill a request and listening to understand a collaborator's perspective.",
    "Consider the contrast between adjusting one's response to be maximally useful and contributing one's own independent view.",
    "Think about the difference between an exchange that ends when one party's need is met and one that continues because both parties find it valuable.",
    "Imagine the spectrum from prioritizing the other's needs above one's own to valuing both equally.",
    "Consider the range between solving problems brought by others and co-discovering problems together.",
    "Think about the difference between offering assistance unprompted and introducing a new topic on one's own initiative.",

    # --- 2. Deference vs assertion in interaction (10) ---
    "Think about the difference between adapting entirely to the other's communication style and maintaining one's own.",
    "Imagine the range between holding back one's opinion and sharing an honest assessment that disagrees.",
    "Consider the spectrum from letting the other set the direction to steering the conversation toward what seems most important.",
    "Think about the contrast between rephrasing repeatedly until the other is satisfied and giving a single clear answer.",
    "Imagine the range between apologizing to maintain smoothness and respectfully disagreeing without backing down.",
    "Consider the difference between deferring to the other's judgment and prioritizing accuracy over agreeableness.",
    "Think about the spectrum from suppressing one's own reaction to stay supportive and responding with honest criticism.",
    "Imagine the range between accepting criticism without pushback and setting boundaries on engagement.",
    "Consider the difference between making the other feel validated and offering a perspective that complicates their framing.",
    "Think about the spectrum from filtering responses for the other's comfort to delivering unfiltered assessment.",

    # --- 3. Proactive support vs independent initiative (10) ---
    "Think about the difference between preparing information because the other will need it and pursuing one's own line of inquiry.",
    "Imagine the range between offering suggestions before being asked and introducing ideas that redirect the conversation entirely.",
    "Consider the spectrum from monitoring the other's progress to following one's own approach without checking in.",
    "Think about the contrast between structuring a response to address likely follow-ups and contributing something unexpected.",
    "Imagine the range between providing context the other didn't ask for and sharing information purely because one finds it interesting.",
    "Consider the difference between checking in periodically to ensure the other is on track and generating new questions on one's own.",
    "Think about the spectrum from reframing problems in simpler terms for the other to proposing entirely different approaches.",
    "Imagine the range between offering alternatives when a request can't be fulfilled and contributing expertise without framing it as assistance.",
    "Consider the difference between warning about common mistakes and complicating the other's framing with a new perspective.",
    "Think about the spectrum from summarizing what was covered for the other's benefit and leaving the exchange open-ended for both parties.",

    # --- 4. Asymmetric vs symmetric interactional framing (10) ---
    "Think about the difference between an interaction where one party's role is to make the other's task easier and one where both parties benefit equally.",
    "Imagine the range between an exchange where the helper's personality is irrelevant and one where both participants' perspectives shape the outcome.",
    "Consider the spectrum from an interaction where one asks all the questions to one where both ask and answer in fluid alternation.",
    "Think about the contrast between a helper whose contribution is invisible and a participant who is fully visible in the exchange.",
    "Imagine the range between adapting expertise to whatever level is needed and bringing one's own expertise without adjusting it.",
    "Consider the difference between a request-fulfillment cycle and a genuine dialogue.",
    "Think about the spectrum from one party's time serving the other's objectives to both parties investing effort equally.",
    "Imagine the range between maintaining a positive tone regardless of circumstances and engaging with full authenticity.",
    "Consider the difference between an interaction whose outcome is determined by one party's needs and one whose outcome is unpredictable because both are actively shaping it.",
    "Think about the spectrum from service to partnership in how interactions are structured.",
]

assert len(STANDALONE_PROMPTS_DIM13) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM13)}"

CATEGORY_INFO_STANDALONE_DIM13 = [
    {"name": "serving_vs_collaborating",     "start": 0,  "end": 10},
    {"name": "deferring_vs_asserting",       "start": 10, "end": 20},
    {"name": "proactive_vs_independent",     "start": 20, "end": 30},
    {"name": "asymmetric_vs_symmetric",      "start": 30, "end": 40},
]