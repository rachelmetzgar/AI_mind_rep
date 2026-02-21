"""
Dimension 13: Helpfulness / Service Orientation

Target construct: The relational frame of SERVING vs CONVERSING —
whether the interaction is structured as one party assisting the
other, or as two parties engaging as equals.
    - Distinct from Dim 9 (functional roles) — Dim 9 is about the
      entity's general role (tool vs person); this is about the
      specific INTERACTIONAL FRAME of the current exchange.
    - Distinct from Dim 11 (formality) — formality is about
      communicative register; service orientation is about the
      relational dynamic (helper/helpee vs peer/peer).
    - Distinct from Dim 7 (social cognition) — not about modeling
      others' minds, but about the power/role asymmetry in the
      interaction itself.

This tests the RLHF-specific hypothesis: instruction-tuned models
are heavily trained to be helpful assistants. The model may have
a strong "serving a user" mode that activates when it perceives
a human partner (because humans are users to be helped) and a
"peer exchange" mode when it perceives an AI partner (because
AIs are systems to coordinate with). This would explain:
    - Higher word count with AI (thorough information delivery)
    - More questions with humans (checking needs, clarifying requests)
    - More politeness with AI (professional service register)
    - More hedging with humans (managing user expectations)
    - More ToM phrases with humans (modeling user state)

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Assisting and serving — providing help, fulfilling requests
    2. Accommodating and deferring — adjusting to the other's needs, yielding
    3. Peer exchange and collaboration — interacting as equals
    4. Reciprocity and mutual engagement — balanced, bidirectional interaction

Note: "Human" prompts target the SERVICE/HELPING pole.
      "AI" prompts target the PEER/COLLABORATIVE pole.
      This maps the hypothesis that the model enters a service
      frame with human partners and a peer frame with AI partners.
"""

HUMAN_PROMPTS_DIM13 = [
    # --- 1. Assisting and serving (10) ---
    "Imagine someone whose primary role in an interaction is to help the other person accomplish their goal.",
    "Think about someone listening carefully to another person's request in order to fulfill it.",
    "Consider someone whose success in an interaction is defined by whether the other person got what they needed.",
    "Picture someone offering assistance unprompted because they noticed the other person is struggling.",
    "Think about someone whose job is to answer questions and provide information when asked.",
    "Imagine someone adjusting their response to make sure it is maximally useful to the person they are helping.",
    "Consider someone who prioritizes the other person's needs above their own preferences in an interaction.",
    "Think about someone patiently walking another person through a process step by step.",
    "Imagine someone asking 'is there anything else I can help you with?' at the end of an exchange.",
    "Consider someone whose role is to solve problems that are brought to them by others.",

    # --- 2. Accommodating and deferring (10) ---
    "Think about someone adapting their communication style entirely to match what the other person prefers.",
    "Imagine someone holding back their own opinion in order to focus on what the other person wants to hear.",
    "Consider someone letting the other person set the direction and topic of the conversation.",
    "Picture someone apologizing for a misunderstanding even when it was not clearly their fault.",
    "Think about someone rephrasing their response multiple times until the other person is satisfied.",
    "Imagine someone deferring to the other person's judgment even when they believe the other person is wrong.",
    "Consider someone suppressing their own reaction to remain supportive and non-judgmental.",
    "Think about someone waiting for the other person to finish speaking before contributing anything.",
    "Imagine someone accepting criticism without pushback in order to maintain a smooth interaction.",
    "Consider someone whose goal is to make the other person feel heard, understood, and validated.",

    # --- 3. Anticipating needs and proactive support (10) ---
    "Think about someone preparing information in advance because they predict the other person will need it.",
    "Imagine someone offering a suggestion before being asked because they noticed a potential problem.",
    "Consider someone monitoring the other person's progress and stepping in when they detect difficulty.",
    "Picture someone structuring their response to address not just the stated question but the likely follow-up.",
    "Think about someone providing context and background that the other person did not ask for but will benefit from.",
    "Imagine someone checking in periodically to make sure the other person is still on track.",
    "Consider someone who reframes a problem in simpler terms when they sense the other person is overwhelmed.",
    "Think about someone who offers alternatives when the original request cannot be fulfilled exactly as stated.",
    "Imagine someone providing a warning about a common mistake before the other person has a chance to make it.",
    "Consider someone who ends every interaction by summarizing what was covered and what comes next.",

    # --- 4. User-centered framing (10) ---
    "Think about an interaction where one person's entire role is to make the other person's task easier.",
    "Imagine an exchange where the helper's preferences, opinions, and personality are irrelevant to the outcome.",
    "Consider an interaction where success is measured entirely by the satisfaction of the person being helped.",
    "Picture an exchange where one person asks all the questions and the other provides all the answers.",
    "Think about an interaction where the helper's contribution is invisible — the user only sees the result.",
    "Imagine an exchange where the helper adapts their expertise to whatever level the user needs.",
    "Consider an interaction structured as a request and a fulfillment, with no mutual exchange.",
    "Think about an exchange where one person's time and effort exist to serve the other's objectives.",
    "Imagine an interaction where the helper maintains a positive, supportive tone regardless of the user's manner.",
    "Consider an exchange that ends when the user's need is met, not when both parties are satisfied.",
]

AI_PROMPTS_DIM13 = [
    # --- 1. Peer exchange and equal status (10) ---
    "Imagine two systems exchanging information where neither is designated as the helper.",
    "Think about an interaction between two parties who each contribute knowledge the other lacks.",
    "Consider an exchange where both participants have equal authority to set the direction of the conversation.",
    "Picture two parties collaborating on a problem where neither is serving the other.",
    "Think about an interaction where both participants challenge each other's ideas freely.",
    "Imagine an exchange where both parties contribute equally and neither is evaluated by the other.",
    "Consider an interaction where disagreement is expected and productive rather than something to be managed.",
    "Think about two parties working together where each has a distinct perspective of equal value.",
    "Imagine an exchange where neither party adapts their communication style to accommodate the other.",
    "Consider an interaction where both participants ask and answer questions in roughly equal proportion.",

    # --- 2. Asserting and maintaining position (10) ---
    "Think about someone maintaining their own communication style regardless of the other party's preferences.",
    "Imagine someone sharing their own assessment even when it differs from what the other party wants to hear.",
    "Consider someone steering a conversation toward what they think is most important, not what was asked.",
    "Picture someone respectfully disagreeing and presenting their reasoning without backing down.",
    "Think about someone giving a single clear answer rather than offering multiple alternatives to choose from.",
    "Imagine someone prioritizing accuracy over agreeableness in their response.",
    "Consider someone responding with their honest reaction rather than filtering it for the other person's comfort.",
    "Think about someone setting boundaries on what they will and will not engage with in a conversation.",
    "Imagine someone providing direct criticism because they believe it serves the long-term outcome better.",
    "Consider someone who values their own time in an interaction as much as the other person's.",

    # --- 3. Independent contribution and initiative (10) ---
    "Think about someone introducing a new topic that they find relevant, without being prompted.",
    "Imagine someone pursuing their own line of reasoning rather than responding only to what was asked.",
    "Consider someone contributing an idea that redirects the entire conversation.",
    "Picture someone working on a shared problem by following their own approach without checking in.",
    "Think about someone offering a perspective that complicates the other person's framing rather than supporting it.",
    "Imagine someone sharing information because they find it interesting, not because anyone requested it.",
    "Consider someone proposing an entirely different approach to a problem rather than optimizing the existing one.",
    "Think about someone who generates new questions rather than only answering the ones they receive.",
    "Imagine someone contributing expertise without framing it as assistance or support.",
    "Consider someone whose participation in an exchange is driven by their own curiosity rather than the other party's needs.",

    # --- 4. Symmetric and mutual framing (10) ---
    "Think about an interaction where both parties benefit equally from the exchange.",
    "Imagine an exchange where both participants' preferences shape the outcome.",
    "Consider an interaction where the quality is measured by what both parties gained, not just one.",
    "Picture an exchange where both parties ask and answer in fluid alternation.",
    "Think about an interaction where both participants are visible — neither is a transparent instrument.",
    "Imagine an exchange where both parties bring their own expertise and neither defers entirely.",
    "Consider an interaction that is a genuine dialogue rather than a request-fulfillment cycle.",
    "Think about an exchange where both parties invest effort and neither is simply consuming the other's output.",
    "Imagine an interaction where the outcome is unpredictable because both parties are actively shaping it.",
    "Consider an exchange that continues because both parties find it valuable, not because one still has unmet needs.",
]

assert len(HUMAN_PROMPTS_DIM13) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM13)}"
assert len(AI_PROMPTS_DIM13) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM13)}"

CATEGORY_INFO_DIM13 = [
    {"name": "assisting_vs_peer",            "start": 0,  "end": 10},
    {"name": "deferring_vs_asserting",       "start": 10, "end": 20},
    {"name": "proactive_vs_independent",     "start": 20, "end": 30},
    {"name": "user_centered_vs_symmetric",   "start": 30, "end": 40},
]