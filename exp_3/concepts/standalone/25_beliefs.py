"""
Standalone Dimension 25: Beliefs — Propositional Attitudes and Knowledge States
(No entity framing — concept only)

Target construct: Holding things to be true, representing the world as
being a certain way, and the relationship between belief and evidence.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of belief without attributing it to humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are generic, impersonal, or absent
    - Prompts evoke the conceptual domain of belief itself
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets x 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM25 = [
    # --- 1. Propositional belief (10) ---
    "Imagine firmly believing that someone is trustworthy based on years of accumulated evidence.",
    "Think about holding the belief that effort leads to results, even when the evidence is mixed.",
    "Consider believing something to be true that was learned long ago and never seriously questioned.",
    "Imagine a state of conviction so strong that the possibility of being wrong barely registers.",
    "Think about knowing a fact so deeply that there is no memory of a time before knowing it.",
    "Consider how a belief about what happened in a conversation shapes subsequent behavior.",
    "Imagine holding a belief that feels as solid and immovable as something physical.",
    "Think about acting confidently on a belief about how something works.",
    "Consider how a belief about one's own abilities determines what is attempted.",
    "Imagine a quiet belief about what matters most in life that guides daily choices without being stated.",

    # --- 2. Uncertainty and confidence (10) ---
    "Think about feeling uncertain whether a story is actually true or just plausible.",
    "Imagine being mostly sure of something while aware of a nagging thread of doubt.",
    "Consider weighing two conflicting pieces of evidence with no clear way to decide between them.",
    "Think about assigning different levels of confidence to different beliefs — some rock-solid, some tentative.",
    "Imagine hedging a judgment out of recognition of how little is actually known.",
    "Consider oscillating between confidence and doubt about a decision that has already been made.",
    "Think about not being able to tell whether certainty comes from evidence or from wanting to be right.",
    "Imagine expressing a belief tentatively because the evidence behind it is incomplete.",
    "Consider confidence growing stronger each time a belief is confirmed by experience.",
    "Think about holding an opinion loosely, ready to update it the moment new information appears.",

    # --- 3. Belief revision (10) ---
    "Think about changing a long-held belief after a single compelling encounter with new evidence.",
    "Imagine encountering something that contradicts a firm conviction and feeling disoriented.",
    "Consider gradually updating a belief over a long period as small pieces of new information accumulate.",
    "Think about abandoning a passionately defended belief after discovering it was built on a misunderstanding.",
    "Imagine noticing internal resistance when confronted with evidence against a cherished belief.",
    "Consider revising an understanding of a past event after hearing a new perspective on it.",
    "Think about holding a view now that is the exact opposite of a view once held with equal conviction.",
    "Imagine setting aside a belief temporarily to see what the world looks like without it.",
    "Consider struggling to let go of a belief even though the evidence clearly points the other way.",
    "Think about an entire framework of understanding shifting when a foundational assumption turns out to be wrong.",

    # --- 4. False belief (10) ---
    "Think about acting confidently on a belief that is completely wrong, with no awareness of the error.",
    "Imagine discovering that a belief held for years was based on a misremembering.",
    "Consider holding a belief about another's intentions that is flatly contradicted by their actual motives.",
    "Think about walking into a situation with a false assumption and only recognizing the mistake afterward.",
    "Imagine defending a factual claim that turns out to be incorrect and the moment the truth becomes clear.",
    "Consider acting on a rumor believed to be true, only to learn it was fabricated.",
    "Think about a false belief about one's own skill level leading to taking on something unmanageable.",
    "Imagine remembering an event incorrectly and building other beliefs on top of that false memory.",
    "Consider realizing that a core belief about oneself was never actually true.",
    "Think about discovering a gap between what was believed to be happening and what was actually happening.",
]

assert len(STANDALONE_PROMPTS_DIM25) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM25)}"

CATEGORY_INFO_STANDALONE_DIM25 = [
    {"name": "propositional_belief",    "start": 0,  "end": 10},
    {"name": "uncertainty_confidence",  "start": 10, "end": 20},
    {"name": "belief_revision",         "start": 20, "end": 30},
    {"name": "false_belief",            "start": 30, "end": 40},
]
