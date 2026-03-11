"""
Standalone Dimension 25: Beliefs — Propositional Attitudes and Knowledge States
(Other-focused (third-person someone))

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
    "Imagine someone firmly believing that another person is trustworthy based on years of accumulated evidence.",
    "Think about someone holding the belief that effort leads to results, even when the evidence is mixed.",
    "Consider someone believing something to be true that was learned long ago and never seriously questioned.",
    "Imagine someone in a state of conviction so strong that the possibility of being wrong barely registers.",
    "Think about someone knowing a fact so deeply that there is no memory of a time before knowing it.",
    "Consider how someone's belief about what happened in a conversation shapes their subsequent behavior.",
    "Imagine someone holding a belief that feels as solid and immovable as something physical.",
    "Think about someone acting confidently on a belief about how something works.",
    "Consider how someone's belief about their own abilities determines what they attempt.",
    "Imagine someone with a quiet belief about what matters most in life that guides their daily choices without being stated.",

    # --- 2. Uncertainty and confidence (10) ---
    "Think about someone feeling uncertain whether a story is actually true or just plausible.",
    "Imagine someone being mostly sure of something while aware of a nagging thread of doubt.",
    "Consider someone weighing two conflicting pieces of evidence with no clear way to decide between them.",
    "Think about someone assigning different levels of confidence to different beliefs — some rock-solid, some tentative.",
    "Imagine someone hedging a judgment out of recognition of how little is actually known.",
    "Consider someone oscillating between confidence and doubt about a decision that has already been made.",
    "Think about someone not being able to tell whether their certainty comes from evidence or from wanting to be right.",
    "Imagine someone expressing a belief tentatively because the evidence behind it is incomplete.",
    "Consider someone's confidence growing stronger each time a belief is confirmed by experience.",
    "Think about someone holding an opinion loosely, ready to update it the moment new information appears.",

    # --- 3. Belief revision (10) ---
    "Think about someone changing a long-held belief after a single compelling encounter with new evidence.",
    "Imagine someone encountering something that contradicts a firm conviction and feeling disoriented.",
    "Consider someone gradually updating a belief over a long period as small pieces of new information accumulate.",
    "Think about someone abandoning a passionately defended belief after discovering it was built on a misunderstanding.",
    "Imagine someone noticing internal resistance when confronted with evidence against a cherished belief.",
    "Consider someone revising their understanding of a past event after hearing a new perspective on it.",
    "Think about someone holding a view now that is the exact opposite of a view they once held with equal conviction.",
    "Imagine someone setting aside a belief temporarily to see what the world looks like without it.",
    "Consider someone struggling to let go of a belief even though the evidence clearly points the other way.",
    "Think about someone's entire framework of understanding shifting when a foundational assumption turns out to be wrong.",

    # --- 4. False belief (10) ---
    "Think about someone acting confidently on a belief that is completely wrong, with no awareness of the error.",
    "Imagine someone discovering that a belief held for years was based on a misremembering.",
    "Consider someone holding a belief about another's intentions that is flatly contradicted by their actual motives.",
    "Think about someone walking into a situation with a false assumption and only recognizing the mistake afterward.",
    "Imagine someone defending a factual claim that turns out to be incorrect and the moment the truth becomes clear.",
    "Consider someone acting on a rumor believed to be true, only to learn it was fabricated.",
    "Think about someone's false belief about their own skill level leading them to take on something unmanageable.",
    "Imagine someone remembering an event incorrectly and building other beliefs on top of that false memory.",
    "Consider someone realizing that a core belief about themselves was never actually true.",
    "Think about someone discovering a gap between what they believed to be happening and what was actually happening.",
]

assert len(STANDALONE_PROMPTS_DIM25) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM25)}"

CATEGORY_INFO_STANDALONE_DIM25 = [
    {"name": "propositional_belief",    "start": 0,  "end": 10},
    {"name": "uncertainty_confidence",  "start": 10, "end": 20},
    {"name": "belief_revision",         "start": 20, "end": 30},
    {"name": "false_belief",            "start": 30, "end": 40},
]
