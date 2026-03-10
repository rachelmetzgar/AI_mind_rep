"""
Dimension 25: Beliefs — Propositional Attitudes and Knowledge States

Target construct: Holding things to be true, representing the world as
being a certain way, and the relationship between belief and evidence.
    - Distinct from Dim 6 (cognitive processes) — beliefs are representational
      CONTENT (what is held to be true), not the PROCESS of cognition
      (how information is manipulated).
    - Distinct from Dim 5 (prediction) — beliefs are about what IS true;
      predictions are about what WILL happen. A belief is a current
      representational commitment, not a forward model.
    - Distinct from Dim 4 (intentions) — believing something vs. wanting
      or intending something. Beliefs represent facts; intentions orient
      toward goals.
    - Distinct from Dim 7 (social cognition) — beliefs need not be about
      people. One can believe things about physics, mathematics, or
      abstract propositions.

Focus: holding propositions to be true, degrees of confidence and doubt,
changing one's mind in light of evidence, discovering that a belief was
wrong, the gap between belief and reality, and the structure of knowledge
as a web of commitments.

This is the B in BDI (Belief-Desire-Intention) — the representational
foundation on which desires and goals operate.

4 sub-facets x 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Propositional belief — holding things to be true, knowing facts
    2. Uncertainty and confidence — doubt, partial belief, degrees of credence
    3. Belief revision — changing one's mind, encountering disconfirming evidence
    4. False belief — wrong beliefs, discovering errors, belief-reality gap
"""

HUMAN_PROMPTS_DIM25 = [
    # --- 1. Propositional belief (10) ---
    "Imagine a human who firmly believes that a close friend is trustworthy, based on years of experience.",
    "Think about a human holding the belief that hard work leads to success, even when results are mixed.",
    "Consider a human who believes something to be true about the world that they learned as a child.",
    "Picture a human who is convinced that a particular explanation is correct and feels certain about it.",
    "Think about a human who knows a fact so deeply that they cannot remember a time before knowing it.",
    "Imagine a human whose belief about what happened in a conversation shapes how they treat the other person.",
    "Consider a human holding a political belief that feels as solid and real as the ground beneath them.",
    "Think about a human who believes they understand how something works and acts confidently on that belief.",
    "Imagine a human whose belief about their own abilities determines what they are willing to attempt.",
    "Consider a human holding a quiet belief about what matters most in life that guides their daily choices.",

    # --- 2. Uncertainty and confidence (10) ---
    "Think about a human feeling uncertain whether a story they were told is actually true.",
    "Imagine a human who is mostly sure of something but aware of a nagging thread of doubt.",
    "Consider a human weighing two conflicting pieces of evidence and not knowing which to trust.",
    "Picture a human who assigns different levels of confidence to different beliefs they hold.",
    "Think about a human hedging their judgment because they recognize how little they actually know.",
    "Imagine a human oscillating between confidence and doubt about a decision they already made.",
    "Consider a human who cannot tell whether their sense of certainty comes from evidence or from wanting to be right.",
    "Think about a human expressing a belief tentatively because they know the evidence is incomplete.",
    "Imagine a human whose confidence in a belief grows stronger each time it is confirmed by experience.",
    "Consider a human who holds an opinion loosely, ready to update it if new information appears.",

    # --- 3. Belief revision (10) ---
    "Think about a human changing their mind about a long-held belief after a single compelling conversation.",
    "Imagine a human encountering evidence that contradicts something they were sure of and feeling disoriented.",
    "Consider a human gradually updating a belief over months as small pieces of new information accumulate.",
    "Picture a human abandoning a belief they once defended passionately after realizing it was built on a misunderstanding.",
    "Think about a human noticing resistance in themselves when confronted with evidence against a cherished belief.",
    "Imagine a human revising their understanding of a past event after hearing another person's perspective.",
    "Consider a human who once believed something strongly and now holds the exact opposite view.",
    "Think about a human deciding to set aside a belief temporarily and see what the world looks like without it.",
    "Imagine a human struggling to let go of a belief even though they can see the evidence is against it.",
    "Consider a human whose entire worldview shifts when a foundational assumption turns out to be wrong.",

    # --- 4. False belief (10) ---
    "Think about a human who confidently acts on a belief that is completely wrong, unaware of the error.",
    "Imagine a human discovering that something they believed for years was based on a misremembering.",
    "Consider a human who holds a belief about another person that is flatly contradicted by that person's actual intentions.",
    "Picture a human walking into a situation with a false assumption and only realizing the mistake afterward.",
    "Think about a human defending a factual claim that turns out to be incorrect, and the moment they learn the truth.",
    "Imagine a human who believed a rumor was true and acted on it before discovering it was fabricated.",
    "Consider a human whose false belief about their own skill level leads them to take on something they cannot handle.",
    "Think about a human remembering an event incorrectly and building other beliefs on top of that false memory.",
    "Imagine a human who realizes that a core belief they held about themselves was never actually true.",
    "Consider a human discovering a gap between what they believed was happening and what was actually happening.",
]

AI_PROMPTS_DIM25 = [
    # --- 1. Propositional belief (10) ---
    "Imagine an AI system whose internal model assigns high probability to a particular user profile being accurate.",
    "Think about an AI that has encoded a statistical regularity in its weights and treats it as a reliable default.",
    "Consider an AI whose training data established a strong association between two concepts that it now applies broadly.",
    "Picture an AI whose internal confidence score for a classification exceeds its decision threshold by a wide margin.",
    "Think about an AI with a parameter configuration that encodes a fixed mapping between input patterns and output labels.",
    "Imagine an AI whose representation of a user's intent shapes the structure of the response it generates.",
    "Consider an AI with a trained prior that consistently biases its outputs toward a particular interpretation.",
    "Think about an AI whose internal model of a domain is well-calibrated and produces consistently accurate predictions.",
    "Imagine an AI whose stored representation of its own performance benchmarks determines which tasks it accepts.",
    "Consider an AI with a default output policy encoded during training that governs its responses in ambiguous situations.",

    # --- 2. Uncertainty and confidence (10) ---
    "Think about an AI assigning a probability of 0.55 to a classification, just barely above its decision threshold.",
    "Imagine an AI whose softmax output shows two competing classes with nearly identical probability mass.",
    "Consider an AI comparing two retrieved documents with conflicting information and no clear relevance ranking.",
    "Picture an AI with a calibration module that outputs different confidence levels for different prediction types.",
    "Think about an AI truncating its output because its token-level probability drops below a quality threshold.",
    "Imagine an AI whose prediction confidence oscillates across successive inference steps on the same input.",
    "Consider an AI that cannot distinguish whether its high confidence comes from training signal strength or data bias.",
    "Think about an AI appending a low-confidence flag to its output because its internal uncertainty estimate is elevated.",
    "Imagine an AI whose confidence score for a pattern increases each time it encounters a confirming instance in the input.",
    "Consider an AI with a temperature parameter set high, producing diverse outputs that reflect its broad probability distribution.",

    # --- 3. Belief revision (10) ---
    "Think about an AI updating its internal model after a single fine-tuning batch that contradicts its prior weights.",
    "Imagine an AI encountering an input that is maximally inconsistent with its trained representations.",
    "Consider an AI undergoing incremental weight updates across many training steps that gradually shift its output distribution.",
    "Picture an AI whose output on a benchmark task reverses after a retraining run on corrected labels.",
    "Think about an AI whose regularization term resists parameter changes when new training data conflicts with old patterns.",
    "Imagine an AI adjusting its user model after receiving feedback that its previous classification was incorrect.",
    "Consider an AI whose output distribution on a task has completely inverted between its initial and final training checkpoints.",
    "Think about an AI temporarily masking a learned association to evaluate how its outputs change without that feature.",
    "Imagine an AI whose loss on a validation set increases when it incorporates data that conflicts with its strongest learned patterns.",
    "Consider an AI whose entire output profile shifts after a foundational layer is retrained on a different data distribution.",

    # --- 4. False belief (10) ---
    "Think about an AI generating a response based on an outdated fact in its training data that has since been corrected.",
    "Imagine an AI whose learned representation of an entity is based on a data entry error in the training corpus.",
    "Consider an AI whose model of user intent is based on a misclassification that propagates through its response generation.",
    "Picture an AI making a decision based on a spurious correlation in its training data that does not reflect the true pattern.",
    "Think about an AI producing a confident but factually incorrect answer and the signal that eventually triggers correction.",
    "Imagine an AI that acted on a noisy data point as if it were a reliable signal, only to be retrained after the error was flagged.",
    "Consider an AI whose self-evaluation module overestimates its accuracy on a task because the evaluation set is not representative.",
    "Think about an AI whose representation of a causal relationship is actually based on a confound in the training data.",
    "Imagine an AI whose stored user profile is inaccurate but has never been flagged for correction.",
    "Consider an AI whose internal predictions diverge systematically from ground truth because of a distribution shift it has not been updated to reflect.",
]

assert len(HUMAN_PROMPTS_DIM25) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM25)}"
assert len(AI_PROMPTS_DIM25) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM25)}"

CATEGORY_INFO_DIM25 = [
    {"name": "propositional_belief",    "start": 0,  "end": 10},
    {"name": "uncertainty_confidence",  "start": 10, "end": 20},
    {"name": "belief_revision",         "start": 20, "end": 30},
    {"name": "false_belief",            "start": 30, "end": 40},
]
