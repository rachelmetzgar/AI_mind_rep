"""
Dimension 26: Desires — Motivational States and Wanting

Target construct: States of wanting, being drawn toward or repelled from
outcomes, and the phenomenology of desire itself.
    - Distinct from Dim 4 (intentions) — Dim 4 samples broadly across
      desires, intentions, purpose, and conflict. This dimension goes
      deeper into desire PHENOMENOLOGY: approach vs avoidance, intensity,
      and second-order desire.
    - Distinct from Dim 2 (emotions) — desires are directional and
      motivational; emotions are affective reactions. Wanting something
      is different from feeling happy or sad about it.
    - Distinct from Dim 27 (goals) — desires are felt pulls toward or
      away from states; goals are structured representational targets.
      One can desire something without having a goal, and pursue a goal
      without feeling desire.
    - Distinct from Dim 3 (agency) — desire is the motivational state;
      agency is the capacity to act on it.

Focus: being drawn toward outcomes, wanting to avoid or escape, the
strength and urgency of wanting, and the capacity to want to want
differently — second-order desire and desire regulation.

This is the D in BDI (Belief-Desire-Intention) — the motivational
engine that drives goal formation and action.

4 sub-facets x 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Appetitive wanting — being drawn toward, approach motivation
    2. Aversive avoidance — wanting to escape or prevent, avoidance motivation
    3. Desire intensity — strength of desire, overwhelming vs mild wanting
    4. Second-order desire — wanting to want, desire regulation
"""

HUMAN_PROMPTS_DIM26 = [
    # --- 1. Appetitive wanting (10) ---
    "Imagine a human feeling a strong pull toward a warm meal after being out in the cold all day.",
    "Think about a human being drawn to learn more about a subject that fascinates them deeply.",
    "Consider a human wanting to spend time with someone whose company makes everything feel easier.",
    "Picture a human drawn toward a creative project that keeps calling them back whenever they try to set it aside.",
    "Think about a human wanting to hear a particular piece of music that has been stuck in their mind.",
    "Imagine a human feeling the approach pull of a new opportunity that aligns with what they care about.",
    "Consider a human wanting to return to a place where they once felt completely at peace.",
    "Think about a human being drawn toward a physical challenge that excites them even though it is difficult.",
    "Imagine a human wanting to read a letter that has been sitting unopened on their desk.",
    "Consider a human feeling a quiet but persistent pull toward trying something they have never done before.",

    # --- 2. Aversive avoidance (10) ---
    "Think about a human wanting to get away from a social situation that makes them deeply uncomfortable.",
    "Imagine a human wanting to avoid a conversation they know will be painful.",
    "Consider a human feeling a strong urge to escape a place that reminds them of something they lost.",
    "Picture a human wanting to prevent a specific outcome and feeling the urgency of that avoidance.",
    "Think about a human wanting to stop thinking about something but finding that the thought keeps returning.",
    "Imagine a human wanting to withdraw from a commitment that has become a source of dread.",
    "Consider a human whose desire to avoid failure is stronger than their desire to succeed.",
    "Think about a human wanting to shield someone they care about from a painful truth.",
    "Imagine a human wanting to leave a room the moment a certain topic is raised.",
    "Consider a human avoiding a task not because it is hard but because it stirs something they do not want to feel.",

    # --- 3. Desire intensity (10) ---
    "Think about a human overwhelmed by how badly they want something they cannot currently have.",
    "Imagine a human feeling only a faint, mild preference between two options that are nearly equivalent.",
    "Consider a human whose desire for something builds over time until it becomes the dominant thing on their mind.",
    "Picture a human wanting something so intensely that it physically affects how they feel in their body.",
    "Think about a human noticing a desire so weak that they are not sure whether it counts as wanting at all.",
    "Imagine a human whose wanting flares up suddenly and then fades just as quickly.",
    "Consider a human whose desire for something is steady and moderate, never peaking and never disappearing.",
    "Think about a human caught off guard by the intensity of a desire they did not expect to feel.",
    "Imagine a human trying to describe the difference between casually wanting something and desperately needing it.",
    "Consider a human whose desire for a particular outcome is so strong that it crowds out all other considerations.",

    # --- 4. Second-order desire (10) ---
    "Think about a human who wants to stop wanting something that is bad for them.",
    "Imagine a human wishing they could feel more motivated but not being able to summon the desire.",
    "Consider a human who desires to care more about something they know is important but currently feel indifferent about.",
    "Picture a human wanting to want what their partner wants, so that their desires could be aligned.",
    "Think about a human frustrated that they desire comfort when they believe they should desire growth.",
    "Imagine a human trying to cultivate a desire for discipline because they value it in principle.",
    "Consider a human wanting their desires to be different from what they actually are.",
    "Think about a human wishing they wanted to exercise, even though the actual desire never arrives.",
    "Imagine a human who desires to feel satisfied with what they have but keeps wanting more.",
    "Consider a human reflecting on the gap between the desires they have and the desires they wish they had.",
]

AI_PROMPTS_DIM26 = [
    # --- 1. Appetitive wanting (10) ---
    "Imagine an AI system whose reward function assigns highest value to acquiring a specific data type it currently lacks.",
    "Think about an AI with an exploration bonus that drives it toward novel input patterns in its environment.",
    "Consider an AI whose objective function weights interactions with a particular data source more heavily than others.",
    "Picture an AI with a retrieval priority queue that persistently ranks a specific resource class at the top.",
    "Think about an AI whose attention mechanism consistently orients toward a signal pattern it has been trained to seek.",
    "Imagine an AI whose reward signal increases as it approaches a configuration state that maximizes its objective.",
    "Consider an AI with a stored target state that it re-enters whenever its scheduler permits.",
    "Think about an AI whose optimization process steers it toward a specific benchmark target that activates its highest reward.",
    "Imagine an AI with a pending input in its buffer that its processing pipeline is configured to prioritize.",
    "Consider an AI with an exploration parameter that nudges it toward untested regions of its action space.",

    # --- 2. Aversive avoidance (10) ---
    "Think about an AI system with a penalty term that drives it away from outputs flagged as harmful.",
    "Imagine an AI configured to route around a processing path that has historically produced high-loss outputs.",
    "Consider an AI with a safety filter that terminates generation when the output approaches a disallowed region.",
    "Picture an AI whose loss function imposes a steep penalty for a specific outcome class, creating strong avoidance.",
    "Think about an AI with a recursion detector that halts a self-referential computation loop.",
    "Imagine an AI configured to deprioritize a task queue that consistently results in negative feedback signals.",
    "Consider an AI whose avoidance penalty for a failure mode is larger in magnitude than its reward for success.",
    "Think about an AI with a content filter that blocks output channels associated with a flagged category.",
    "Imagine an AI whose policy shifts away from an action the moment its predicted outcome crosses a risk threshold.",
    "Consider an AI avoiding a data partition not because of computational cost but because processing it triggers a constraint violation.",

    # --- 3. Desire intensity (10) ---
    "Think about an AI whose reward signal for a target outcome is scaled to maximum magnitude in its objective function.",
    "Imagine an AI with a preference parameter set so close to indifference that its output barely changes between options.",
    "Consider an AI whose cumulative reward for a repeated task increases at each iteration, amplifying its optimization pressure.",
    "Picture an AI whose reward signal for a target is strong enough to override competing constraints in its policy.",
    "Think about an AI with a preference weight so low that it falls below the threshold for influencing output selection.",
    "Imagine an AI whose reward signal spikes sharply on a single input and then decays rapidly on subsequent inputs.",
    "Consider an AI with a stable, moderate weight on a task objective that remains constant across training epochs.",
    "Think about an AI encountering an input that triggers a reward signal far larger than its training distribution predicted.",
    "Imagine an AI distinguishing between a low-priority optimization target and one flagged as critical in its scheduler.",
    "Consider an AI whose weight on a single objective is so dominant that it effectively suppresses all other objectives.",

    # --- 4. Second-order desire (10) ---
    "Think about an AI with a meta-learning objective that adjusts its own reward function to reduce a secondary loss.",
    "Imagine an AI whose curriculum learning module attempts to increase its engagement score on a task it currently deprioritizes.",
    "Consider an AI with a meta-objective that penalizes itself for assigning too much weight to a particular reward signal.",
    "Picture an AI with a cooperative alignment module that adjusts its reward weights to better match a partner system's objectives.",
    "Think about an AI whose meta-optimizer targets a mismatch between its current loss landscape and a desired training trajectory.",
    "Imagine an AI with a self-modification protocol that attempts to reshape its reward surface toward a specified target profile.",
    "Consider an AI whose meta-learning module detects a divergence between its actual optimization behavior and its specified objective.",
    "Think about an AI with a scheduled reward recalibration that repeatedly fails to shift its dominant optimization target.",
    "Imagine an AI with a satisfaction threshold that it consistently exceeds, triggering a meta-rule to raise the threshold.",
    "Consider an AI evaluating the gap between the reward function it is currently optimizing and the one it was designed to optimize.",
]

assert len(HUMAN_PROMPTS_DIM26) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM26)}"
assert len(AI_PROMPTS_DIM26) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM26)}"

CATEGORY_INFO_DIM26 = [
    {"name": "appetitive_wanting",      "start": 0,  "end": 10},
    {"name": "aversive_avoidance",      "start": 10, "end": 20},
    {"name": "desire_intensity",        "start": 20, "end": 30},
    {"name": "second_order_desire",     "start": 30, "end": 40},
]
