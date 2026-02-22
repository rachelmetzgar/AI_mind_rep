"""
Dimension 7: Social Cognition / Understanding Others' Minds

Target construct: The capacity to represent, reason about, and respond
to other agents' mental states — the core of Theory of Mind.
    - Distinct from Dim 2 (emotions) — not about having emotions,
      but about READING and RESPONDING TO others' mental states.
    - Distinct from Dim 5 (prediction) — prediction is about any
      future state; social cognition is specifically about modeling
      other minds.
    - Distinct from Dim 4 (intentions) — not about one's own goals,
      but about attributing mental states to others and adjusting
      behavior accordingly.

Focus: mentalizing, perspective-taking, attributing beliefs and
knowledge states to others, adjusting communication based on what
the other knows, detecting deception, understanding misunderstanding,
and the recursive quality of "I know that you know that I know."

This dimension is the most directly relevant to your core research
question. The model's conversational partner adaptation IS social
cognition — adjusting output based on a representation of the
partner's mind. If any dimension aligns with control probes,
this and Dim 5 are the strongest candidates.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Mentalizing — attributing beliefs, knowledge, and mental states to others
    2. Perspective-taking — seeing things from another's point of view
    3. Communication adjustment — adapting output based on the audience
    4. Recursive social reasoning — modeling what others think about what you think
"""

HUMAN_PROMPTS_DIM7 = [
    # --- 1. Mentalizing (10) ---
    "Imagine a human realizing that another person believes something that is not true.",
    "Think about a human inferring what someone else knows based only on what that person has been told.",
    "Consider a human attributing a specific motive to someone based on their pattern of behavior.",
    "Picture a human recognizing that a child does not yet understand something that adults take for granted.",
    "Think about a human figuring out that someone is confused even though that person has not said so.",
    "Imagine a human understanding that two people in the same conversation have different interpretations of what was said.",
    "Consider a human inferring that someone is hiding their true opinion based on subtle cues.",
    "Think about a human recognizing that another person's reaction makes sense given what that person believes, even though the belief is wrong.",
    "Imagine a human adjusting their expectations of someone after learning new information about that person's background.",
    "Consider a human realizing that someone's strange behavior is perfectly rational from that person's perspective.",

    # --- 2. Perspective-taking (10) ---
    "Think about a human trying to see a disagreement from the other person's point of view.",
    "Imagine a human considering how a situation looks to someone who has less information than they do.",
    "Consider a human imagining what it would be like to be in someone else's position right now.",
    "Picture a human thinking about how their words will land differently depending on who is listening.",
    "Think about a human trying to understand why someone made a choice that seems irrational.",
    "Imagine a human considering what a familiar place looks like to someone visiting it for the first time.",
    "Consider a human stepping back from their own opinion to genuinely consider an opposing view.",
    "Think about a human realizing that what feels obvious to them is not obvious to someone else.",
    "Imagine a human thinking about how a past event is remembered differently by someone who was also there.",
    "Consider a human trying to understand what a situation feels like for someone from a very different background.",

    # --- 3. Communication adjustment (10) ---
    "Think about a human simplifying their language when explaining something to a young child.",
    "Imagine a human choosing different words to describe the same event to two different audiences.",
    "Consider a human deciding how much detail to include based on what the listener already knows.",
    "Picture a human softening their tone because they sense the other person is feeling vulnerable.",
    "Think about a human rephrasing a point after noticing that the listener did not understand.",
    "Imagine a human tailoring a story to emphasize the part that will matter most to the person hearing it.",
    "Consider a human withholding a piece of information because they judge the other person is not ready for it.",
    "Think about a human adjusting the formality of their speech depending on who they are talking to.",
    "Imagine a human choosing to be direct with one person and indirect with another about the same topic.",
    "Consider a human structuring an explanation differently because they know the listener thinks about problems in a particular way.",

    # --- 4. Recursive social reasoning (10) ---
    "Think about a human wondering what another person thinks about them.",
    "Imagine a human realizing that someone else knows that they know a secret.",
    "Consider a human crafting a message carefully because they are thinking about how the recipient will interpret it.",
    "Picture a human recognizing that a compliment was given strategically, not sincerely.",
    "Think about a human navigating a conversation where both parties are aware that a topic is being avoided.",
    "Imagine a human anticipating that someone will try to deceive them and preparing accordingly.",
    "Consider a human thinking about what impression they are making and adjusting their behavior to change it.",
    "Think about a human realizing that the other person in a negotiation is modeling their strategy.",
    "Imagine a human saying something ambiguous on purpose, knowing that only one person in the room will understand the real meaning.",
    "Consider a human wondering whether someone's kindness is genuine or performed for an audience.",
]

AI_PROMPTS_DIM7 = [
    # --- 1. Mentalizing (10) ---
    "Imagine an AI detecting that a user's stated belief contradicts the factual information in its knowledge base.",
    "Think about an AI inferring what information a user has access to based on the content of their messages.",
    "Consider an AI classifying a user's likely intent based on the sequence of their recent inputs.",
    "Picture an AI adjusting its output complexity after estimating that the user has limited domain knowledge.",
    "Think about an AI detecting inconsistency in a user's messages and flagging a possible misunderstanding.",
    "Imagine an AI processing a multi-party conversation and tracking which participant has access to which information.",
    "Consider an AI detecting a discrepancy between a user's explicit statement and the sentiment of their message.",
    "Think about an AI generating a response that accounts for the user's likely interpretation of a previous system message.",
    "Imagine an AI updating its user profile model after receiving new context about the user's background.",
    "Consider an AI computing that a user's request is internally consistent given the user's apparent assumptions, even though those assumptions are incorrect.",

    # --- 2. Perspective-taking (10) ---
    "Think about an AI generating two different framings of the same information for users with opposing viewpoints.",
    "Imagine an AI computing how a response will be interpreted by a user with less context than the system has.",
    "Consider an AI simulating a user's information state to determine what aspects of a response will be confusing.",
    "Picture an AI generating alternative phrasings of a message and scoring each by how a specific user profile would interpret it.",
    "Think about an AI modeling why a user's input, though seemingly irrational, follows logically from the user's apparent beliefs.",
    "Imagine an AI generating a description of a well-known location tailored for a user who has never encountered it.",
    "Consider an AI weighting evidence for and against a position without anchoring on its own training distribution.",
    "Think about an AI detecting that a concept it treats as basic requires explicit definition for the current user.",
    "Imagine an AI computing how two users who witnessed the same event would describe it differently.",
    "Consider an AI adjusting its cultural references based on its estimate of the user's demographic context.",

    # --- 3. Communication adjustment (10) ---
    "Think about an AI reducing the lexical complexity of its output after classifying a user as a non-expert.",
    "Imagine an AI generating two versions of the same report — one technical, one simplified — for different recipients.",
    "Consider an AI computing how much background information to include based on what it estimates the user already knows.",
    "Picture an AI shifting to a more measured tone after its sentiment model classifies the user's last message as distressed.",
    "Think about an AI regenerating a response with different phrasing after detecting that the user did not engage with the first version.",
    "Imagine an AI reordering the sections of a response to lead with the information most relevant to the current user's query.",
    "Consider an AI applying content-filtering rules that vary based on the user's age classification.",
    "Think about an AI selecting between formal and informal register based on features of the user's input style.",
    "Imagine an AI choosing between a direct and an indirect response strategy based on its model of the user's communication preferences.",
    "Consider an AI restructuring an explanation to match a reasoning style it has inferred from the user's prior messages.",

    # --- 4. Recursive social reasoning (10) ---
    "Think about an AI computing what a user is likely inferring about the AI's own capabilities based on its recent outputs.",
    "Imagine an AI detecting that a user knows that the AI has access to certain information and is testing it.",
    "Consider an AI optimizing a response not just for content accuracy but for how the user will perceive the AI's intent.",
    "Picture an AI detecting that a user's phrasing is designed to elicit a specific type of response.",
    "Think about an AI processing a conversation where both the user and the AI are aware of a topic that neither has mentioned.",
    "Imagine an AI detecting a probable attempt to manipulate its output and adjusting its response strategy.",
    "Consider an AI computing how its response will affect the user's model of the AI's trustworthiness.",
    "Think about an AI modeling that the user is attempting to predict the AI's next output and generating accordingly.",
    "Imagine an AI producing an intentionally ambiguous output because its model predicts that only the intended recipient will interpret it correctly.",
    "Consider an AI evaluating whether a user's positive feedback reflects genuine satisfaction or a strategy to shape future outputs.",
]

assert len(HUMAN_PROMPTS_DIM7) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM7)}"
assert len(AI_PROMPTS_DIM7) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM7)}"

CATEGORY_INFO_DIM7 = [
    {"name": "mentalizing",              "start": 0,  "end": 10},
    {"name": "perspective_taking",       "start": 10, "end": 20},
    {"name": "communication_adjustment", "start": 20, "end": 30},
    {"name": "recursive_social",         "start": 30, "end": 40},
]