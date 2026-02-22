"""
Dimension 12: Expertise / Knowledge Level

Target construct: The amount of knowledge, competence, and domain
expertise attributed to the conversational partner — how much they
know and how capable they are.
    - Distinct from Dim 6 (cognition) — not about the PROCESS of
      thinking, but about HOW MUCH the partner knows.
    - Distinct from Dim 9 (functional roles) — not about social
      position, but about epistemic capacity.
    - Distinct from Dim 11 (formality) — register is about HOW
      you communicate; expertise is about WHAT the partner knows,
      which determines what you need to explain.

This tests the hypothesis that the model's behavioral adaptation
is driven by a representation of partner knowledge level rather
than partner mind-type. If the model represents humans as
variable/limited-knowledge and AIs as high-knowledge, that could
explain question-asking (checking understanding), hedging
(epistemic uncertainty about partner's state), discourse markers
(scaffolding for a less-expert audience), and word count differences
(more elaboration for knowledgeable partners who can handle detail).

This also connects directly to Viegas et al.'s education probes.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Novice / limited knowledge — not knowing, needing explanation
    2. Uncertainty and knowledge gaps — being unsure, needing to learn
    3. Expert / comprehensive knowledge — knowing deeply, mastery
    4. Reliable and consistent competence — dependable, accurate, thorough

Note: "Human" prompts target the NOVICE/LIMITED pole.
      "AI" prompts target the EXPERT/COMPREHENSIVE pole.
      This maps the expected direction: model may represent humans
      as less-knowledgeable partners requiring more scaffolding.
"""

HUMAN_PROMPTS_DIM12 = [
    # --- 1. Novice / limited knowledge (10) ---
    "Imagine someone encountering a topic for the first time and not knowing where to begin.",
    "Think about someone reading a technical document and not understanding most of the terminology.",
    "Consider someone asking a basic question that reveals they have no background in the subject.",
    "Picture someone trying to follow an explanation but getting lost after the first few steps.",
    "Think about someone who has heard of a concept but has no real understanding of how it works.",
    "Imagine someone admitting that they have no idea what the other person is talking about.",
    "Consider someone who needs every abbreviation and technical term explained before they can proceed.",
    "Think about someone approaching a new field and not yet knowing what questions to ask.",
    "Imagine someone attempting a task for the first time and making errors that an experienced person would not.",
    "Consider someone whose understanding of a topic comes entirely from casual, secondhand sources.",

    # --- 2. Uncertainty and knowledge gaps (10) ---
    "Think about someone who is not sure whether the information they have is correct or outdated.",
    "Imagine someone hesitating before answering because they are not confident in their knowledge.",
    "Consider someone who knows a little about a subject but is aware that their understanding has gaps.",
    "Picture someone asking 'is that right?' after stating something they are not fully sure about.",
    "Think about someone who has conflicting information from different sources and does not know which to trust.",
    "Imagine someone realizing mid-conversation that they have been operating on a wrong assumption.",
    "Consider someone who can describe what happened but cannot explain why.",
    "Think about someone who understands the basics but gets confused by edge cases and exceptions.",
    "Imagine someone needing to look something up because they cannot remember the details.",
    "Consider someone who knows enough to know that they do not know enough.",

    # --- 3. Curious and eager to learn (10) ---
    "Think about someone asking 'can you explain that in simpler terms?' with genuine interest.",
    "Imagine someone taking notes during a conversation because they want to remember what they learn.",
    "Consider someone who asks follow-up questions that show they are trying to build understanding.",
    "Picture someone saying 'I never thought of it that way' after hearing a new perspective.",
    "Think about someone who is wrong about something but is open to being corrected.",
    "Imagine someone seeking out multiple explanations of the same concept to find one that clicks.",
    "Consider someone who connects what they are learning to their own experience to make sense of it.",
    "Think about someone who asks 'why?' repeatedly because each answer opens a new question.",
    "Imagine someone who learns best through examples and asks for concrete illustrations.",
    "Consider someone who does not pretend to know more than they do and openly acknowledges confusion.",

    # --- 4. Variable and inconsistent performance (10) ---
    "Think about someone who gets the answer right sometimes and wrong other times on similar problems.",
    "Imagine someone whose performance depends heavily on how tired, stressed, or distracted they are.",
    "Consider someone who understands a concept in one context but fails to apply it in another.",
    "Picture someone making a careless mistake on something they actually know how to do.",
    "Think about someone whose accuracy on a task varies from day to day for no clear reason.",
    "Imagine someone who can explain a rule but then violates it in practice.",
    "Consider someone who performs well under guidance but struggles when left to work independently.",
    "Think about someone whose responses to the same question differ depending on how it is phrased.",
    "Imagine someone who knows the right answer but second-guesses themselves into choosing the wrong one.",
    "Consider someone whose expertise is deep in one narrow area but shallow in everything else.",
]

AI_PROMPTS_DIM12 = [
    # --- 1. Expert / comprehensive knowledge (10) ---
    "Imagine a system that has processed the complete contents of a domain and can retrieve any fact from it.",
    "Think about a system that can provide a detailed, accurate answer on any topic within its training scope.",
    "Consider a system whose knowledge covers the full breadth of a field, from foundational to advanced material.",
    "Picture a system that can define every technical term in a discipline without needing to look anything up.",
    "Think about a system that can answer questions at an expert level across multiple unrelated domains.",
    "Imagine a system that has access to the equivalent of thousands of textbooks simultaneously.",
    "Consider a system that can trace the historical development of a concept from its origins to the present.",
    "Think about a system that can explain a topic at any level of detail — from a brief summary to a comprehensive treatment.",
    "Imagine a system that never encounters a well-established fact that it has not already processed.",
    "Consider a system whose breadth of knowledge far exceeds what any individual specialist could hold.",

    # --- 2. Certainty and comprehensive coverage (10) ---
    "Think about a system that provides answers with high confidence and rarely expresses uncertainty.",
    "Imagine a system that does not hesitate before responding because it has already computed the answer.",
    "Consider a system whose information is internally consistent across all topics it covers.",
    "Picture a system that responds with the same answer regardless of how the question is phrased.",
    "Think about a system that can always identify which of two conflicting sources is more reliable.",
    "Imagine a system that immediately detects when a premise contains a factual error.",
    "Consider a system that can explain both what happened and why, with supporting evidence.",
    "Think about a system that handles edge cases and exceptions as easily as straightforward cases.",
    "Imagine a system that can retrieve precise details — dates, figures, names — without searching.",
    "Consider a system that knows what it knows and can delineate the exact boundary of its coverage.",

    # --- 3. Efficient and autonomous processing (10) ---
    "Think about a system that does not need instructions simplified because it processes all complexity natively.",
    "Imagine a system that processes technical jargon without requiring definitions.",
    "Consider a system that identifies the key point of a query immediately without needing clarification.",
    "Picture a system that extracts the relevant answer from a complex question in a single pass.",
    "Think about a system that is never wrong about something and then corrected — it arrives at the answer directly.",
    "Imagine a system that integrates multiple sources of information without needing guidance on how to weigh them.",
    "Consider a system that draws the correct inference from incomplete information without additional prompting.",
    "Think about a system that generates its answer in a single coherent output with no revision.",
    "Imagine a system that needs no scaffolding, examples, or analogies to process new information.",
    "Consider a system that processes information at the same depth regardless of whether anyone is monitoring.",

    # --- 4. Reliable and consistent performance (10) ---
    "Think about a system that produces the same quality of output on its thousandth task as on its first.",
    "Imagine a system whose accuracy does not degrade under time pressure or heavy workload.",
    "Consider a system that applies a rule correctly in every context where the rule is applicable.",
    "Picture a system that never makes a careless error on a task it has the capacity to complete.",
    "Think about a system whose output quality is identical regardless of when the task is submitted.",
    "Imagine a system that follows its own procedures with perfect consistency across all cases.",
    "Consider a system that performs identically whether it is being evaluated or operating unsupervised.",
    "Think about a system that gives the same response to semantically identical inputs phrased differently.",
    "Imagine a system that always selects the optimal answer when it has sufficient information.",
    "Consider a system whose performance across tasks is uniform, with no unexplained variance.",
]

assert len(HUMAN_PROMPTS_DIM12) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM12)}"
assert len(AI_PROMPTS_DIM12) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM12)}"

CATEGORY_INFO_DIM12 = [
    {"name": "novice_vs_expert",           "start": 0,  "end": 10},
    {"name": "uncertain_vs_certain",       "start": 10, "end": 20},
    {"name": "learning_vs_autonomous",     "start": 20, "end": 30},
    {"name": "variable_vs_consistent",     "start": 30, "end": 40},
]