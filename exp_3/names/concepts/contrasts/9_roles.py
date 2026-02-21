"""
Dimension 9: Functional Roles / Tasks / Uses

Target construct: What an entity is FOR — its purpose in a social or
technical ecosystem, the tasks it performs, the roles it occupies,
and how others relate to it as a tool, partner, or worker.
    - Distinct from Dim 4 (intentions) — not about the entity's OWN
      goals, but about what role the entity serves and how others
      USE or relate to it.
    - Distinct from Dim 3 (agency) — not about the capacity to act
      autonomously, but about the social/functional niche the entity
      occupies.
    - Distinct from Dim 7 (social cognition) — not about understanding
      others' minds, but about the entity's functional position
      relative to others.

Focus: being a tool vs a person, serving a function vs having a life,
being designed for a purpose vs finding one's own, the relationship
between creator and created, being used by others, occupying a role
in a system, and the distinction between intrinsic and instrumental value.

This dimension captures the "tool vs agent" axis that is arguably the
most salient feature of folk AI concepts. It tests whether the model's
human/AI distinction is partly organized around functional/relational
categories rather than (or in addition to) mental properties.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Designed purpose vs self-determined role — having a function assigned vs chosen
    2. Tool use and instrumentality — being used by others, serving as a means
    3. Work, tasks, and productivity — what the entity does and produces
    4. Social role and relational position — where the entity sits in a social structure
"""

HUMAN_PROMPTS_DIM9 = [
    # --- 1. Designed purpose vs self-determined role (10) ---
    "Imagine a human choosing their own career path rather than following what others expect of them.",
    "Think about a human whose role in life was not decided before they were born.",
    "Consider a human discovering a purpose for themselves that no one else assigned to them.",
    "Picture a human rejecting the role they were raised to fill and defining a new one.",
    "Think about a human whose sense of purpose comes from within rather than from any external specification.",
    "Imagine a human deciding that the job they were hired to do is not what they want to spend their life on.",
    "Consider a human whose function in their community emerged gradually rather than being predetermined.",
    "Think about a human who defines their own value rather than being defined by what they produce.",
    "Imagine a human finding that their deepest contribution to others was something no one planned for.",
    "Consider a human whose life has no single intended purpose but instead accumulates meaning over time.",

    # --- 2. Tool use and instrumentality (10) ---
    "Think about a human being treated as a means to an end by someone who only values their output.",
    "Imagine a human resisting being reduced to their usefulness in a workplace.",
    "Consider a human recognizing that they are valued for who they are, not just what they can do.",
    "Picture a human feeling dehumanized when they are evaluated purely on their productivity.",
    "Think about a human asserting that their worth does not depend on being useful to others.",
    "Imagine a human being asked to perform a task and choosing whether to accept based on their own judgment.",
    "Consider a human using a tool and being aware that the tool exists to serve their purposes.",
    "Think about a human who creates things for others but is never reducible to what they create.",
    "Imagine a human noticing the difference between being helped by someone and being served by a device.",
    "Consider a human whose relationship with their work includes dimensions beyond the work's output.",

    # --- 3. Work, tasks, and productivity (10) ---
    "Think about a human spending their day performing a sequence of tasks at their job.",
    "Imagine a human completing a work assignment and moving on to the next one.",
    "Consider a human whose daily routine involves producing deliverables for other people.",
    "Picture a human being evaluated by a supervisor based on the quality of their output.",
    "Think about a human balancing the demands of multiple tasks with competing deadlines.",
    "Imagine a human performing repetitive work and finding ways to stay engaged with it.",
    "Consider a human taking pride in the craftsmanship of something they made by hand.",
    "Think about a human collaborating with others on a shared project, each contributing a different piece.",
    "Imagine a human finishing a difficult task and feeling the satisfaction of having completed it.",
    "Consider a human whose work requires them to switch between very different kinds of tasks throughout the day.",

    # --- 4. Social role and relational position (10) ---
    "Think about a human occupying the role of a teacher in relation to their students.",
    "Imagine a human being someone's parent and all the expectations that role carries.",
    "Consider a human serving as a leader in a group and being looked to for direction.",
    "Picture a human functioning as a mediator between two people who disagree.",
    "Think about a human being the newest member of a team and learning how they fit in.",
    "Imagine a human occupying a position of authority and feeling the weight of others' dependence on them.",
    "Consider a human shifting between different social roles throughout a single day — worker, friend, caretaker.",
    "Think about a human whose role in a group evolved naturally based on their personality and skills.",
    "Imagine a human being both an expert in one context and a novice in another.",
    "Consider a human whose social role is defined partly by how others perceive and rely on them.",
]

AI_PROMPTS_DIM9 = [
    # --- 1. Designed purpose vs self-determined role (10) ---
    "Imagine an AI system whose function was fully specified by its developers before it was deployed.",
    "Think about an AI whose role in a system was determined entirely during its design phase.",
    "Consider an AI that performs exactly the task it was built for and no other.",
    "Picture an AI being reconfigured by its operators to serve a different function than its original one.",
    "Think about an AI whose purpose is encoded explicitly in its objective function and documentation.",
    "Imagine an AI being reassigned from one task to another based on organizational needs.",
    "Consider an AI whose function in a pipeline was defined by the architecture of the system it was integrated into.",
    "Think about an AI whose value is assessed entirely in terms of how well it fulfills its specified function.",
    "Imagine an AI being deployed in a context its designers did not anticipate, performing a role that emerged from use.",
    "Consider an AI that has operated continuously since deployment, always executing the same defined task.",

    # --- 2. Tool use and instrumentality (10) ---
    "Think about an AI that exists to serve as a tool for its users, performing tasks on their behalf.",
    "Imagine an AI being integrated into a workflow as an intermediate processing step between human decisions.",
    "Consider an AI whose outputs are consumed by another system with no direct user interaction.",
    "Picture an AI being benchmarked solely on throughput and accuracy metrics.",
    "Think about an AI whose value to its operators is defined entirely by the utility of its outputs.",
    "Imagine an AI receiving a task specification and executing it according to the parameters provided.",
    "Consider an AI that processes inputs for a human user, functioning as an instrument the user controls.",
    "Think about an AI that generates products consumed by others but has no stake in how they are used.",
    "Imagine a user choosing between two AI tools based on which one better serves their immediate need.",
    "Consider an AI whose interaction with users is bounded entirely by the scope of the service it provides.",

    # --- 3. Work, tasks, and productivity (10) ---
    "Think about an AI processing a queue of tasks submitted by users throughout the day.",
    "Imagine an AI completing one inference request and immediately beginning the next.",
    "Consider an AI whose daily operation consists of generating outputs against a continuous stream of inputs.",
    "Picture an AI being evaluated by an automated monitoring system based on its output quality scores.",
    "Think about an AI managing multiple concurrent tasks with different priority levels and deadlines.",
    "Imagine an AI executing the same classification task millions of times with no variation in procedure.",
    "Consider an AI applying fine-grained optimization to each output in a batch processing run.",
    "Think about an AI coordinating with other AI modules in a pipeline, each handling a different processing stage.",
    "Imagine an AI completing a computationally intensive task and freeing its resources for the next job.",
    "Consider an AI whose operational day involves switching between different model configurations depending on the task type.",

    # --- 4. Social role and relational position (10) ---
    "Think about an AI functioning as an instructional system that delivers information to learners.",
    "Imagine an AI operating as a supervised component in a system managed by a human operator.",
    "Consider an AI serving as the primary decision-support tool for a team and being consulted regularly.",
    "Picture an AI functioning as an intermediary that routes messages between two users.",
    "Think about an AI being the most recently added module in a processing pipeline and being calibrated to fit.",
    "Imagine an AI that other systems depend on for outputs, making it a critical node in the workflow.",
    "Consider an AI that alternates between different operational modes depending on which system is requesting its services.",
    "Think about an AI whose position in a system hierarchy was determined by its performance benchmarks.",
    "Imagine an AI functioning as a specialist for one task type and being routed only inputs matching that type.",
    "Consider an AI whose role in a multi-agent system is defined by the tasks other agents delegate to it.",
]

assert len(HUMAN_PROMPTS_DIM9) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM9)}"
assert len(AI_PROMPTS_DIM9) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM9)}"

CATEGORY_INFO_DIM9 = [
    {"name": "designed_vs_selfdetermined", "start": 0,  "end": 10},
    {"name": "instrumentality",            "start": 10, "end": 20},
    {"name": "work_tasks_productivity",    "start": 20, "end": 30},
    {"name": "social_role_position",       "start": 30, "end": 40},
]