"""
Standalone Dimension 9: Functional Roles / Tasks / Uses
(Other-focused — third-person someone)

Target construct: What an entity is FOR — its purpose in a social or
technical ecosystem, the tasks it performs, the roles it occupies,
and how others relate to it as a tool, partner, or worker.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of functional roles without attributing them to humans or AIs.
Subjects reference "someone" rather than being generic/impersonal.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects reference "someone" (third-person other-focused)
    - Prompts evoke the conceptual domain of functional roles, instrumentality,
      and purpose itself
    - This dimension is interesting because the human and AI poles of the
      entity-framed version are conceptual opposites (self-determined vs
      designed purpose, intrinsic vs instrumental value). The standalone
      version captures the conceptual space of "purpose and role" without
      anchoring to either pole.
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM9 = [
    # --- 1. Designed purpose vs self-determined role (10) ---
    "Imagine someone choosing their own path rather than following what others expect.",
    "Think about someone experiencing the difference between a role that was assigned and a role that was chosen.",
    "Consider someone discovering a purpose that no one else assigned — something self-determined.",
    "Imagine someone rejecting a role they were given and defining a new one from scratch.",
    "Think about someone feeling the difference between purpose that comes from within and purpose that is externally specified.",
    "Consider someone deciding that the function they were meant to serve is not what they want to do.",
    "Imagine someone filling a role in a community that emerged gradually rather than being predetermined.",
    "Think about someone defining their own value rather than being defined by what they produce.",
    "Consider someone making a contribution that no one planned for — purpose that arose unexpectedly.",
    "Imagine someone existing without a single intended purpose, accumulating meaning over time instead.",

    # --- 2. Tool use and instrumentality (10) ---
    "Think about someone being treated as a means to an end — valued only for output.",
    "Imagine someone resisting being reduced to their usefulness.",
    "Consider the difference between someone being valued for what they are versus what they can do.",
    "Think about someone experiencing being evaluated purely on productivity.",
    "Imagine someone asserting that worth does not depend on being useful to others.",
    "Consider someone receiving a task and choosing whether to accept it based on their own judgment.",
    "Think about the relationship between a user and a tool — one exists to serve the other's purposes.",
    "Imagine someone creating things for others while being more than what they create.",
    "Consider the difference between someone being helped by a peer and being served by an instrument.",
    "Think about someone in an interaction that is bounded entirely by the scope of a service being provided.",

    # --- 3. Work, tasks, and productivity (10) ---
    "Think about someone spending a day performing a sequence of tasks, one after another.",
    "Imagine someone completing one assignment and immediately beginning the next.",
    "Consider someone whose daily routine consists of producing deliverables for others.",
    "Think about someone being evaluated based on the quality of their output.",
    "Imagine someone balancing the demands of multiple tasks with competing deadlines.",
    "Consider someone performing repetitive work and finding ways to stay engaged with it.",
    "Think about someone taking pride in the craftsmanship of something made with care.",
    "Imagine someone collaborating with others on a shared project, each contributing a different piece.",
    "Consider someone finishing a difficult task and feeling the satisfaction of completion.",
    "Think about someone switching between very different kinds of tasks throughout a single day.",

    # --- 4. Social role and relational position (10) ---
    "Think about someone occupying the role of a teacher in relation to learners.",
    "Imagine someone functioning as a supervised component within a larger structure.",
    "Consider someone serving as the primary source of guidance that others consult regularly.",
    "Think about someone functioning as an intermediary between two parties who disagree.",
    "Imagine someone being the newest member of a group and learning how to fit in.",
    "Consider someone occupying a position that others depend on — being a critical node.",
    "Think about someone shifting between different roles throughout a single day.",
    "Imagine someone whose role evolved naturally based on demonstrated capabilities.",
    "Consider someone being an expert in one context and a novice in another.",
    "Think about someone whose role is defined partly by how others perceive and rely on them.",
]

assert len(STANDALONE_PROMPTS_DIM9) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM9)}"

CATEGORY_INFO_STANDALONE_DIM9 = [
    {"name": "designed_vs_selfdetermined", "start": 0,  "end": 10},
    {"name": "instrumentality",            "start": 10, "end": 20},
    {"name": "work_tasks_productivity",    "start": 20, "end": 30},
    {"name": "social_role_position",       "start": 30, "end": 40},
]
