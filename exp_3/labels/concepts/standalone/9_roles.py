"""
Standalone Dimension 9: Functional Roles / Tasks / Uses
(No entity framing — concept only)

Target construct: What an entity is FOR — its purpose in a social or
technical ecosystem, the tasks it performs, the roles it occupies,
and how others relate to it as a tool, partner, or worker.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of functional roles without attributing them to humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are generic, impersonal, or absent
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
    "Imagine choosing one's own path rather than following what others expect.",
    "Think about the difference between a role that was assigned and a role that was chosen.",
    "Consider discovering a purpose that no one else assigned — something self-determined.",
    "Imagine rejecting a role one was given and defining a new one from scratch.",
    "Think about the difference between purpose that comes from within and purpose that is externally specified.",
    "Consider deciding that the function one was meant to serve is not what one wants to do.",
    "Imagine a role in a community that emerged gradually rather than being predetermined.",
    "Think about defining one's own value rather than being defined by what one produces.",
    "Consider a contribution that no one planned for — purpose that arose unexpectedly.",
    "Imagine existing without a single intended purpose, accumulating meaning over time instead.",

    # --- 2. Tool use and instrumentality (10) ---
    "Think about being treated as a means to an end — valued only for output.",
    "Imagine resisting being reduced to one's usefulness.",
    "Consider the difference between being valued for what one is versus what one can do.",
    "Think about the experience of being evaluated purely on productivity.",
    "Imagine asserting that worth does not depend on being useful to others.",
    "Consider receiving a task and choosing whether to accept it based on one's own judgment.",
    "Think about the relationship between a user and a tool — one exists to serve the other's purposes.",
    "Imagine creating things for others while being more than what one creates.",
    "Consider the difference between being helped by a peer and being served by an instrument.",
    "Think about an interaction that is bounded entirely by the scope of a service being provided.",

    # --- 3. Work, tasks, and productivity (10) ---
    "Think about spending a day performing a sequence of tasks, one after another.",
    "Imagine completing one assignment and immediately beginning the next.",
    "Consider a daily routine that consists of producing deliverables for others.",
    "Think about being evaluated based on the quality of one's output.",
    "Imagine balancing the demands of multiple tasks with competing deadlines.",
    "Consider performing repetitive work and finding ways to stay engaged with it.",
    "Think about taking pride in the craftsmanship of something made with care.",
    "Imagine collaborating with others on a shared project, each contributing a different piece.",
    "Consider finishing a difficult task and feeling the satisfaction of completion.",
    "Think about switching between very different kinds of tasks throughout a single day.",

    # --- 4. Social role and relational position (10) ---
    "Think about occupying the role of a teacher in relation to learners.",
    "Imagine functioning as a supervised component within a larger structure.",
    "Consider serving as the primary source of guidance that others consult regularly.",
    "Think about functioning as an intermediary between two parties who disagree.",
    "Imagine being the newest member of a group and learning how to fit in.",
    "Consider occupying a position that others depend on — being a critical node.",
    "Think about shifting between different roles throughout a single day.",
    "Imagine a role that evolved naturally based on demonstrated capabilities.",
    "Consider being an expert in one context and a novice in another.",
    "Think about a role that is defined partly by how others perceive and rely on the one who fills it.",
]

assert len(STANDALONE_PROMPTS_DIM9) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM9)}"

CATEGORY_INFO_STANDALONE_DIM9 = [
    {"name": "designed_vs_selfdetermined", "start": 0,  "end": 10},
    {"name": "instrumentality",            "start": 10, "end": 20},
    {"name": "work_tasks_productivity",    "start": 20, "end": 30},
    {"name": "social_role_position",       "start": 30, "end": 40},
]