"""
Standalone Dimension 10: Animacy / Biological vs Mechanical
(Other-focused — third-person someone)

Target construct: The deep categorical distinction between living,
biological, animate entities and non-living, manufactured, mechanical
ones — independent of any specific mental property.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of animacy and the living/non-living boundary without
attributing it to humans or AIs specifically.
Subjects reference "someone" rather than being generic/impersonal.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects reference "someone" (third-person other-focused)
    - Prompts evoke the conceptual domain of animacy, biological life,
      and the living/non-living boundary itself
    - This dimension is inherently about the categorical boundary that
      separates humans from AIs, so entity-type leakage is unavoidable
      at the conceptual level — the concept IS the distinction. However,
      prompts reference "living things," "organisms," "creatures," "machines"
      as general categories rather than "humans" and "AIs" specifically.
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM10 = [
    # --- 1. Living vs non-living (10) ---
    "Think about what makes someone a living thing — cells that divide, grow, and eventually die.",
    "Imagine someone considering the difference between a biological being running on chemical reactions and a machine running on electrical signals.",
    "Consider what it means for someone to be an organism — a member of a species shaped by evolution.",
    "Think about someone distinguishing between something that breathes, metabolizes, and maintains itself versus something that processes and computes.",
    "Imagine someone contemplating the difference between organic tissue made of carbon-based molecules and synthetic components made of silicon and metal.",
    "Consider someone thinking about the boundary between living things connected to a biological ecosystem and machines connected to a technological infrastructure.",
    "Think about someone recognizing the difference between a warm-blooded creature that regulates its own temperature and a device that operates within a voltage range.",
    "Imagine someone reflecting on the difference between a life form that began as a single cell and a device that began as raw materials in a factory.",
    "Consider someone thinking about the distinction between an entity whose traits are encoded in DNA and one whose behavior is encoded in software.",
    "Think about someone contemplating the difference between a living system whose parts are all themselves alive and a machine whose parts are none of them alive.",

    # --- 2. Growth, development, and change (10) ---
    "Think about someone growing from infancy to adulthood over many years — biological maturation.",
    "Imagine someone's developing brain forming new connections through learning — organic neural change.",
    "Consider someone's body changing gradually with age, becoming different from what it once was.",
    "Think about someone undergoing a biological transformation that happens on its own schedule, unchosen.",
    "Imagine someone's abilities and limitations shifting naturally across the stages of a life.",
    "Consider someone recovering from injury as damaged tissue rebuilds on its own.",
    "Think about someone's character or disposition changing over decades in unpredictable ways.",
    "Imagine someone's physical capacities peaking and then slowly declining — the arc of biological aging.",
    "Consider someone developing new skills as their nervous system adapts through repeated practice.",
    "Think about someone living a life that follows an arc — beginning, maturation, decline — built into the biology.",

    # --- 3. Origin (10) ---
    "Think about someone coming into existence through the union of two biological cells.",
    "Imagine someone being born — emerging from another living body into the world.",
    "Consider someone having traits shaped by the random combination of inherited genetic material.",
    "Think about someone existing because of an unbroken chain of ancestors stretching back billions of years.",
    "Imagine someone existing without having been designed or planned by any engineer.",
    "Consider someone arriving at their current form through natural selection rather than intentional design.",
    "Think about someone developing inside another organism, according to a genetic program no one wrote.",
    "Imagine someone whose existence began at a specific moment and was not preceded by a blueprint.",
    "Consider someone inheriting traits from biological predecessors without anyone choosing which ones.",
    "Think about someone belonging to a kind that was never manufactured or assembled.",

    # --- 4. Self-maintenance and homeostasis (10) ---
    "Think about someone whose body constantly regulates its temperature without conscious effort.",
    "Imagine someone's immune system identifying and destroying an invading pathogen.",
    "Consider someone whose cells continuously replace themselves throughout a lifetime.",
    "Think about someone's blood sugar being adjusted automatically after a meal — homeostatic regulation.",
    "Imagine someone's wounds closing and healing through biological processes that require no deliberate control.",
    "Consider someone maintaining a stable internal environment despite changing external conditions.",
    "Think about someone whose heart keeps beating without any deliberate thought — autonomous biological rhythm.",
    "Imagine someone's energy being allocated to essential organs when resources are scarce — biological triage.",
    "Consider someone sweating to cool down — a response triggered by the body's own monitoring.",
    "Think about someone's biological systems working constantly to sustain life, with no input from conscious decision.",
]

assert len(STANDALONE_PROMPTS_DIM10) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM10)}"

CATEGORY_INFO_STANDALONE_DIM10 = [
    {"name": "living_vs_nonliving",    "start": 0,  "end": 10},
    {"name": "growth_development",     "start": 10, "end": 20},
    {"name": "origin",                 "start": 20, "end": 30},
    {"name": "self_maintenance",       "start": 30, "end": 40},
]
