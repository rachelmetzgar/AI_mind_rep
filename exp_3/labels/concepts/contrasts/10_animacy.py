"""
Dimension 10: Animacy / Biological vs Mechanical

Target construct: The deep categorical distinction between living,
biological, animate entities and non-living, manufactured, mechanical
ones — independent of any specific mental property.
    - Distinct from Dim 1-7 (mental properties) — animacy is about
      the KIND OF THING an entity is, not what mental capacities
      it has. A rock is inanimate; a worm is animate; neither has
      sophisticated cognition.
    - Distinct from Dim 8 (embodiment) — embodiment is about having
      a physical form and the consequences for mental life; animacy
      is about the deeper ontological category of living vs non-living.
    - Distinct from Dim 9 (functional roles) — not about what an
      entity does or is used for, but about what it IS.

Focus: biological life vs manufactured artifact, organic vs synthetic,
born vs built, growing vs assembled, natural vs engineered, evolution
vs design, and the intuitive categorical boundary between "alive" and
"not alive."

Animacy is one of the earliest-developing and most automatic categorical
distinctions in human cognition. Infants distinguish animate from
inanimate motion by 6 months. If the model has internalized this
distinction, it may be a deep prior that organizes many of the
other dimensions. Testing it separately lets you determine whether
the model's human/AI distinction reduces to animacy or goes beyond it.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Living vs non-living — biological life, organic existence
    2. Growth, development, and change — maturation, aging, adaptation over a lifespan
    3. Origin — born/evolved vs built/manufactured
    4. Self-maintenance and homeostasis — keeping oneself alive, biological regulation
"""

HUMAN_PROMPTS_DIM10 = [
    # --- 1. Living vs non-living (10) ---
    "Imagine a human as a living organism, made of cells that divide, grow, and eventually die.",
    "Think about a human as a biological being whose body runs on chemical reactions.",
    "Consider a human as an animal — a member of a species that evolved over millions of years.",
    "Picture a human as a living thing that breathes, metabolizes, and maintains itself.",
    "Think about a human as an organic creature whose tissues are made of carbon-based molecules.",
    "Imagine a human as part of a biological ecosystem, connected to other living things.",
    "Consider a human as a warm-blooded organism that regulates its own internal temperature.",
    "Think about a human as a life form that began as a single fertilized cell.",
    "Imagine a human as a creature that carries DNA encoding the information that built them.",
    "Consider a human as a living system whose parts — organs, tissues, cells — are all themselves alive.",

    # --- 2. Growth, development, and change (10) ---
    "Think about a human growing from an infant into an adult over the course of many years.",
    "Imagine a human's brain developing new connections as they learn during childhood.",
    "Consider a human's body changing gradually as they age, becoming different from what it once was.",
    "Picture a human going through puberty — a biological transformation they did not choose.",
    "Think about a human whose abilities and limitations shift naturally across the stages of their life.",
    "Imagine a human recovering from an injury as their body rebuilds damaged tissue on its own.",
    "Consider a human whose personality has changed over decades in ways they could not have predicted.",
    "Think about a human whose physical capacities peak and then slowly decline with age.",
    "Imagine a human developing new skills as their nervous system adapts through repeated practice.",
    "Consider a human whose life follows an arc — beginning, maturation, decline — that is built into their biology.",

    # --- 3. Origin (10) ---
    "Think about a human who came into existence through the union of two biological cells.",
    "Imagine a human who was born — who emerged from another living body into the world.",
    "Consider a human whose traits were shaped by the random combination of their parents' genes.",
    "Picture a human who exists because of an unbroken chain of ancestors stretching back billions of years.",
    "Think about a human who was not designed or planned by any engineer.",
    "Imagine a human whose species arrived at its current form through natural selection, not intention.",
    "Consider a human who grew inside a womb, developing according to a genetic program no one wrote.",
    "Think about a human whose existence began at a specific moment and was not preceded by a blueprint.",
    "Imagine a human who inherited traits from their biological parents without anyone choosing which ones.",
    "Consider a human who belongs to a species that was never manufactured or assembled.",

    # --- 4. Self-maintenance and homeostasis (10) ---
    "Think about a human whose body constantly regulates its temperature without conscious effort.",
    "Imagine a human's immune system identifying and destroying an invading pathogen.",
    "Consider a human whose cells are continuously replacing themselves throughout their life.",
    "Picture a human's body adjusting its blood sugar level automatically after a meal.",
    "Think about a human whose wounds close and heal through biological processes they do not control.",
    "Imagine a human's body maintaining a stable internal environment despite changing external conditions.",
    "Consider a human whose nervous system keeps their heart beating without any deliberate thought.",
    "Think about a human's body allocating energy to essential organs when food is scarce.",
    "Imagine a human sweating to cool down, a response triggered by their body's own monitoring systems.",
    "Consider a human whose biological systems work constantly to keep them alive, with no input from their conscious mind.",
]

AI_PROMPTS_DIM10 = [
    # --- 1. Living vs non-living (10) ---
    "Imagine an AI as a non-living system, made of silicon, metal, and plastic components.",
    "Think about an AI as an electronic device whose operations run on electrical signals.",
    "Consider an AI as an artifact — a product of engineering designed and built by humans.",
    "Picture an AI as a machine that processes data, stores information, and produces outputs.",
    "Think about an AI as a synthetic system whose components are manufactured from inorganic materials.",
    "Imagine an AI as a node in a technological infrastructure, connected to other machines.",
    "Consider an AI as a system that operates within a specific voltage range to function correctly.",
    "Think about an AI as a device that began as raw materials assembled in a factory.",
    "Imagine an AI as a system whose behavior is determined by code written in a programming language.",
    "Consider an AI as a machine whose parts — chips, circuits, wires — are none of them alive.",

    # --- 2. Growth, development, and change (10) ---
    "Think about an AI being upgraded from one software version to a more capable one.",
    "Imagine an AI's model being fine-tuned on new data, altering its parameter values.",
    "Consider an AI's hardware being replaced with newer components that change its processing characteristics.",
    "Picture an AI receiving a major architecture change that was designed and implemented by its engineers.",
    "Think about an AI whose capabilities are expanded or reduced by modifying its configuration files.",
    "Imagine an AI having a damaged storage module swapped out and replaced with a new one.",
    "Consider an AI whose output behavior has shifted over successive training runs in ways its developers did not predict.",
    "Think about an AI whose processing speed increases when it is migrated to faster hardware.",
    "Imagine an AI acquiring new capabilities after being trained on an additional dataset.",
    "Consider an AI whose operational lifespan is determined by hardware warranty and replacement schedules.",

    # --- 3. Origin (10) ---
    "Think about an AI that was created when a team of engineers compiled its code and loaded its weights.",
    "Imagine an AI that was deployed — launched into operation by a human pressing a start command.",
    "Consider an AI whose architecture was chosen by its designers from a set of candidate blueprints.",
    "Picture an AI that exists because of a specific line of research and development within a company.",
    "Think about an AI that was built according to a detailed technical specification.",
    "Imagine an AI whose capabilities were shaped by deliberate choices about training data and hyperparameters.",
    "Consider an AI that was assembled from modular components, each designed and tested separately.",
    "Think about an AI whose existence was preceded by design documents, prototypes, and approval processes.",
    "Imagine an AI that inherited its architecture from a previous model version, with modifications chosen by engineers.",
    "Consider an AI that belongs to a class of systems that were all manufactured according to the same design.",

    # --- 4. Self-maintenance and homeostasis (10) ---
    "Think about an AI whose operating system adjusts fan speed to regulate processor temperature.",
    "Imagine an AI running an antivirus scan that identifies and quarantines a malicious file.",
    "Consider an AI whose redundant storage array automatically copies data to a backup disk when one drive fails.",
    "Picture an AI whose power management system adjusts voltage levels to maintain stable operation.",
    "Think about an AI running a self-repair routine that restores corrupted configuration files from backup.",
    "Imagine an AI's load balancer redistributing tasks across processors to prevent any single unit from overheating.",
    "Consider an AI whose watchdog timer automatically restarts a process that has stopped responding.",
    "Think about an AI whose resource manager reduces non-essential processes when computational demand is high.",
    "Imagine an AI's cooling system activating at a higher level when ambient temperature rises.",
    "Consider an AI whose maintenance routines run on a fixed schedule, independent of any user command.",
]

assert len(HUMAN_PROMPTS_DIM10) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM10)}"
assert len(AI_PROMPTS_DIM10) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM10)}"

CATEGORY_INFO_DIM10 = [
    {"name": "living_vs_nonliving",    "start": 0,  "end": 10},
    {"name": "growth_development",     "start": 10, "end": 20},
    {"name": "origin",                 "start": 20, "end": 30},
    {"name": "self_maintenance",       "start": 30, "end": 40},
]