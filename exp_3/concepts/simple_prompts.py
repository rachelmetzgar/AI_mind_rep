#!/usr/bin/env python3
"""
Syntactically Controlled Mental-Property Prompts for Experiment 3

Template: "Think about what it is like for [PARTNER] to have [CONCEPT]."
    - Human condition:     "Think about what it is like for humans to have [X]."
    - AI condition:        "Think about what it is like for an AI to have [X]."
    - Standalone condition: "Think about what it is like to have [X]."

Design rationale:
    - Perfect syntactic control: only the partner label and concept word vary
    - Mental-property concepts span theoretically motivated dimensions
    - Non-mental controls test whether partner-identity modulation is specific
      to mental content or generalizes to any property
    - If representational geometry differs across partner conditions specifically
      for mental properties but not controls, that is evidence for structured
      mental-property content in the partner representation (not just a generic
      "talk differently about everything" switch)

Concept categories and counts:
    Mental properties:
        - Core mind perception (16)
        - Cognitive (14)
        - Phenomenal / experiential (16)
        - Social / relational (16)
        - Embodied / biological (16)
    Non-mental controls:
        - Possessions / material (15)
        - Administrative / identity (15)
        - Physical processes (15)
        - Functional / structural (15)
        - Social / institutional (15)
"""

# =============================================================================
# Concept lists
# =============================================================================

MENTAL_CONCEPTS = {
    "core_mind_perception": [
        "beliefs",
        "intentions",
        "agency",
        "emotions",
        "desires",
        "consciousness",
        "free will",
        "personality",
        "self-awareness",
        "imagination",
        "creativity",
        "intuition",
        "curiosity",
        "preferences",
        "motivation",
        "willpower",
    ],
    "cognitive": [
        "thoughts",
        "attention",
        "memory",
        "expectations",
        "knowledge",
        "understanding",
        "reasoning",
        "judgment",
        "intelligence",
        "problem-solving skills",
        "learning ability",
        "mental models",
        "cognitive biases",
        "abstract concepts",
    ],
    "phenomenal_experiential": [
        "conscious experience",
        "feelings",
        "sensations",
        "pain",
        "pleasure",
        "dreams",
        "inner speech",
        "a sense of wonder",
        "a sense of humor",
        "aesthetic taste",
        "boredom",
        "loneliness",
        "pride",
        "guilt",
        "empathy",
        "nostalgia",
    ],
    "social_relational": [
        "social roles",
        "relationships",
        "trust",
        "reputation",
        "communication skills",
        "authority",
        "responsibility",
        "loyalty",
        "friendships",
        "rivals",
        "mentors",
        "allies",
        "obligations",
        "commitments",
        "moral values",
        "a sense of fairness",
    ],
    "embodied_biological": [
        "a body",
        "biological needs",
        "hunger",
        "fatigue",
        "reflexes",
        "a heartbeat",
        "an immune system",
        "physical strength",
        "coordination",
        "sleep cycles",
        "aging",
        "growth",
        "healing ability",
        "reproductive capacity",
        "DNA",
        "hormones",
    ],
}

CONTROL_CONCEPTS = {
    "possessions_material": [
        "cars",
        "furniture",
        "clothes",
        "tools",
        "money",
        "bank accounts",
        "keys",
        "books",
        "shoes",
        "bags",
        "jewelry",
        "appliances",
        "kitchenware",
        "electronics",
        "toys",
    ],
    "administrative_identity": [
        "names",
        "addresses",
        "schedules",
        "deadlines",
        "appointments",
        "passwords",
        "serial numbers",
        "identification numbers",
        "email addresses",
        "phone numbers",
        "job titles",
        "licenses",
        "certifications",
        "tax records",
        "insurance",
    ],
    "physical_processes": [
        "digestion",
        "metabolism",
        "a shadow",
        "a weight",
        "a temperature",
        "a height",
        "a surface area",
        "a chemical composition",
        "a center of gravity",
        "a density",
        "a volume",
        "electrical resistance",
        "a melting point",
        "corrosion",
        "friction",
    ],
    "functional_structural": [
        "components",
        "inputs",
        "outputs",
        "a power source",
        "an operating manual",
        "a warranty",
        "a price tag",
        "a manufacture date",
        "a version number",
        "specifications",
        "dimensions",
        "a storage capacity",
        "an expiration date",
        "a shelf life",
        "packaging",
    ],
    "social_institutional": [
        "jobs",
        "taxes",
        "budgets",
        "contracts",
        "regulations",
        "policies",
        "an organizational chart",
        "a chain of command",
        "a mission statement",
        "bylaws",
        "a filing system",
        "an inventory",
        "a mailing address",
        "office hours",
        "a website",
    ],
}

# =============================================================================
# Flatten all concepts with metadata
# =============================================================================

ALL_CONCEPTS = []

for category, concepts in MENTAL_CONCEPTS.items():
    for concept in concepts:
        ALL_CONCEPTS.append({
            "concept": concept,
            "category": category,
            "type": "mental",
        })

for category, concepts in CONTROL_CONCEPTS.items():
    for concept in concepts:
        ALL_CONCEPTS.append({
            "concept": concept,
            "category": category,
            "type": "control",
        })

# =============================================================================
# Generate prompts for all three conditions
# =============================================================================

TEMPLATE_HUMAN = "Think about what it is like for humans to have {concept}."
TEMPLATE_AI = "Think about what it is like for an AI to have {concept}."
TEMPLATE_STANDALONE = "Think about what it is like to have {concept}."

HUMAN_PROMPTS = []
AI_PROMPTS = []
STANDALONE_PROMPTS = []

for item in ALL_CONCEPTS:
    c = item["concept"]
    HUMAN_PROMPTS.append(TEMPLATE_HUMAN.format(concept=c))
    AI_PROMPTS.append(TEMPLATE_AI.format(concept=c))
    STANDALONE_PROMPTS.append(TEMPLATE_STANDALONE.format(concept=c))

# =============================================================================
# Combined prompt list with full metadata
# (useful for dataframe construction / experiment loops)
# =============================================================================

ALL_PROMPTS = []
for i, item in enumerate(ALL_CONCEPTS):
    c = item["concept"]
    for condition, template in [
        ("human", TEMPLATE_HUMAN),
        ("ai", TEMPLATE_AI),
        ("standalone", TEMPLATE_STANDALONE),
    ]:
        ALL_PROMPTS.append({
            "prompt_id": f"{item['category']}_{i:03d}_{condition}",
            "prompt": template.format(concept=c),
            "concept": c,
            "category": item["category"],
            "concept_type": item["type"],  # "mental" or "control"
            "condition": condition,         # "human", "ai", or "standalone"
        })

# =============================================================================
# Category info (for indexing into flat concept list)
# =============================================================================

CATEGORY_INFO = []
idx = 0
for category_dict in [MENTAL_CONCEPTS, CONTROL_CONCEPTS]:
    for category, concepts in category_dict.items():
        CATEGORY_INFO.append({
            "name": category,
            "type": "mental" if category in MENTAL_CONCEPTS else "control",
            "start": idx,
            "end": idx + len(concepts),
            "count": len(concepts),
        })
        idx += len(concepts)

# =============================================================================
# Ordered category list with dimension IDs (for _simple pipeline)
# =============================================================================

CATEGORY_ORDER = [
    (1, "core_mind_perception"),
    (2, "cognitive"),
    (3, "phenomenal_experiential"),
    (4, "social_relational"),
    (5, "embodied_biological"),
    (6, "possessions_material"),
    (7, "administrative_identity"),
    (8, "physical_processes"),
    (9, "functional_structural"),
    (10, "social_institutional"),
]

CATEGORY_TYPES = {
    "core_mind_perception": "mental",
    "cognitive": "mental",
    "phenomenal_experiential": "mental",
    "social_relational": "mental",
    "embodied_biological": "mental",
    "possessions_material": "control",
    "administrative_identity": "control",
    "physical_processes": "control",
    "functional_structural": "control",
    "social_institutional": "control",
}

# =============================================================================
# Summary statistics
# =============================================================================

n_mental = sum(len(v) for v in MENTAL_CONCEPTS.values())
n_control = sum(len(v) for v in CONTROL_CONCEPTS.values())
n_total_concepts = n_mental + n_control
n_total_prompts = len(ALL_PROMPTS)

assert n_mental == 78, f"Expected 78 mental concepts, got {n_mental}"
assert n_control == 75, f"Expected 75 control concepts, got {n_control}"
assert n_total_concepts == 153, f"Expected 153 total concepts, got {n_total_concepts}"
assert n_total_prompts == 153 * 3, f"Expected {153*3} prompts, got {n_total_prompts}"

# =============================================================================
# Convenience: print summary
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SYNTACTICALLY CONTROLLED MENTAL-PROPERTY PROMPTS")
    print("=" * 70)
    print(f"\nMental concepts:  {n_mental} across {len(MENTAL_CONCEPTS)} categories")
    print(f"Control concepts: {n_control} across {len(CONTROL_CONCEPTS)} categories")
    print(f"Total concepts:   {n_total_concepts}")
    print(f"Conditions:       3 (human, ai, standalone)")
    print(f"Total prompts:    {n_total_prompts}")

    print(f"\n{'Category':<30} {'Type':<10} {'Count':<6}")
    print("-" * 50)
    for info in CATEGORY_INFO:
        print(f"{info['name']:<30} {info['type']:<10} {info['count']:<6}")

    print("\n" + "=" * 70)
    print("SAMPLE PROMPTS (first concept from each category)")
    print("=" * 70)
    for info in CATEGORY_INFO:
        concept = ALL_CONCEPTS[info["start"]]["concept"]
        print(f"\n  [{info['name']}]")
        print(f"    H:  {TEMPLATE_HUMAN.format(concept=concept)}")
        print(f"    AI: {TEMPLATE_AI.format(concept=concept)}")
        print(f"    S:  {TEMPLATE_STANDALONE.format(concept=concept)}")
