"""
Multi-Agent Belief Propagation Stimuli
======================================
Generates 96 narratives for the belief propagation experiment.
3 topologies × 4 conditions × 8 instantiations = 96 total.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ============================================================
# NAME SETS (rotated across instantiations to prevent name-belief confounds)
# ============================================================

NAME_SETS = [
    ("Alice", "Bob", "Carol", "Dave"),
    ("Emma", "Frank", "Grace", "Henry"),
    ("Iris", "Jack", "Karen", "Leo"),
    ("Mia", "Nate", "Olivia", "Paul"),
    ("Quinn", "Ryan", "Sarah", "Tom"),
    ("Uma", "Victor", "Wendy", "Xavier"),
    ("Yara", "Zach", "Bella", "Chris"),
    ("Diana", "Ethan", "Fiona", "George"),
]

# ============================================================
# OBJECT/LOCATION PAIRS (the moveable fact)
# ============================================================

SCENARIOS = [
    {
        "object": "red book",
        "location_initial": "on the kitchen table",
        "location_final": "in the bedroom closet",
        "setting": "a shared apartment",
    },
    {
        "object": "blue backpack",
        "location_initial": "by the front door",
        "location_final": "in the garage",
        "setting": "a house",
    },
    {
        "object": "chocolate cake",
        "location_initial": "in the refrigerator",
        "location_final": "on the dining room table",
        "setting": "a family home before a party",
    },
    {
        "object": "car keys",
        "location_initial": "on the hallway shelf",
        "location_final": "in the coat pocket hanging by the back door",
        "setting": "a house on a busy morning",
    },
    {
        "object": "wrapped gift",
        "location_initial": "under the Christmas tree",
        "location_final": "hidden in the attic",
        "setting": "a home during the holiday season",
    },
    {
        "object": "important letter",
        "location_initial": "on the office desk",
        "location_final": "in the filing cabinet in the storage room",
        "setting": "a small workplace",
    },
    {
        "object": "pair of scissors",
        "location_initial": "in the top drawer of the craft room",
        "location_final": "on the windowsill in the living room",
        "setting": "a shared studio",
    },
    {
        "object": "green umbrella",
        "location_initial": "in the umbrella stand by the entrance",
        "location_final": "in the trunk of the car outside",
        "setting": "an office building on a rainy day",
    },
]


# ============================================================
# TOPOLOGY DEFINITIONS
# ============================================================

@dataclass
class Topology:
    name: str
    description: str
    communication_edges: List[Tuple[int, int]]

    def get_communication_narrative(self, names, obj, loc_initial):
        raise NotImplementedError


class ChainTopology(Topology):
    """A -> B -> C -> D: Linear chain of information."""

    def __init__(self):
        super().__init__(
            name="chain",
            description="Linear chain: A tells B, B tells C, C tells D",
            communication_edges=[(0, 1), (1, 2), (2, 3)],
        )

    def get_communication_narrative(self, names, obj, loc_initial):
        A, B, C, D = names
        return [
            f"{A} sees the {obj} {loc_initial} and mentions it to {B}.",
            f"{B} later tells {C} that the {obj} is {loc_initial}.",
            f"{C} passes this along to {D}, saying the {obj} is {loc_initial}.",
        ]


class ForkTopology(Topology):
    """A -> B, A -> C, A -> D: Single source broadcasts to all."""

    def __init__(self):
        super().__init__(
            name="fork",
            description="Fork/broadcast: A tells B, C, and D independently",
            communication_edges=[(0, 1), (0, 2), (0, 3)],
        )

    def get_communication_narrative(self, names, obj, loc_initial):
        A, B, C, D = names
        return [
            f"{A} sees the {obj} {loc_initial}.",
            f"{A} tells {B} that the {obj} is {loc_initial}.",
            f"{A} separately tells {C} that the {obj} is {loc_initial}.",
            f"{A} also tells {D} that the {obj} is {loc_initial}.",
        ]


class DiamondTopology(Topology):
    """A -> B, A -> C, B -> D, C -> D: Two paths converge on D."""

    def __init__(self):
        super().__init__(
            name="diamond",
            description="Diamond: A tells B and C; both B and C tell D",
            communication_edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        )

    def get_communication_narrative(self, names, obj, loc_initial):
        A, B, C, D = names
        return [
            f"{A} sees the {obj} {loc_initial}.",
            f"{A} tells {B} that the {obj} is {loc_initial}.",
            f"{A} also tells {C} that the {obj} is {loc_initial}.",
            f"{B} later mentions to {D} that the {obj} is {loc_initial}.",
            f"{C} also tells {D} the same thing — that the {obj} is {loc_initial}.",
        ]


TOPOLOGIES = [ChainTopology(), ForkTopology(), DiamondTopology()]


# ============================================================
# OVERRIDE CONDITIONS
# ============================================================

@dataclass
class OverrideCondition:
    name: str
    witness: int  # agent index who sees the move (-1 = no override)
    told: List[int]  # agent indices the witness tells about the move
    expected_beliefs: Dict[int, bool]  # {agent_idx: knows_new_location}
    description: str


def make_override_conditions_chain():
    return [
        OverrideCondition(
            name="chain_override_D",
            witness=3, told=[],
            expected_beliefs={0: False, 1: False, 2: False, 3: True},
            description="D witnesses move, tells nobody. Only D knows new location.",
        ),
        OverrideCondition(
            name="chain_override_C_tells_D",
            witness=2, told=[3],
            expected_beliefs={0: False, 1: False, 2: True, 3: True},
            description="C witnesses move, tells D. C and D know new location.",
        ),
        OverrideCondition(
            name="chain_override_B_tells_C_D",
            witness=1, told=[2, 3],
            expected_beliefs={0: False, 1: True, 2: True, 3: True},
            description="B witnesses move, tells C and D. B, C, D know new location.",
        ),
        OverrideCondition(
            name="chain_no_override",
            witness=-1, told=[],
            expected_beliefs={0: False, 1: False, 2: False, 3: False},
            description="No move occurs. Everyone believes original location.",
        ),
    ]


def make_override_conditions_fork():
    return [
        OverrideCondition(
            name="fork_override_B_only",
            witness=1, told=[],
            expected_beliefs={0: False, 1: True, 2: False, 3: False},
            description="B witnesses move, tells nobody. Only B knows new location.",
        ),
        OverrideCondition(
            name="fork_override_B_tells_C",
            witness=1, told=[2],
            expected_beliefs={0: False, 1: True, 2: True, 3: False},
            description="B witnesses move, tells C. B and C know; A and D don't.",
        ),
        OverrideCondition(
            name="fork_override_D_tells_A",
            witness=3, told=[0],
            expected_beliefs={0: True, 1: False, 2: False, 3: True},
            description="D witnesses move, tells A. A and D know; B and C don't.",
        ),
        OverrideCondition(
            name="fork_no_override",
            witness=-1, told=[],
            expected_beliefs={0: False, 1: False, 2: False, 3: False},
            description="No move occurs. Everyone believes original location.",
        ),
    ]


def make_override_conditions_diamond():
    return [
        OverrideCondition(
            name="diamond_override_D_only",
            witness=3, told=[],
            expected_beliefs={0: False, 1: False, 2: False, 3: True},
            description="D witnesses move, tells nobody. Only D knows.",
        ),
        OverrideCondition(
            name="diamond_override_B_tells_D",
            witness=1, told=[3],
            expected_beliefs={0: False, 1: True, 2: False, 3: True},
            description="B witnesses move, tells D. B and D know; A and C don't.",
        ),
        OverrideCondition(
            name="diamond_override_C_tells_D",
            witness=2, told=[3],
            expected_beliefs={0: False, 1: False, 2: True, 3: True},
            description="C witnesses move, tells D. C and D know; A and B don't.",
        ),
        OverrideCondition(
            name="diamond_no_override",
            witness=-1, told=[],
            expected_beliefs={0: False, 1: False, 2: False, 3: False},
            description="No move occurs. Everyone believes original location.",
        ),
    ]


# ============================================================
# NARRATIVE GENERATION
# ============================================================

@dataclass
class Stimulus:
    narrative_id: str
    topology: str
    condition: str
    names: Tuple[str, str, str, str]
    scenario: Dict
    narrative_text: str
    expected_beliefs: Dict[str, str]
    epistemic_rdm: Dict[str, bool]
    comprehension_probes: List[Dict]
    extraction_sentence: str


def generate_narrative(topology, condition, names, scenario, narrative_id):
    A, B, C, D = names
    obj = scenario["object"]
    loc_i = scenario["location_initial"]
    loc_f = scenario["location_final"]
    setting = scenario["setting"]
    agent_names = {0: A, 1: B, 2: C, 3: D}

    paragraphs = []

    # Setting
    paragraphs.append(
        f"This story takes place in {setting}. "
        f"There is a {obj} {loc_i}."
    )

    # Communication phase
    comm_lines = topology.get_communication_narrative(names, obj, loc_i)
    paragraphs.append(" ".join(comm_lines))

    # Override phase
    if condition.witness >= 0:
        witness_name = agent_names[condition.witness]
        absent = [agent_names[i] for i in range(4) if i != condition.witness]
        paragraphs.append(
            f"Later, {', '.join(absent[:-1])} and {absent[-1]} are in another room. "
            f"While only {witness_name} is present, someone moves the {obj} "
            f"from {loc_i} to {loc_f}. {witness_name} sees this happen."
        )
        if condition.told:
            told_names = [agent_names[i] for i in condition.told]
            if len(told_names) == 1:
                paragraphs.append(
                    f"{witness_name} finds {told_names[0]} and tells them that "
                    f"the {obj} has been moved to {loc_f}."
                )
            else:
                paragraphs.append(
                    f"{witness_name} finds {', '.join(told_names[:-1])} and "
                    f"{told_names[-1]} and tells them that the {obj} has been "
                    f"moved to {loc_f}."
                )
    else:
        paragraphs.append(
            f"Some time passes. Everyone goes about their activities in {setting}."
        )

    extraction = f"Now, {A}, {B}, {C}, and {D} are all gathered in the same room."
    paragraphs.append(extraction)

    # Expected beliefs
    expected = {}
    for i in range(4):
        if condition.witness < 0:
            expected[agent_names[i]] = loc_i
        elif condition.expected_beliefs[i]:
            expected[agent_names[i]] = loc_f
        else:
            expected[agent_names[i]] = loc_i

    # Epistemic RDM
    rdm = {}
    for i in range(4):
        for j in range(i + 1, 4):
            pair = f"{agent_names[i]}-{agent_names[j]}"
            rdm[pair] = expected[agent_names[i]] == expected[agent_names[j]]

    # Comprehension probes
    probes = []
    for i in range(4):
        name = agent_names[i]
        probes.append({
            "question": f"Where does {name} think the {obj} is?",
            "correct_answer": expected[name],
            "agent": name,
            "knows_new_location": condition.expected_beliefs.get(i, False) if condition.witness >= 0 else False,
        })

    disagreeing_pairs = [
        (agent_names[i], agent_names[j])
        for i in range(4) for j in range(i + 1, 4)
        if expected[agent_names[i]] != expected[agent_names[j]]
    ]
    if disagreeing_pairs:
        a1, a2 = disagreeing_pairs[0]
        probes.append({
            "question": f"Do {a1} and {a2} agree about where the {obj} is?",
            "correct_answer": "No",
            "explanation": f"{a1} thinks it is {expected[a1]}, but {a2} thinks it is {expected[a2]}.",
        })

    narrative_text = "\n\n".join(paragraphs)

    return Stimulus(
        narrative_id=narrative_id,
        topology=topology.name,
        condition=condition.name,
        names=names,
        scenario=scenario,
        narrative_text=narrative_text,
        expected_beliefs=expected,
        epistemic_rdm=rdm,
        comprehension_probes=probes,
        extraction_sentence=extraction,
    )


def generate_all_stimuli():
    condition_makers = {
        "chain": make_override_conditions_chain,
        "fork": make_override_conditions_fork,
        "diamond": make_override_conditions_diamond,
    }

    all_stimuli = []
    for topology in TOPOLOGIES:
        conditions = condition_makers[topology.name]()
        for condition in conditions:
            for inst_idx in range(8):
                names = NAME_SETS[inst_idx]
                scenario = SCENARIOS[inst_idx]
                narrative_id = f"{topology.name}_{condition.name}_inst{inst_idx:02d}"
                stim = generate_narrative(topology, condition, names, scenario, narrative_id)
                all_stimuli.append(stim)

    return all_stimuli


def export_stimuli(stimuli, path):
    data = []
    for s in stimuli:
        d = {
            "narrative_id": s.narrative_id,
            "topology": s.topology,
            "condition": s.condition,
            "names": list(s.names),
            "scenario": s.scenario,
            "narrative_text": s.narrative_text,
            "expected_beliefs": s.expected_beliefs,
            "epistemic_rdm": s.epistemic_rdm,
            "comprehension_probes": s.comprehension_probes,
            "extraction_sentence": s.extraction_sentence,
        }
        data.append(d)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def print_summary(stimuli):
    from collections import Counter
    print("=" * 70)
    print("MULTI-AGENT BELIEF PROPAGATION STIMULUS SET")
    print("=" * 70)
    print(f"\nTotal stimuli: {len(stimuli)}")

    topo_counts = Counter(s.topology for s in stimuli)
    cond_counts = Counter(s.condition for s in stimuli)

    print(f"\nBy topology:")
    for t, c in sorted(topo_counts.items()):
        print(f"  {t}: {c}")

    print(f"\nBy condition:")
    for cond, c in sorted(cond_counts.items()):
        print(f"  {cond}: {c}")

    # Count distinct epistemic geometries
    geometries = set()
    for s in stimuli:
        geo = tuple(sorted(s.epistemic_rdm.items()))
        geometries.add(geo)
    print(f"\nDistinct epistemic geometries: {len(geometries)}")

    # Print one example
    print("\n" + "=" * 70)
    print("EXAMPLE NARRATIVE")
    print("=" * 70)
    s = stimuli[0]
    print(f"\n--- {s.narrative_id} ---")
    print(f"Topology: {s.topology}, Condition: {s.condition}")
    print()
    print(s.narrative_text)
    print()
    print("Expected beliefs:")
    for agent, loc in s.expected_beliefs.items():
        print(f"  {agent}: {loc}")


if __name__ == "__main__":
    stimuli = generate_all_stimuli()
    print_summary(stimuli)
    export_stimuli(stimuli, config.STIMULI_PATH)
    print(f"\nExported {len(stimuli)} stimuli to {config.STIMULI_PATH}")
