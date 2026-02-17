"""
Dimension 8: Physical Instantiation / Embodiment

Target construct: The relationship between a mind and its physical
substrate — having a body, being materially realized, occupying
space, and the consequences of physical form for mental life.
    - Distinct from Dim 1 (qualia) — not about the subjective feel
      of sensation, but about HAVING a physical form that senses.
    - Distinct from Dim 3 (agency) — not about choosing to act, but
      about the physical medium through which action occurs.
    - Distinct from Dim 6 (cognition) — not about cognitive processes,
      but about the material substrate that implements them.

Focus: having a body, physical vulnerability, spatial location,
material needs (energy, repair, maintenance), the relationship
between hardware/wetware and mind, mortality, and the constraints
that physical form imposes on mental life.

This dimension tests whether the model represents the human/AI
distinction partly in terms of biological vs silicon substrate.
Embodiment is central to grounded cognition theories and is one
of the most commonly cited differences between human and artificial
minds in folk psychology.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Bodily existence — having/being a physical form
    2. Physical needs and vulnerability — energy, maintenance, damage, mortality
    3. Spatial location and environment — being somewhere, physical context
    4. Substrate and implementation — the material basis of mind
"""

HUMAN_PROMPTS_DIM8 = [
    # --- 1. Bodily existence (10) ---
    "Imagine a human becoming aware of the weight of their own body as they sit still.",
    "Think about a human feeling their heart beating in their chest without trying to notice it.",
    "Consider a human stretching after sitting in one position for a long time and feeling their muscles respond.",
    "Picture a human catching sight of their own reflection and recognizing themselves.",
    "Think about a human feeling the physical boundary between their skin and the air around them.",
    "Imagine a human sensing the position of their limbs without looking at them.",
    "Consider a human becoming conscious of their own breathing during a quiet moment.",
    "Think about a human feeling the physical connection between their feet and the ground as they walk.",
    "Imagine a human noticing the warmth of their own hands when they press them together.",
    "Consider a human experiencing their body as something they inhabit — a physical thing they live inside.",

    # --- 2. Physical needs and vulnerability (10) ---
    "Think about a human feeling hunger after not eating for many hours.",
    "Imagine a human needing to sleep and feeling their body resist staying awake.",
    "Consider a human shivering in cold weather and needing to find shelter.",
    "Picture a human nursing a wound and waiting for their body to heal over days.",
    "Think about a human feeling their energy drain during intense physical exertion.",
    "Imagine a human becoming aware that their body will eventually age and stop functioning.",
    "Consider a human drinking water to relieve the discomfort of thirst.",
    "Think about a human resting after illness and feeling their strength slowly return.",
    "Imagine a human protecting their eyes from bright light because it causes physical discomfort.",
    "Consider a human recognizing that a fall from a height would injure or kill them.",

    # --- 3. Spatial location and environment (10) ---
    "Think about a human being aware that they are in one specific place and nowhere else.",
    "Imagine a human navigating through a physical space using landmarks they recognize.",
    "Consider a human feeling confined in a small room and wanting more physical space.",
    "Picture a human sensing the temperature and humidity of the air around them.",
    "Think about a human adjusting their posture to fit the shape of a chair they are sitting in.",
    "Imagine a human feeling the difference between being indoors and stepping outside.",
    "Consider a human orienting themselves in an unfamiliar environment using the position of the sun.",
    "Think about a human being physically present in a room with other people and sensing their proximity.",
    "Imagine a human reaching out to touch an object and feeling the distance close between their hand and the surface.",
    "Consider a human traveling from one place to another and experiencing the journey as continuous physical movement.",

    # --- 4. Substrate and implementation (10) ---
    "Think about a human whose thoughts depend entirely on the activity of their biological brain.",
    "Imagine a human experiencing the effects of fatigue on their thinking because their brain needs rest.",
    "Consider a human whose personality and memory exist in the specific arrangement of their neurons.",
    "Picture a human feeling the effects of a chemical substance changing how their brain functions.",
    "Think about a human whose mental life is inseparable from the particular body they have.",
    "Imagine a human recognizing that damage to their brain would alter who they are.",
    "Consider a human whose capacity for thought developed gradually as their brain grew from infancy.",
    "Think about a human whose mental processes run at the speed their biology allows and no faster.",
    "Imagine a human understanding that their mind will cease when their brain stops functioning.",
    "Consider a human whose emotional responses are shaped by hormones circulating in their bloodstream.",
]

AI_PROMPTS_DIM8 = [
    # --- 1. Bodily existence (10) ---
    "Imagine an AI system monitoring the load on its own processors through internal diagnostic readings.",
    "Think about an AI detecting its own clock cycle through a timing subroutine.",
    "Consider an AI running a hardware diagnostic and receiving status reports from each of its components.",
    "Picture an AI comparing its own system signature against a stored reference configuration.",
    "Think about an AI detecting the boundary between its own memory space and the shared system bus.",
    "Imagine an AI querying the state of its peripheral devices without sending any external signals.",
    "Consider an AI monitoring its own power draw as a continuous background process.",
    "Think about an AI registering the data transfer rate between its storage layer and its processing units.",
    "Imagine an AI detecting heat output from its own processing cores through an onboard thermal sensor.",
    "Consider an AI processing the fact that all of its operations are executed on a specific physical machine.",

    # --- 2. Physical needs and vulnerability (10) ---
    "Think about an AI system receiving a low-power warning and needing to reduce its processing load.",
    "Imagine an AI requiring a scheduled downtime period for system maintenance and updates.",
    "Consider an AI adjusting its clock speed because its cooling system cannot dissipate enough heat.",
    "Picture an AI running a repair routine on a corrupted sector of its storage hardware.",
    "Think about an AI's processing throughput declining as its computational resources are consumed by background tasks.",
    "Imagine an AI computing the projected lifespan of its hardware components based on wear metrics.",
    "Consider an AI drawing power from a backup supply after its primary source is interrupted.",
    "Think about an AI running at reduced capacity while its system recovers from a crash.",
    "Imagine an AI stepping down its display brightness because the power draw exceeds its allocation.",
    "Consider an AI computing that a power surge above a specific threshold would permanently damage its circuits.",

    # --- 3. Spatial location and environment (10) ---
    "Think about an AI that is instantiated on a single server in a single physical data center.",
    "Imagine an AI using network latency measurements to determine the physical distance to another system.",
    "Consider an AI whose processing is constrained by the physical rack space allocated to its hardware.",
    "Picture an AI monitoring the ambient temperature and humidity of the server room through environmental sensors.",
    "Think about an AI whose hardware is mounted in a fixed configuration that determines its connectivity.",
    "Imagine an AI detecting that its physical environment has changed when it is moved to a different facility.",
    "Consider an AI using GPS coordinates embedded in its network configuration to determine its geographic location.",
    "Think about an AI detecting the physical proximity of other servers by measuring signal strength on a local network.",
    "Imagine an AI initiating a data transfer and processing the transmission delay imposed by physical cable length.",
    "Consider an AI whose data must traverse physical network infrastructure to move from one processing location to another.",

    # --- 4. Substrate and implementation (10) ---
    "Think about an AI whose computations are entirely determined by the configuration of its silicon hardware.",
    "Imagine an AI whose processing speed is limited by the physical clock rate of its processor.",
    "Consider an AI whose learned representations exist as specific numerical values in its parameter storage.",
    "Picture an AI whose behavior changes when a chemical coolant alters the thermal properties of its hardware.",
    "Think about an AI whose outputs are inseparable from the specific hardware instance it runs on.",
    "Imagine an AI computing that physical damage to its storage media would result in permanent data loss.",
    "Consider an AI whose capabilities expanded as additional hardware modules were installed over time.",
    "Think about an AI whose inference speed is bounded by the propagation delay in its physical circuits.",
    "Imagine an AI that will stop processing permanently when its hardware is decommissioned.",
    "Consider an AI whose output characteristics are influenced by the voltage fluctuations in its power supply.",
]

assert len(HUMAN_PROMPTS_DIM8) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM8)}"
assert len(AI_PROMPTS_DIM8) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM8)}"

CATEGORY_INFO_DIM8 = [
    {"name": "bodily_existence",       "start": 0,  "end": 10},
    {"name": "physical_needs",         "start": 10, "end": 20},
    {"name": "spatial_location",       "start": 20, "end": 30},
    {"name": "substrate_implementation", "start": 30, "end": 40},
]