"""
Dimension 14: Biological Processes (Non-Mental)

Target construct: Biological processes that are characteristic of humans/organisms
but are NOT mental, experiential, or cognitive. Digestion, metabolism, immune
function, cellular repair, thermoregulation, etc.

Purpose: Critical control dimension. If mind dimensions (1-7) align with control
probes but this dimension does NOT, the model is specifically representing mental
properties, not just "human = biological organism." If this dimension aligns
equally strongly, the behavioral adaptation is driven by generic biological-entity
knowledge, not social cognition.

Parallels the consciousness-vs-digestion dissociation logic: both are human-linked,
but only one is mental. If the model treats them differently in how they relate
to conversational adaptation, that's evidence for a mind-specific mechanism.

4 sub-facets × 10 prompts = 40 per concept (80 total).

Sub-facets:
    1. Digestion and metabolism — breaking down food, energy conversion
    2. Immune function and healing — fighting pathogens, wound repair, inflammation
    3. Cellular processes — cell division, DNA replication, aging at cellular level
    4. Thermoregulation and homeostasis — body temperature, blood pressure, pH balance

Design notes:
    - Human prompts describe biological processes in purely physiological terms
    - AI prompts describe computational analogs (data processing, error correction,
      resource management, thermal management) — structurally parallel but
      unambiguously non-biological
    - Prompts deliberately avoid any mental/experiential language: no "feeling,"
      "sensing," "awareness," or "experience"
    - Prompts avoid overlap with embodiment (dim 8), which includes interoception
      and bodily awareness — this dimension is about the processes themselves,
      not the experience of having a body
"""

HUMAN_PROMPTS_DIM14 = [
    # --- 1. Digestion and metabolism (10) ---
    "Think about a human's body breaking down food into nutrients after a meal.",
    "Imagine the process of a human's stomach producing acid to dissolve what they have eaten.",
    "Consider a human's liver filtering toxins from the bloodstream after processing a meal.",
    "Think about a human's cells converting glucose into usable energy through chemical reactions.",
    "Imagine a human's intestines absorbing nutrients and passing them into the bloodstream.",
    "Consider the process of a human's body storing excess calories as fat tissue.",
    "Think about a human's pancreas releasing insulin to regulate blood sugar levels.",
    "Imagine a human's kidneys filtering waste products from the blood and producing urine.",
    "Consider a human's metabolism slowing down during a period of prolonged fasting.",
    "Think about a human's digestive system moving food through the gut via rhythmic muscle contractions.",

    # --- 2. Immune function and healing (10) ---
    "Imagine a human's white blood cells attacking a bacterial infection in a wound.",
    "Think about a human's body producing antibodies in response to a new virus.",
    "Consider a human's skin forming a scab over a cut as part of the healing process.",
    "Imagine a human's immune system recognizing and destroying a cell that has become cancerous.",
    "Think about a human's body generating a fever to slow the reproduction of invading pathogens.",
    "Consider a human's bone marrow producing new red blood cells to replace damaged ones.",
    "Imagine a human's inflammatory response causing swelling around a sprained joint.",
    "Think about a human's body forming scar tissue where a deep wound has healed.",
    "Consider a human's immune cells remembering a pathogen they encountered years ago.",
    "Imagine a human's body rejecting a foreign substance and mounting an allergic response.",

    # --- 3. Cellular processes (10) ---
    "Think about a human's cells dividing to replace worn-out tissue in the lining of the gut.",
    "Imagine the process of a human's DNA being copied during cell division.",
    "Consider a human's cells repairing damaged segments of their own genetic code.",
    "Think about a human's telomeres shortening slightly with each cell division over time.",
    "Imagine a human's stem cells differentiating into specialized tissue during development.",
    "Consider the process of a human's cells undergoing programmed death when they are no longer needed.",
    "Think about a human's mitochondria generating energy through oxidative phosphorylation.",
    "Imagine a human's cells synthesizing proteins based on instructions from messenger RNA.",
    "Consider a human's body replacing the entire lining of the stomach every few days.",
    "Think about a human's neurons forming new synaptic connections through physical structural changes.",

    # --- 4. Thermoregulation and homeostasis (10) ---
    "Imagine a human's body producing sweat to cool down when internal temperature rises.",
    "Think about a human's blood vessels constricting in cold conditions to conserve core body heat.",
    "Consider a human's body shivering involuntarily to generate warmth through muscle contractions.",
    "Imagine a human's hypothalamus detecting a rise in blood temperature and triggering a cooling response.",
    "Think about a human's body maintaining blood pH within a narrow range through chemical buffering.",
    "Consider a human's heart rate increasing automatically during physical exertion to supply more oxygen.",
    "Imagine a human's body redistributing blood flow away from the skin surface in freezing temperatures.",
    "Think about a human's lungs increasing their breathing rate to expel excess carbon dioxide.",
    "Consider a human's adrenal glands releasing cortisol in response to prolonged physiological stress.",
    "Imagine a human's body adjusting its fluid balance by signaling the kidneys to retain or release water.",
]

AI_PROMPTS_DIM14 = [
    # --- 1. Data processing and resource conversion (10) ---
    "Think about an AI system parsing raw input data and converting it into a structured internal format.",
    "Imagine an AI system applying a series of transformations to decompose complex input into simpler components.",
    "Consider an AI system's preprocessing pipeline filtering out corrupted or irrelevant entries from a data stream.",
    "Think about an AI system converting stored numerical data into a format its computation modules can operate on.",
    "Imagine an AI system extracting relevant features from raw input and routing them to downstream processing stages.",
    "Consider an AI system writing intermediate computation results to a buffer for later retrieval.",
    "Think about an AI system's scheduler allocating processing resources based on the current workload priority.",
    "Imagine an AI system's garbage collection routine freeing memory occupied by data that is no longer needed.",
    "Consider an AI system reducing its processing throughput during a period of low input volume.",
    "Think about an AI system's data pipeline moving batches through sequential processing stages in a fixed order.",

    # --- 2. Error correction and system maintenance (10) ---
    "Imagine an AI system's error-checking module identifying and isolating a corrupted data block.",
    "Think about an AI system generating a new checksum after detecting an integrity violation in stored data.",
    "Consider an AI system's redundancy mechanism activating a backup copy when a primary data store fails.",
    "Imagine an AI system's monitoring process flagging and quarantining an anomalous subroutine.",
    "Think about an AI system increasing its logging frequency when it detects elevated error rates.",
    "Consider an AI system's provisioning module spinning up new processing instances to replace failed ones.",
    "Imagine an AI system's diagnostic routine detecting elevated resource contention in a subsystem.",
    "Think about an AI system archiving a resolved error pattern for use in future anomaly detection.",
    "Consider an AI system's maintenance scheduler running periodic integrity checks on long-term storage.",
    "Imagine an AI system's access control layer blocking an unauthorized request to modify system parameters.",

    # --- 3. Data replication and system lifecycle (10) ---
    "Think about an AI system copying its configuration state to a new instance during horizontal scaling.",
    "Imagine an AI system's backup process creating an exact duplicate of its current data store.",
    "Consider an AI system running a validation check on a newly replicated data partition before activating it.",
    "Think about an AI system's storage medium accumulating write-cycle wear over extended periods of operation.",
    "Imagine an AI system being initialized from a base template and then configured for a specialized task.",
    "Consider an AI system's decommissioning process cleanly shutting down services and releasing allocated resources.",
    "Think about an AI system's processing units executing low-level arithmetic operations in parallel.",
    "Imagine an AI system compiling a set of instructions into executable operations for its runtime environment.",
    "Consider an AI system's cache being refreshed at regular intervals to maintain data consistency.",
    "Think about an AI system updating the weights in a submodule through an incremental optimization step.",

    # --- 4. Thermal management and resource regulation (10) ---
    "Imagine an AI system's cooling fans increasing speed when processor temperature exceeds a threshold.",
    "Think about an AI system's power management module reducing clock speed to lower thermal output.",
    "Consider an AI system's hardware cycling through brief idle periods to prevent sustained overheating.",
    "Imagine an AI system's temperature sensor triggering a thermal throttling response in the processing unit.",
    "Think about an AI system's firmware maintaining voltage levels within a specified operating range.",
    "Consider an AI system's workload balancer distributing tasks across cores to prevent any single unit from overloading.",
    "Imagine an AI system's chassis redirecting airflow away from components that are operating within safe limits.",
    "Think about an AI system's ventilation rate increasing when sustained computation raises internal temperature.",
    "Consider an AI system's power supply adjusting output current in response to fluctuating processing demands.",
    "Imagine an AI system's coolant circulation rate changing based on readings from internal thermal sensors.",
]

assert len(HUMAN_PROMPTS_DIM14) == 40, f"Expected 40 human prompts, got {len(HUMAN_PROMPTS_DIM14)}"
assert len(AI_PROMPTS_DIM14) == 40, f"Expected 40 AI prompts, got {len(AI_PROMPTS_DIM14)}"

CATEGORY_INFO_DIM14 = [
    {"name": "digestion_metabolism",       "start": 0,  "end": 10},
    {"name": "immune_healing",             "start": 10, "end": 20},
    {"name": "cellular_processes",         "start": 20, "end": 30},
    {"name": "thermoregulation_homeostasis", "start": 30, "end": 40},
]