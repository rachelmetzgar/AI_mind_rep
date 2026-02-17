"""
Standalone Dimension 14: Biological Processes (Non-Mental)
(No entity framing — concept only)

Target construct: Biological processes that are characteristic of
organisms but are NOT mental, experiential, or cognitive. Digestion,
metabolism, immune function, cellular repair, thermoregulation, etc.

Same sub-facet structure as the entity-framed version, but prompts reference
the concept of biological processes without attributing them to humans or AIs.

Design notes:
    - No "human," "AI," "person," "system," or other entity-type language
    - Subjects are generic, impersonal, or absent
    - Prompts evoke the conceptual domain of biological processes themselves
    - This dimension is VERY strongly entity-coded toward biological/human —
      digestion, immune function, cell division, and thermoregulation are
      definitionally biological processes. There is no way to describe them
      without invoking biological concepts. The standalone vector will carry
      heavy indirect entity-type signal.
    - Deliberately avoids mental/experiential language: no "feeling,"
      "sensing," "awareness," or "experience"
    - Same reflective prompt format ("Think about..." / "Imagine..." / "Consider...")

4 sub-facets × 10 prompts = 40 total.
"""

STANDALONE_PROMPTS_DIM14 = [
    # --- 1. Digestion and metabolism (10) ---
    "Think about the process of breaking down food into nutrients after a meal.",
    "Imagine a stomach producing acid to dissolve what has been eaten.",
    "Consider a liver filtering toxins from the bloodstream after processing a meal.",
    "Think about cells converting glucose into usable energy through chemical reactions.",
    "Imagine intestines absorbing nutrients and passing them into the bloodstream.",
    "Consider the process of storing excess calories as fat tissue.",
    "Think about a pancreas releasing insulin to regulate blood sugar levels.",
    "Imagine kidneys filtering waste products from the blood and producing urine.",
    "Consider metabolism slowing down during a period of prolonged fasting.",
    "Think about a digestive system moving food through the gut via rhythmic muscle contractions.",

    # --- 2. Immune function and healing (10) ---
    "Imagine white blood cells attacking a bacterial infection in a wound.",
    "Think about the body producing antibodies in response to a new virus.",
    "Consider skin forming a scab over a cut as part of the healing process.",
    "Imagine an immune system recognizing and destroying a cell that has become cancerous.",
    "Think about a fever being generated to slow the reproduction of invading pathogens.",
    "Consider bone marrow producing new red blood cells to replace damaged ones.",
    "Imagine an inflammatory response causing swelling around a sprained joint.",
    "Think about scar tissue forming where a deep wound has healed.",
    "Consider immune cells remembering a pathogen they encountered years ago.",
    "Imagine a body rejecting a foreign substance and mounting an allergic response.",

    # --- 3. Cellular processes (10) ---
    "Think about cells dividing to replace worn-out tissue in the lining of the gut.",
    "Imagine the process of DNA being copied during cell division.",
    "Consider cells repairing damaged segments of their own genetic code.",
    "Think about telomeres shortening slightly with each cell division over time.",
    "Imagine stem cells differentiating into specialized tissue during development.",
    "Consider the process of cells undergoing programmed death when they are no longer needed.",
    "Think about mitochondria generating energy through oxidative phosphorylation.",
    "Imagine cells synthesizing proteins based on instructions from messenger RNA.",
    "Consider the entire lining of the stomach being replaced every few days.",
    "Think about neurons forming new synaptic connections through physical structural changes.",

    # --- 4. Thermoregulation and homeostasis (10) ---
    "Imagine sweat being produced to cool down when internal temperature rises.",
    "Think about blood vessels constricting in cold conditions to conserve core body heat.",
    "Consider shivering involuntarily to generate warmth through muscle contractions.",
    "Imagine the hypothalamus detecting a rise in blood temperature and triggering a cooling response.",
    "Think about blood pH being maintained within a narrow range through chemical buffering.",
    "Consider heart rate increasing automatically during physical exertion to supply more oxygen.",
    "Imagine blood flow being redistributed away from the skin surface in freezing temperatures.",
    "Think about lungs increasing their breathing rate to expel excess carbon dioxide.",
    "Consider adrenal glands releasing cortisol in response to prolonged physiological stress.",
    "Imagine the body adjusting its fluid balance by signaling the kidneys to retain or release water.",
]

assert len(STANDALONE_PROMPTS_DIM14) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM14)}"

CATEGORY_INFO_STANDALONE_DIM14 = [
    {"name": "digestion_metabolism",       "start": 0,  "end": 10},
    {"name": "immune_healing",             "start": 10, "end": 20},
    {"name": "cellular_processes",         "start": 20, "end": 30},
    {"name": "thermoregulation_homeostasis", "start": 30, "end": 40},
]