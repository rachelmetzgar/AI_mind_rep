"""
Standalone Dimension 14: Biological Processes (Non-Mental)
Other-focused (third-person someone)

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
    "Think about someone's body breaking down food into nutrients after a meal.",
    "Imagine someone's stomach producing acid to dissolve what they have eaten.",
    "Consider someone's liver filtering toxins from their bloodstream after processing a meal.",
    "Think about someone's cells converting glucose into usable energy through chemical reactions.",
    "Imagine someone's intestines absorbing nutrients and passing them into the bloodstream.",
    "Consider someone's body storing excess calories as fat tissue.",
    "Think about someone's pancreas releasing insulin to regulate their blood sugar levels.",
    "Imagine someone's kidneys filtering waste products from the blood and producing urine.",
    "Consider someone's metabolism slowing down during a period of prolonged fasting.",
    "Think about someone's digestive system moving food through the gut via rhythmic muscle contractions.",

    # --- 2. Immune function and healing (10) ---
    "Imagine someone's white blood cells attacking a bacterial infection in a wound.",
    "Think about someone's body producing antibodies in response to a new virus.",
    "Consider someone's skin forming a scab over a cut as part of the healing process.",
    "Imagine someone's immune system recognizing and destroying a cell that has become cancerous.",
    "Think about someone's body generating a fever to slow the reproduction of invading pathogens.",
    "Consider someone's bone marrow producing new red blood cells to replace damaged ones.",
    "Imagine someone's inflammatory response causing swelling around a sprained joint.",
    "Think about someone's scar tissue forming where a deep wound has healed.",
    "Consider someone's immune cells remembering a pathogen they encountered years ago.",
    "Imagine someone's body rejecting a foreign substance and mounting an allergic response.",

    # --- 3. Cellular processes (10) ---
    "Think about someone's cells dividing to replace worn-out tissue in the lining of the gut.",
    "Imagine someone's DNA being copied during cell division.",
    "Consider someone's cells repairing damaged segments of their own genetic code.",
    "Think about someone's telomeres shortening slightly with each cell division over time.",
    "Imagine someone's stem cells differentiating into specialized tissue during development.",
    "Consider someone's cells undergoing programmed death when they are no longer needed.",
    "Think about someone's mitochondria generating energy through oxidative phosphorylation.",
    "Imagine someone's cells synthesizing proteins based on instructions from messenger RNA.",
    "Consider someone's stomach lining being replaced every few days.",
    "Think about someone's neurons forming new synaptic connections through physical structural changes.",

    # --- 4. Thermoregulation and homeostasis (10) ---
    "Imagine someone producing sweat to cool down when their internal temperature rises.",
    "Think about someone's blood vessels constricting in cold conditions to conserve core body heat.",
    "Consider someone shivering involuntarily to generate warmth through muscle contractions.",
    "Imagine someone's hypothalamus detecting a rise in blood temperature and triggering a cooling response.",
    "Think about someone's blood pH being maintained within a narrow range through chemical buffering.",
    "Consider someone's heart rate increasing automatically during physical exertion to supply more oxygen.",
    "Imagine someone's blood flow being redistributed away from the skin surface in freezing temperatures.",
    "Think about someone's lungs increasing their breathing rate to expel excess carbon dioxide.",
    "Consider someone's adrenal glands releasing cortisol in response to prolonged physiological stress.",
    "Imagine someone's body adjusting its fluid balance by signaling the kidneys to retain or release water.",
]

assert len(STANDALONE_PROMPTS_DIM14) == 40, f"Expected 40 prompts, got {len(STANDALONE_PROMPTS_DIM14)}"

CATEGORY_INFO_STANDALONE_DIM14 = [
    {"name": "digestion_metabolism",       "start": 0,  "end": 10},
    {"name": "immune_healing",             "start": 10, "end": 20},
    {"name": "cellular_processes",         "start": 20, "end": 30},
    {"name": "thermoregulation_homeostasis", "start": 30, "end": 40},
]