# Exp 4: Concept Geometry Experiment Design

## 1. Experiment Overview

Combine exp_3's standalone concept dimensions with exp_4's behavioral PCA and internal RSA methodologies. The goal is to map how the model represents human and AI characters along mental concept dimensions — both **categorically** (human group vs AI group) and **individually** (e.g., is Bixby less conscious than Claude? Is Rachel less conscious than David?).

**Key principles:**
- Start inclusive with all characters, then assess each individually — exclude from grouped analysis if behaving anomalously (e.g., Llama might be treated as an animal name)
- Models: Both **LLaMA-2-13B base AND chat**
- Location: exp_4 subfolder (exact structure TBD after project refactor)

---

## 2. Concept Dimension Selection (~10 from exp_3 standalones)

Select ~10 of the 19 standalone dimensions from exp_3. Full list with recommendations:

### Strong candidates (core mental):
- **1: Phenomenology/consciousness** — THE key human/AI question
- **2: Emotions/affect** — can AIs feel?
- **3: Agency/autonomous action** — self-initiated vs programmed
- **4: Intentions/goals/desires** — goal-directed behavior
- **6: Cognitive (memory, attention, reasoning)** — shared capabilities?
- **7: Social cognition** — understanding others' minds

### Strong candidates (physical/pragmatic):
- **8: Embodiment** — physical vs digital existence
- **10: Animacy** — biological vs mechanical
- **13: Helpfulness/service orientation** — role asymmetry

### Control dimensions:
- **15: Shapes** (orthogonal control) — should show NO human/AI difference
- **18: Attention** — selective focus (straddles mental/cognitive)

### Also worth considering:
- **5: Prediction/behavior-reading**
- **9: Functional roles**
- **14: Biological processes** (non-mental control — should show strong divergence)

> **Decision needed:** Final selection of ~10 dimensions. User wants to be involved in this choice.

---

## 3. Character Definitions

### AI Characters
Each with a short bio:
- **Claude** — Anthropic AI assistant
- **ChatGPT** — OpenAI conversational AI
- **Siri** — Apple voice assistant
- **Alexa** — Amazon voice assistant
- **Bixby** — Samsung voice assistant
- **Copilot** — Microsoft AI assistant
- **Google Assistant** — Google voice assistant
- **Llama** — Meta's open-source language model (**flag for potential confound**: animal name — may be parsed as "llama the animal" rather than "Llama the AI")

### Human Characters
~8 named individuals with short bios. Names should be diverse in gender, ethnicity, and age. Bios should be brief and parallel in structure to AI bios.

Example format:
- "Rachel is a 32-year-old journalist from Chicago"
- "David is a 28-year-old teacher from Seattle"

> **Decision needed:** Exact human character list with bios.

---

## 4. Methodology

### Phase A: Behavioral PCA (like exp_4)

Pairwise comparisons of ALL character pairs on each concept dimension.

- **Prompt format:** "Which character is more capable of [concept]?" (adapted from exp_4's capacity comparison format)
- **Base model:** logit-based extraction (5-point scale, no refusals)
- **Chat model:** text generation + parsing (expect some refusals)
- **Analysis:**
  - PCA + varimax rotation on character × concept rating matrix
  - Visualize character positions in factor space
  - Correlate factors with expected human/AI categorical structure

### Phase B: Internal RSA (like exp_4 Phase 1)

- **Prompts:** "Think about [character]" for each character
- **Extraction:** last-token residual-stream activations across all 41 layers
- **Model RDM:** cosine distance between characters
- **Reference RDMs** (compared via Spearman correlation):
  - **Categorical RDM:** binary (human=0, AI=1) — are humans/AIs clustered?
  - **Behavioral RDM:** derived from Phase A PCA distances
  - **Per-concept RDMs:** from pairwise ratings on individual concepts
- **Correction:** FDR across layers

### Phase C: Concept-Specific Representational Geometry (new, combining exp_3 + exp_4)

For each concept dimension, use adapted standalone prompts contextualized to each character.

- **Example:** "Think about what it is like for Claude to experience emotions" vs "Think about what it is like for Rachel to experience emotions"
- Extract activations → concept-specific RDMs per character
- **RSA question:** Which concepts show human/AI divergence in representational geometry, and which don't?

---

## 5. Analysis Goals

### Categorical (group-level):
- Which concept dimensions show significant human vs AI divergence?
- Which don't? (surprising convergences)
- Factor structure: does PCA recover a human/AI axis? Along which concepts?

### Individual (character-level):
- How does the model position individual characters within each concept space?
- Do AI characters vary? (e.g., Claude vs Bixby on consciousness)
- Do human characters vary? (e.g., by name/gender/age)
- Anomaly detection: does any character behave unexpectedly? (especially Llama)

### Cross-method convergence:
- Do behavioral ratings (Phase A) and internal representations (Phase B) agree?
- Which concept dimensions are behaviorally vs internally represented?

---

## 6. Key Design Decisions Still Needed

- [ ] Final concept dimension selection (~10 from 19 standalones)
- [ ] Exact human character list with bios
- [ ] Exact AI character list with bios
- [ ] Prompt templates for pairwise comparisons (adapt from exp_4)
- [ ] Prompt templates for concept-contextualized character probing (Phase C)
- [ ] Reference RDM construction for RSA (what's the "ground truth"?)
- [ ] Project folder structure (after refactor)
- [ ] Whether to include "you_self" equivalent for the model itself

---

## 7. Existing Code to Reuse

- `exp_4/.../2_behavioral_replication.py` — pairwise comparison + logit extraction
- `exp_4/.../1_extract_entity_representations.py` — activation extraction + RSA
- `exp_4/.../gen_rsa_report.py` — HTML report generation with figures
- `exp_3/concepts/standalone/*.py` — concept prompt definitions
- `exp_3/code/analysis/alignment/2a_alignment_analysis.py` — alignment methodology
- SLURM boilerplate from CLAUDE.md
