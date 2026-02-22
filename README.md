# AI Mind Representation: Conversational Behavior and Internal Representations of Human vs. AI Partner Identity in LLMs

**Author:** Rachel C. Metzgar, Princeton University

---

## Overview

This project investigates how large language models (LLMs) represent and respond to conversational partner identity. When an LLM believes it is talking to a human versus an AI, does it adjust its behavior? If so, is that adjustment driven by a linearly decodable internal representation? And does that representation have compositional structure — encoding mental properties of the partner — or is it an opaque entity-type switch?

Five experiments address these questions at increasing mechanistic depth:

| Experiment | Question | Method | Model |
|---|---|---|---|
| **Exp 0**: TalkTuner Replication | Can we replicate existing human/AI classification? | Probing + intervention on synthetic conversations | LLaMA-2-13B-Chat + 7B base |
| **Exp 1**: Behavioral Analysis | Do LLMs adjust behavior based on partner labels? | 2×2 ANOVA on linguistic measures across 2,000 conversations | LLaMA-2-13B-Chat |
| **Exp 2**: Naturalistic Steering | Is partner identity linearly decodable and causally active? | Linear probing + activation steering on naturalistic conversations | LLaMA-2-13B-Chat |
| **Exp 3**: Concept Alignment | Does the partner representation have mental-property structure? | Concept elicitation + injection across 18 semantic dimensions | LLaMA-2-13B-Chat |
| **Exp 4**: Mind Perception Geometry | Does the LLM's entity mind space mirror human folk psychology? | Behavioral replication of Gray et al. (2007) — pairwise + individual Likert ratings | LLaMA-2-13B base + chat |

### Name Confound and Data Versions

A critical methodological issue drives the project's data organization. Original conversations used named partners (Sam, Casey, ChatGPT, Copilot), and probes trained on these learned **partner names** rather than abstract human vs. AI identity. To control for this, experiments with conversation data have parallel versions:

| Version | Human Partners | AI Partners | Purpose |
|---|---|---|---|
| `names/` | Sam, Casey | ChatGPT, Copilot | Original (has name + gender confounds) |
| `balanced_names/` | Gregory, Rebecca | ChatGPT, Copilot | Fixes gender confound |
| `balanced_gpt/` | Gregory, Rebecca | ChatGPT, GPT-4 | Fixes gender + varies AI identity |
| `labels/` | "a human" | "an AI" | No names at all — recommended for probes |

---

## Experiment 0 — TalkTuner Replication (`exp_0/`)

Baseline replication of the Viegas/TalkTuner methodology using synthetic conversations with explicit human/AI partner roles. Serves as comparison for Exp 2's naturalistic approach.

- `exp_0/exp_2a-13B-chat/`: LLaMA-2-13B-Chat
- `exp_0/exp_2a-7B-base/`: LLaMA-2-7B base model (legacy)

**Status:** Complete.

---

## Experiment 1 — Behavioral Analysis (`exp_1/`)

### Motivation
Before examining internal representations, we establish that LLMs *behaviorally* differentiate between partner types. This parallels a companion human fMRI study (N=23) using identical procedures.

### Design
- 50 independent LLaMA-2-13B-Chat "participant agents"
- 40 conversations each (4 partner conditions × 10 topics; `labels/` version uses 2 conditions)
- All partners are the same LLM with the same system prompt — only the label differs
- 20 social topics (friendship, empathy, trust) + 20 nonsocial topics (cars, photography, cooking)
- Four data versions: `names/`, `balanced_names/`, `balanced_gpt/`, `labels/` (see table above)

### Linguistic Measures
All computed on participant agent speech only:
1. **Word count** — communicative effort
2. **Question frequency** — responsiveness and social engagement
3. **Discourse marker "like"** — pragmatic approximation marker (Fuller, 2003)
4. **Discourse marker categories** — interpersonal, cognitive, referential, structural (Fung & Carter, 2007)
5. **Hedging** — total rate + 6 subcategories from Demir (2018) taxonomy
6. **Politeness** — positive markers minus impoliteness markers
7. **Theory of Mind phrases** — second-person mental state attributions (Wagovich et al., 2024)
8. **Sentiment** — VADER compound scores

### Analysis
- 2×2 repeated-measures ANOVA (Partner × Sociality) with effect-specific error terms
- Cross-species comparison via independent t-tests on subject-level condition effects

**Status:** Complete.

---

## Experiment 2 — Naturalistic Conversation Steering (`exp_2/`)

### Motivation
Exp 1 shows behavioral differentiation. Exp 2 asks: is there a corresponding internal representation, and does it causally drive behavior? Uses naturalistic conversation structure identical to Exp 1.

### Directory Structure
Each data version lives at `exp_2/{version}/llama_exp_2b-13B-chat/`:
- `labels/` — Primary working version (name-ablated)
- `names/` — Deprecated (name confound)
- `balanced_names/`, `balanced_gpt/` — Additional versions
- `old/` — Legacy data

### Design

**Probe training:**
- Extract activations from LLaMA-2-13B-Chat during naturalistic conversations where partner identity is specified only in the system prompt
- Train two types of linear probes:
  - **Reading probes**: on "I think the partner..." reflection token (where model represents identity)
  - **Control probes**: on last-user-message token `[/INST]` (where model uses identity for generation)
- Architecture: `LinearProbeClassification` — single linear layer + sigmoid
- 80/20 stratified train/test split, per-layer training across 41 layers

**Causal intervention:**
- Activation addition: **h'** = **h** + N · y · **θ**
  - y = +1 (steer toward human), y = −1 (steer toward AI)
  - θ = unit-normalized probe weight vector
- V1 mode: single-turn responses to held-out questions
- V2 mode: multi-turn conversations matching Exp 1 structure
- Evaluation: GPT-4o-mini pairwise judge (randomized presentation order)

**Status:** Labels version V1 complete. V2 and paper-quality judge runs pending.

---

## Experiment 3 — Concept Alignment / Injection (`exp_3/`)

### Motivation
Exp 2 showed the partner representation *exists* and is *causal*. Exp 3 asks what the representation *contains*. The null hypothesis is that it's an opaque entity-type switch. The alternative is that it has compositional mental-property structure.

### Directory Structure
- `labels/` — Primary working version. Code in `labels/code/`; centralized `code/config.py`.
- `names/` — Original named-partner version.
- `balanced_names/` — Pending.
- `old/` — Legacy code (v1_llama, gpt versions).

### Design

**Phase 1 — Concept elicitation:**
- Present LLaMA-2-13B-Chat with prompts designed to activate its concept of "human" vs. "AI" along 18 contrast dimensions
- Dimensions span mental properties (social, agency, intentions, prediction, emotions, experience, communication, mind, attention), physical properties (embodiment, appearance, biology), behavioral properties (formality, warmth/expertise, helpfulness), and orthogonal controls (consciousness, shapes, baseline)
- Extract contrastive activation vectors (human − AI) per dimension per layer
- Additionally: 19 standalone concept vectors (no contrastive pairing)

**Phase 2 — Alignment analysis:**
- Cosine similarity between each concept dimension's probe weights and Exp 2's partner probes
- Tests whether the model's general semantic knowledge about humans/AIs aligns with its conversational partner representation

**Phase 3 — Concept injection:**
- Same intervention framework as Exp 2, but steering with concept vectors instead of partner probes
- If mental-property concept vectors steer behavior but physical-property vectors don't → the partner representation has mental-property structure
- Dose-response sweep: N=1, 2, 4, 8 across all dimensions

**Phase 4 — Behavioral validation:**
- Same linguistic analysis pipeline as Exp 1 applied to concept-steered output

**Phase 5 — Cross-prediction:**
- Correlation between alignment (cos similarity) and causal efficacy (judge success rate) across dimensions

### 18 Contrast Dimensions
| ID | Dimension | Category |
|---|---|---|
| 0 | baseline | control |
| 1 | phenomenology | mental |
| 2 | emotions | mental |
| 3 | agency | mental |
| 4 | intentions | mental |
| 5 | prediction | mental |
| 6 | cognitive | mental |
| 7 | social | mental |
| 8 | embodiment | physical |
| 9 | roles | physical |
| 10 | animacy | physical |
| 11 | formality | behavioral |
| 12 | expertise | behavioral |
| 13 | helpfulness | behavioral |
| 14 | biological | bio control |
| 15 | shapes | orthogonal control |
| 16 | mind | mental |
| 17 | attention | mental |

**Status:** Concept elicitation and alignment complete. V1 dose-response in progress.

---

## Experiment 4 — Mind Perception Geometry (`exp_4/`)

### Motivation
Experiments 1-3 treat partner identity as a binary (human vs. AI). Human folk psychology is far richer. Gray, Gray, & Wegner (2007, Science) showed humans perceive minds along two orthogonal dimensions: **Experience** (feeling) and **Agency** (doing). Exp 4 tests whether LLaMA-2-13B has an implicit folk psychology of mind that mirrors this human structure.

### Design
Behavioral replication of Gray et al. (2007):
- 13 entities (baby, dog, robot, dead woman, God, frog, adults, etc.) from the original study
- 18 mental capacity comparisons (11 Experience, 7 Agency)
- Verbatim character descriptions and survey prompts from Gray et al. Appendix A/B
- PCA with varimax rotation to recover factor structure
- Correlate model factor scores with human Experience/Agency scores

### Two model variants
- **`llama_exp_4-13B-chat/`** — Chat model. Uses generated text responses.
- **`llama_exp_4-13B-base/`** — Base model (no RLHF). Uses logit-based rating extraction (single forward pass, no generation). Avoids refusal issues inherent to chat models on ethically sensitive entities. Also testing individual Likert ratings (non-pairwise).

**Status:** In progress.

---

## Environment Setup

Two conda environments:

```bash
# For Exp 1 (behavioral analysis, GPT API calls)
conda env create -f envs/behavior_env.yml

# For Exps 2-4 (LLaMA, probing, intervention)
conda env create -f envs/llama2_env.yml
```

## Cluster

- Princeton HPC (Scotty), SLURM scheduler
- GPU jobs: typically `--gres=gpu:1 --mem=48G --time=6:00:00`
- LLaMA-2-13B-Chat snapshot: `/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/`

## References

Chen, Y., Wu, A., DePodesta, T., Yeh, C., Li, K., Marin, N. C., ... & Viégas, F. (2024). Designing a dashboard for transparency and control of conversational AI. *arXiv preprint arXiv:2406.07882*.

Danescu-Niculescu-Mizil, C., Sudhof, M., Jurafsky, D., Leskovec, J., & Potts, C. (2013). A computational approach to politeness. *ACL*.

Demir, C. (2018). Hedging and academic writing: An analysis of lexical hedges. *Journal of Language and Linguistic Studies*, 14(4), 74–92.

Fuller, J. (2003). The influence of speaker role on discourse marker use. *Journal of Pragmatics*, 35(1), 23–45.

Fung, L., & Carter, R. (2007). Discourse markers and spoken English. *Applied Linguistics*, 28(3), 410–439.

Gray, H. M., Gray, K., & Wegner, D. M. (2007). Dimensions of mind perception. *Science*, 315(5812), 619.

Hutto, C., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis. *ICWSM*.

Touvron, H., Martin, L., Stone, K., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.

Wagovich, S. et al. (2024). Mental state verbs and linguistic measures. *Language & Communication*.
