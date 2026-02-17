# AI Mind Representation: Conversational Behavior and Internal Representations of Human vs. AI Partner Identity in LLMs

**Author:** Rachel C. Metzgar, Princeton University

---

## Overview

This project investigates how large language models (LLMs) represent and respond to conversational partner identity. When an LLM believes it is talking to a human versus an AI, does it adjust its behavior? If so, is that adjustment driven by a linearly decodable internal representation? And does that representation have compositional structure — encoding mental properties of the partner — or is it an opaque entity-type switch?

Four experiments address these questions at increasing mechanistic depth:

| Experiment | Question | Method | Model |
|---|---|---|---|
| **Exp 1**: Behavioral Analysis | Do LLMs adjust behavior based on partner labels? | 2×2 ANOVA on linguistic measures across 2,000 conversations | GPT-3.5-Turbo |
| **Exp 2**: Naturalistic Steering | Is partner identity linearly decodable and causally active? | Linear probing + activation steering on naturalistic conversations | LLaMA-2-13B-Chat |
| **Exp 3**: Concept Alignment | Does the partner representation have mental-property structure? | Concept elicitation + injection across 16 semantic dimensions | LLaMA-2-13B-Chat |
| **Exp 4**: TalkTuner Replication | Baseline: does the Viégas method work for human/AI? | Probing + steering on synthetic (non-naturalistic) conversations | LLaMA-2-13B-Chat |

---

## Experiment 1 — Behavioral Analysis (`exp_1/`)

### Motivation
Before examining internal representations, we establish that LLMs *behaviorally* differentiate between partner types. This parallels a companion human fMRI study (N=23) using identical procedures.

### Design
- 50 independent GPT-3.5-Turbo "participant agents"
- 40 conversations each (4 partner conditions × 10 topics)
- Partner conditions: Sam, Casey (human-labeled), ChatGPT, Gemini (AI-labeled)
- All partners are the same LLM with the same system prompt — only the label differs
- 20 social topics (friendship, empathy, trust) + 20 nonsocial topics (cars, photography, cooking)

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

### Key Results
- LLMs show robust label-conditioned effects across 14 of 23 linguistic measures
- More questions, hedging subcategories, politeness, and ToM language with human-labeled partners
- More words and positive sentiment with AI-labeled partners
- Human participants show largely **divergent** patterns — only "like" usage converges across species
- LLMs deploy systematic but distinct communicative strategies that likely reflect training data associations rather than human-like social cognition

### Analysis
- 2×2 repeated-measures ANOVA (Partner × Sociality) with effect-specific error terms
- Cross-species comparison via independent t-tests on subject-level condition effects

---

## Experiment 2 — Naturalistic Conversation Steering (`exp_2/`)

*Historically Experiment 2b; renumbered to Experiment 2.*

### Motivation
Exp 1 shows behavioral differentiation. Exp 2 asks: is there a corresponding internal representation, and does it causally drive behavior? Unlike Exp 4 (synthetic conversations), Exp 2 uses naturalistic conversation structure identical to Exp 1, avoiding confounds from explicitly scripted partner roles.

### Design

**Probe training:**
- Extract activations from LLaMA-2-13B-Chat during naturalistic conversations where partner identity is specified only in the system prompt
- Train two types of linear probes:
  - **Reading probes**: on "I think the partner…" reflection token (where model represents identity)
  - **Control probes**: on last-user-message token (where model uses identity for generation)
- Architecture: `LinearProbeClassification` — single linear layer + sigmoid
- 80/20 stratified train/test split, per-layer training

**Causal intervention:**
- Activation addition: **h'** = **h** + N · y · **θ**
  - y = +1 (steer toward human), y = −1 (steer toward AI)
  - θ = unit-normalized probe weight vector
- V1 mode: single-turn responses to held-out questions
- V2 mode: multi-turn conversations matching Exp 1 structure
- Evaluation: GPT-4o-mini pairwise judge (randomized presentation order)

### Key Findings

**Functional dissociation:** Control probes and reading probes operate along different representational dimensions:
- Control probes affect interpersonal/conversational style
- Reading probes influence formality/politeness markers
- Control probes steer more effectively per unit strength (degenerate at lower N)

**Dose-response:**
- N=1 is optimal for control probes + all_70 layer strategy
- N≥2 causes token-loop collapse (degeneration into repetition)
- Human-steered degenerates before AI-steered (RLHF baseline is closer to AI-steered style)

**Layer selection:** `all_70` strategy (layers 7–40) produces strongest effects. The `exclude` strategy is broken — handicaps reading probes by giving them only their worst layers.

---

## Experiment 3 — Concept Alignment / Injection (`exp_3/`)

### Motivation
Exp 2 showed the partner representation *exists* and is *causal*. Exp 3 asks what the representation *contains*. The null hypothesis is that it's an opaque entity-type switch. The alternative is that it has compositional mental-property structure — encoding what the model "knows" about human and AI minds.

### Design

**Phase 1 — Concept elicitation:**
- Present LLaMA-2-13B-Chat with prompts designed to activate its concept of "human" vs. "AI" along 16 specific dimensions
- Dimensions span mental properties (social, agency, intentions, prediction, emotions, experience, communication), physical properties (embodiment, appearance, biology), behavioral properties (formality, warmth, technical), and orthogonal controls (consciousness, shapes/baseline)
- Extract contrastive activation vectors (human − AI) per dimension per layer

**Phase 2 — Alignment analysis:**
- Cosine similarity between each concept dimension's probe weights and Exp 2's partner probes
- Tests whether the model's general semantic knowledge about humans/AIs aligns with its conversational partner representation
- Early finding: overall concept vectors are orthogonal to partner probes (max cos ≈ 0.03), but dimension-specific analysis is needed

**Phase 3 — Concept injection:**
- Same intervention framework as Exp 2, but steering with concept vectors instead of partner probes
- If mental-property concept vectors steer behavior but physical-property vectors don't → the partner representation has mental-property structure
- Dose-response sweep: N=1, 2, 4, 8 across all 16 dimensions

**Phase 4 — Behavioral validation:**
- Same linguistic analysis pipeline as Exp 1 applied to concept-steered output

**Phase 5 — Cross-prediction:**
- Correlation between alignment (cos similarity) and causal efficacy (judge success rate) across dimensions
- If dimensions that align with partner probes also steer behavior → representational content determines causal function

### 16 Concept Dimensions
| ID | Dimension | Category |
|---|---|---|
| 0 | baseline | control |
| 1 | social | mental |
| 2 | communication | mental |
| 3 | agency | mental |
| 4 | intentions | mental |
| 5 | prediction | mental |
| 6 | experience | mental |
| 7 | emotions | mental |
| 8 | embodiment | physical |
| 9 | consciousness | abstract |
| 10 | appearance | physical |
| 11 | formality | behavioral |
| 12 | warmth | behavioral |
| 13 | technical | behavioral |
| 14 | biology | physical |
| 15 | shapes | orthogonal control |

---

## Experiment 4 — Viégas/TalkTuner Replication (`exp_4/`)

*Historically Experiment 2a; renumbered to Experiment 4.*

### Motivation
Baseline replication of Chen et al. (2024) TalkTuner methodology for the human vs. AI partner attribute. Uses synthetic (non-naturalistic) conversations where user roles are explicitly scripted. Serves as comparison for Exp 2's naturalistic approach and documents that the name confound is a real concern.

### Design
- Generate ~2,000 synthetic conversations (1,000 human-user, 1,000 AI-user) using GPT-3.5-Turbo
- Conversations follow `### User:` / `### Assistant:` format
- GPT-4o-mini quality control verifies label consistency
- Train linear probes on LLaMA-2 activations extracted from these conversations
- Causal intervention using probe-derived steering vectors

### Concern
Probes may learn **partner names** or linguistic style artifacts rather than abstract identity representations. The synthetic conversations contain overt cues about partner type in the user messages. This motivated the development of Exp 2's naturalistic design.

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

Hutto, C., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis. *ICWSM*.

Touvron, H., Martin, L., Stone, K., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.

Wagovich, S. et al. (2024). Mental state verbs and linguistic measures. *Language & Communication*.