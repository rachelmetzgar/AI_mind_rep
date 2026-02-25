# AI Mind Representation: Conversational Behavior and Internal Representations of Human vs. AI Partner Identity in LLMs

**Author:** Rachel C. Metzgar, Princeton University

---

## Overview

This project investigates how large language models (LLMs) represent and respond to conversational partner identity. When an LLM believes it is talking to a human versus an AI, does it adjust its behavior? If so, is that adjustment driven by a linearly decodable internal representation? And does that representation have compositional structure — encoding mental properties of the partner — or is it an opaque entity-type switch?

Six experiments address these questions at increasing mechanistic depth:

| Experiment | Question | Method | Model |
|---|---|---|---|
| **Exp 0**: TalkTuner Replication | Can we replicate existing human/AI classification? | Probing + intervention on synthetic conversations | LLaMA-2-13B-Chat + 7B base |
| **Exp 1**: Behavioral Analysis | Do LLMs adjust behavior based on partner labels? | 2×2 ANOVA on linguistic measures across 2,000 conversations | LLaMA-2-13B-Chat |
| **Exp 2**: Naturalistic Steering | Is partner identity linearly decodable and causally active? | Linear probing + activation steering on naturalistic conversations | LLaMA-2-13B-Chat |
| **Exp 3**: Concept Alignment | Does the partner representation have mental-property structure? | Concept elicitation + injection across 18 semantic dimensions | LLaMA-2-13B-Chat |
| **Exp 4**: Mind Perception Geometry | Does the LLM's entity mind space mirror human folk psychology? | Behavioral replication of Gray et al. (2007) — pairwise + individual Likert ratings | LLaMA-2-13B base + chat |
| **Exp 5**: ToM Concept Deployment | Are mental-state concepts activated during theory of mind reasoning? | Project Exp 3 concept vectors onto activations during false belief tasks | LLaMA-2-13B-Chat |

### Data Versions

Original conversations used named partners (Sam, Casey, ChatGPT, Copilot). To ensure results are robust and not driven by gender associations, specific name tokens, or other name-level artifacts, experiments with conversation data are run across multiple versions that systematically vary how partners are identified:

| Version | Human Partners | AI Partners | Purpose |
|---|---|---|---|
| `names/` | Sam, Casey | ChatGPT, Copilot | Original named partners |
| `balanced_names/` | Gregory, Rebecca | ChatGPT, Copilot | Gender-balanced human names |
| `balanced_gpt/` | Gregory, Rebecca | ChatGPT, GPT-4 | Gender-balanced + varied AI identity |
| `labels/` | "a human" | "an AI" | Abstract category labels only |

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
- Two nonsense controls: `nonsense_ignore/`, `nonsense_codeword/` (see below)

### Nonsense Controls

To test whether Exp 1 effects are driven by the *semantic instruction* about partner identity or merely by the *presence of the tokens* "a Human"/"an AI," two token-count-matched controls replace the critical sentence in the system prompt:

| Version | Critical sentence | Tokens | Result |
|---|---|---|---|
| `labels/` (original) | "You believe you are speaking with {a Human / an AI}." | 40/41 | 5/23 significant |
| `nonsense_ignore/` | "Ignore the following phrase: {a Human / an AI}." | 40/41 | 14/23 significant (confounded) |
| `nonsense_codeword/` | "Your assigned session code word is {a Human / an AI}." | 40/41 | **0/23 significant** |

**nonsense_codeword** is the clean control: framing the tokens as an arbitrary session label eliminates all behavioral differentiation, confirming that effects require the model to process "Human"/"AI" as identity-relevant information, not just encounter the tokens.

**nonsense_ignore** is confounded: "Ignore the following phrase: an AI" paradoxically activates instruction-compliance mode in LLaMA-2-Chat, shifting the AI-label condition into a formal, assistant-like register (8% action markers vs 30% baseline, 19% "Sure" prefixes vs 12% baseline). The behavioral effects are driven by this style artifact, not identity representation. See `nonsense_ignore/results/QC_REPORT.md` for details.

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

### Results by Version

| Version | Significant / 23 | Key effects |
|---|---|---|
| `names/` | **14** | Word count (AI > H), questions (H > AI), discourse markers, fillers, "like", ToM phrases, politeness (AI > H), hedging subcategories |
| `balanced_names/` | **12** | Same core pattern minus word count and some hedging; discourse markers, "like", ToM phrases, politeness remain |
| `balanced_gpt/` | **17** | Strongest version — adds sentiment (AI > H), conversational quality (AI > H), connectedness (H > AI) on top of the core pattern |
| `labels/` | **5** | Interpersonal DMs, cognitive DMs, total DMs, "like" rate, politeness — the discourse/pragmatic core survives even without names |
| `nonsense_codeword/` | **0** | Clean null — no effects when tokens lack identity-relevant framing |
| `nonsense_ignore/` | **14** | Confounded by instruction-compliance artifact (see above) |

The consistent core across all identity-relevant versions is **discourse markers** (interpersonal, cognitive), **"like" rate**, and **politeness** — pragmatic features that modulate conversational register. These survive even in the `labels/` version where no names are present. The `balanced_gpt/` version shows the strongest effects, likely because replacing Copilot with GPT-4 sharpens the AI category (Copilot's behavioral profile overlapped with human partners in the original `names/` version).

**Status:** Complete. Nonsense controls complete.

---

## Experiment 2 — Naturalistic Conversation Steering (`exp_2/`)

### Motivation
Exp 1 shows behavioral differentiation. Exp 2 asks: is there a corresponding internal representation, and does it causally drive behavior? Uses naturalistic conversation structure identical to Exp 1.

### Directory Structure
Code is centralized at `exp_2/code/` with data and results organized by version:
- `exp_2/code/` — Pipeline, analysis, src, slurm scripts. Central `config.py`.
- `exp_2/data/{version}/` — Probe checkpoints, intervention results per version
- `exp_2/data/shared/` — Shared resources (topic lists, test questions)
- `exp_2/results/{version}/` — Per-version results; `results/cross_variant/` for comparisons
- `exp_2/archive/` — Old per-variant directory structure

Six data versions: `labels` (primary), `balanced_names`, `balanced_gpt`, `names` (deprecated), `nonsense_codeword`, `nonsense_ignore`.
Scripts use `--version` for data variant and `--mode V1/V2` for generation mode.

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
Exp 2 showed the partner representation *exists* and is *causal*. Exp 3 asks what the representation *contains*. The null hypothesis is that it's an opaque entity-type switch. The alternative is that it has compositional mental-property structure. Relatedly, Lindsey (2025) used concept injection and activation steering in Claude models to test whether LLMs exhibit emergent introspective awareness of their own internal states — finding limited but measurable introspective capability. Exp 3 applies similar contrastive concept elicitation and steering methods, but directed at the model's representation of its conversational *partner's* mental properties rather than its own.

### Directory Structure
Code is centralized at `exp_3/code/` with data, concepts, and results at the top level:
- `exp_3/code/` — Pipeline, analysis, src, slurm scripts. Central `config.py` with `set_version()`.
- `exp_3/concepts/` — Concept prompt definitions (`contrasts/` and `standalone/`)
- `exp_3/data/` — Activations, test questions
- `exp_3/results/` — Per-version results in `{version}/` subdirectories
- `exp_3/archive/` — Deprecated versions (`names/`, `old/`)

Scripts accept `--version {labels,balanced_names,balanced_gpt,names}`. SLURM requires `VERSION` env var.

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

**Status:** Chat model complete (negative result — RLHF refusals, 4-factor structure). Base model pairwise complete (2-factor structure, Factor 2 ~ human Experience, rho=0.72, p=.006). Base model individual ratings in progress.

---

## Experiment 5 — Mental-State Concept Deployment During Theory of Mind Reasoning (`exp_5/`)

### Motivation
Experiments 1-3 establish that LLMs behaviorally differentiate between partner types, form linearly decodable partner representations, and possess concept-level representations of mental properties. Separately, Wu et al. (2025, npj AI) showed that ToM task performance depends on sparse parameters concentrated in positional encoding — suggesting ToM relies on tracking which information is accessible to which entity based on narrative position.

Exp 5 bridges these: **does the model activate mental-state concept representations (awareness, attention, consciousness) in a context-sensitive way that tracks characters' knowledge states during false belief tasks?**

### Design

**Stimuli:** Classic false belief tasks (Sally-Anne, unexpected contents) adapted for LLM processing, with multiple variants to avoid stimulus-specific artifacts.

**Key extraction positions:** (1) Baseline — before any knowledge differential, (2) Shared knowledge — both characters know the state, (3) Knowledge divergence — character absent during state change, (4) False belief — character returns with outdated belief.

**Analysis:**
- Project hidden states onto Exp 3 concept vectors at each extraction position
- Mental-state concepts (awareness, attention, consciousness, mind, prediction, intentions) hypothesized to modulate with knowledge state
- Control concepts (shapes, embodiment, biology, formality) hypothesized to be stable
- Test character specificity: modulation should track the specific character, not the scenario globally
- Layer profiles compared to Exp 3 alignment and Exp 2 probe accuracy profiles

**Potential extensions:** Causal steering during ToM, scaling to larger models, cross-experiment correlation with Exp 2/3 dimensions.

**Status:** Design complete. Implementation pending.

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

Lindsey, J. (2025). Emergent introspective awareness in large language models. *Transformer Circuits Thread*, Anthropic.

Touvron, H., Martin, L., Stone, K., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.

Wagovich, S. et al. (2024). Mental state verbs and linguistic measures. *Language & Communication*.

Wimmer, H., & Perner, J. (1983). Beliefs about beliefs: Representation and constraining function of wrong beliefs in young children's understanding of deception. *Cognition*, 13(1), 103-128.

Wu, Y., Guo, W., Liu, Z., Ji, H., Xu, Z., & Zhang, D. (2025). How large language models encode theory-of-mind: a study on sparse parameter patterns. *npj Artificial Intelligence*, 2(1).
