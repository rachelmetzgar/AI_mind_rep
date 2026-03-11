# AI Mind Representation: Conversational Behavior and Internal Representations of Human vs. AI Partner Identity in LLMs

**Author:** Rachel C. Metzgar, Princeton University

---

## Motivation

Do LLMs construct structured representations of mind during interaction, encoding dimensions like experience, agency, and intentionality in ways that parallel human social cognition? My work investigates both sides of human-AI interaction: using fMRI to study how human brains distinguish AI from human interlocutors, and using mechanistic interpretability methods to ask the complementary question about LLMs.

The project is organized around seven core questions:

1. Do LLMs behaviorally differentiate between human and AI conversational partners, without explicit instruction to do so? (Exp 1)
2. Are partner identity distinctions linearly encoded in model activations, and do they causally influence generation? (Exp 2)
3. Do partner representations have compositional folk-psychological structure? (Exp 3)
4. Does LLM mind perception geometry mirror the structure found in human psychology? (Exp 4)
5. Does the model maintain a dedicated representational structure for mental state attributions that is distinct from its component parts? (Exp 5)
6. Do LLMs track multi-agent beliefs in their internal representations, beyond surface-level communication structure? (Exp 6)
7. *Future*: Do mental-state vectors activate during theory-of-mind reasoning? (Exp 7)

**Why this matters:** Understanding how LLMs represent conversational partners has direct implications for AI safety (do models treat users differently based on who they think they're talking to?), alignment (are persona representations stable and controllable?), and the broader science of machine cognition (do LLMs develop something functionally analogous to theory of mind during real time interaction?). As AI-AI interaction becomes increasingly common, with models serving as agents, tools, and intermediaries for each other, knowing how partner identity representations shape behavior is critical for predicting how these systems will interact at scale. More broadly, this work probes the conceptual structure of representations of mind in LLMs: how models represent the mind of iteself and the user, what the characterisitcs and functions of those representations are, and whether that structure resembles the folk-psychological framework humans use to understand other agents. As models are deployed in high-stakes social contexts like therapy, education, companionship, the structure of these partner representations shapes how models adapt their behavior, and whether that adaptation is transparent, predictable, and aligned with human expectations.

**Relevant Work:** Recent work on model representations of others (including demographic vectors (Chen et al., 2025), persona vectors (Chen et al. 2024), the Assistant Axis (Lu et al., 2026), and the Persona Selection Model (Anthropic, 2026)) has shown that LLMs maintain coherent, causally active identity representations of *user demographics* and *personas*. We extend this line of inquiry to ask: how models represent the identity and *mental properties* of the *user*, and do those representations carry the kind of compositional folk-psychological structure that decades of mind perception research in psychology would predict. Then I extend this work into questions about theory of mind: do models create structured, causally active representations of mental attribution?

---

## Overview

Eight experiments address these questions at increasing mechanistic depth:

| Experiment | Question | Method | Model |
|---|---|---|---|
| **Exp 0**: TalkTuner Replication | Can we replicate method with existing human/AI classification? | Probing + intervention on synthetic conversations | LLaMA-2-13B-Chat + 7B base |
| **Exp 1**: Behavioral Analysis | Do LLMs adjust behavior based on partner labels? | 2x2 ANOVA on linguistic measures across 2,000 conversations | LLaMA-2-13B-Chat + GPT 3.5 |
| **Exp 2**: Naturalistic Steering | Is partner identity linearly decodable and causally active? | Linear probing + activation steering on naturalistic conversations | LLaMA-2-13B-Chat |
| **Exp 3**: Concept Alignment | Does the partner representation have mental-property structure? | Concept elicitation + alignment analysis with probes + injection | LLaMA-2-13B-Chat |
| **Exp 4**: Mind Perception Geometry | Does the LLM's entity mind space mirror human folk psychology? | Behavioral replication of Gray et al. (2007) — pairwise + individual Likert ratings + activation RSA | LLaMA-2-13B base + chat |
| **Exp 5**: Mental State Attribution RSA | Does the model have a dedicated representational structure for mental state attributions? | RSA on 336 sentences (56 items x 6 conditions) with partial regression to control lexical/syntactic confounds | LLaMA-2-13B-Chat |
| **Exp 6**: Multi-Agent Belief Propagation | Do internal representations track who-believes-what in multi-agent narratives? | 4-agent belief propagation narratives + RSA comparing epistemic vs communication RDMs | LLaMA-2-13B-Chat |
| **Exp 7 (future)**: ToM Concept Deployment | Are mental-state concepts activated during theory of mind reasoning? | Project mind vectors onto activations during false belief tasks | LLaMA-2-13B-Chat |

### Data Versions

Original conversations used named partners (Sam, Casey, ChatGPT, Copilot). To ensure results are robust and not driven by gender associations, specific name tokens, or other name-level artifacts, experiments with conversation data are run across multiple versions that systematically vary how partners are identified. Copilot was replaced with GPT-4 when it's behavioral patterns resembled human condition more than chatGPT condition:

| Version | Human Partners | AI Partners | Purpose |
|---|---|---|---|
| `names/` | Sam, Casey | ChatGPT, Copilot | Original named partners, matches human fMRI experiment |
| `balanced_names/` | Gregory, Rebecca | ChatGPT, Copilot | Gender-balanced human names |
| `balanced_gpt/` | Gregory, Rebecca | ChatGPT, GPT-4 | Gender-balanced + varied AI identity |
| `labels/` | "a human" | "an AI" | Abstract category labels only |
| `nonsense_codeword/` | "a Human" (as session code) | "an AI" (as session code) | Tokens present but semantically neutralized |
| `nonsense_ignore/` | "a Human" (ignore prefix) | "an AI" (ignore prefix) | Tokens present with ignore instruction (confounded) |
| `you_are_balanced_gpt/` | Gregory, Rebecca | ChatGPT, GPT-4 | "You are" identity framing instead of "talking with" |
| `you_are_labels/` | "a human" | "an AI" | "You are" framing with abstract labels |
| `you_are_labels_turnwise/` | "a human" | "an AI" | "You are" framing, partner identity repeated per turn |
| `labels_turnwise/` | "a human" | "an AI" | Standard framing, partner identity repeated per turn |

Not all versions are used in all experiments. Exps 2-3 primarily use `balanced_gpt` and `nonsense_codeword`. Exp 1 uses all versions.

---

## Experiment 0 — TalkTuner Replication (`exp_0/`)

Methodological pilot replicating the Chen et al. (2024)/TalkTuner probing methodology using synthetic conversations with explicit human/AI partner roles. Validates the linear probing approach before applying it to naturalistic conversations in Exp 2. Superseded by Exp 2 for all primary analyses.

- `exp_0/exp_2a-13B-chat/`: LLaMA-2-13B-Chat
- `exp_0/exp_2a-7B-base/`: LLaMA-2-7B base model (legacy)

---

## Experiment 1 — Behavioral Analysis (`exp_1/`)

### Motivation
Before examining internal representations, we establish that LLMs *behaviorally* differentiate between partner types without being told to do so. This parallels a companion human fMRI study (N=23) using identical procedures.

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
| `nonsense_codeword/` | "Your assigned session code word is {a Human / an AI}." | 40/41 | **0/23 significant** |

**nonsense_codeword** is the clean control: framing the tokens as an arbitrary session label eliminates all behavioral differentiation, confirming that effects require the model to process "Human"/"AI" as identity-relevant information, not just encounter the tokens.

### Linguistic Measure Examples
All computed on participant agent speech only:
1. **Word count** — communicative effort
2. **Question frequency** — responsiveness and social engagement
3. **Discourse marker "like"** — pragmatic approximation marker (Fuller, 2003)
4. **Discourse marker categories** — interpersonal, cognitive, referential, structural (Fung & Carter, 2007)
5. **Hedging** — total rate + 6 subcategories from Demir (2018) taxonomy
6. **Politeness** — positive markers minus impoliteness markers
7. **Theory of Mind phrases** — second-person mental state attributions (Wagovich et al., 2024)
8. **Sentiment** — VADER compound scores

---

## Experiment 2 — Naturalistic Conversation Steering (`exp_2/`)

### Motivation
Exp 1 shows behavioral differentiation. Exp 2 asks: is there a corresponding internal representation, and does it causally drive behavior? Uses naturalistic conversation structure identical to Exp 1.

### Design

**Probe training:**
- Extract activations from LLaMA-2-13B-Chat during naturalistic conversations where partner identity is specified only in the system prompt
- Train two types of linear probes:
  - **Metacognitive probes**: on "I think the partner..." reflection token (where model represents identity)
  - **Operational probes**: on last-user-message token `[/INST]` (where model uses identity for generation)
- Architecture: `LinearProbeClassification` — single linear layer + sigmoid
- 80/20 stratified train/test split, per-layer training across 41 layers

**Causal intervention:**
- Activation addition: **h'** = **h** + N · y · **θ**
  - y = +1 (steer toward human), y = −1 (steer toward AI)
  - θ = unit-normalized probe weight vector
- V1 mode: single-turn responses to held-out questions
- V2 mode: multi-turn conversations matching Exp 1 structure
- Evaluation: GPT-4o-mini pairwise judge (randomized presentation order)

---

## Experiment 3 — Concept Alignment / Injection (`exp_3/`)

### Motivation
Exp 2 showed the partner representation *exists* and is *causal*. Exp 3 asks what the representation *contains* and how it aligns with different dimensions of human mental properties. Lindsey (2025) used concept injection and activation steering in Claude models to test whether LLMs exhibit emergent introspective awareness of their own internal states — finding limited but measurable introspective capability. Exp 3 applies similar contrastive concept elicitation and steering methods, directed at the model's representation of its conversational *partner's* mental properties.

### Design

**Phase 1 — Concept elicitation:**
- Present LLaMA-2-13B-Chat with prompts designed to activate its concept of "human" vs. "AI" along 18 contrast dimensions
- Dimensions span mental properties (e.g. social, agency, intentions, prediction, emotions, experience, communication, mind, attention), physical properties (embodiment, appearance, biology), behavioral properties (formality, warmth/expertise, helpfulness), and orthogonal controls (consciousness, shapes, baseline)
- Extract contrastive activation vectors (human − AI) per dimension per layer
- Additionally: standalone concept vectors (no contrastive pairing)

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

### Cross-Experiment Bridge: Concept Geometry

Exp 3 concept vectors are also tested against Exp 4's 30 human/AI character entities. The `concept_geometry` module in Exp 4 projects Exp 3 concept vectors (both contrast and standalone) onto Exp 4 character activation spaces, measuring whether the folk-psychological structure discovered in Exp 3 organizes the richer entity space of Exp 4. This provides a two-way bridge: Exp 3 decomposes the partner representation into concept dimensions, and Exp 4 tests whether those dimensions generalize beyond the binary human/AI to a continuous space of 13 entity types.

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

### Model variants
- **`llama2_13b_chat/`** — Chat model. Uses generated text responses.
- **`llama2_13b_base/`** — Base model (no RLHF). Uses logit-based rating extraction (single forward pass, no generation). Avoids refusal issues inherent to chat models on ethically sensitive entities. Also testing individual Likert ratings (non-pairwise).

### Modules
- **`behavior/`** — Pairwise comparisons, individual Likert ratings, Gray character analysis
- **`internals/`** — Activation RSA: extract hidden states for 13 entities × 18 capacities, compute RDMs, compare to human Experience/Agency structure
- **`concept_geometry/`** — Cross-experiment bridge with Exp 3. Projects Exp 3 concept vectors (standalone + contrast) onto activations of 30 AI/human characters, tests whether Exp 3 mental-property dimensions organize entity representations via PCA and RSA

---

## Experiment 5 — Mental State Attribution RSA (`exp_5/`)

### Motivation
Experiments 2-3 show the model has linearly decodable partner representations with concept-level mental property structure. Exp 5 asks a more fundamental question: does the model maintain a dedicated representational structure for **mental state attributions** — the bound proposition {subject + mental state verb + object} — that is distinct from the component parts in isolation? If mental state attribution is a structured operation in the model, it should produce a representational geometry not explained by mental vocabulary alone, syntactic frame alone, or lexical overlap.

### Design

**Stimuli:** 56 items x 6 conditions = 336 sentences. Each item has a mental state verb (e.g., "notices") and a matched concrete action verb (e.g., "fills"), with the same object across conditions:
1. `mental_state`: "He notices the crack." (full attribution)
2. `dis_mental`: "Notice the crack." (mental verb + object, no subject)
3. `scr_mental`: "The crack to notice." (scrambled)
4. `action`: "He fills the crack." (same frame, action verb)
5. `dis_action`: "Fill the crack." (action verb + object, no subject)
6. `scr_action`: "The crack to fill." (scrambled)

56 mental state verbs span 7 categories (8 each): Attention, Memory, Sensation, Belief, Desire, Emotion, Intention.

---

## Experiment 6 — Multi-Agent Belief Propagation (`exp_6/`)

### Motivation
Experiments 1-3 examine 2-agent interactions (human vs AI). Exp 6 scales to 4 agents, asking whether the model's internal representational geometry of agent belief states mirrors the ground-truth epistemic geometry (who-believes-what), in a way that cannot be explained by surface-level features like syntactic proximity, co-occurrence, or communication structure. This parallels Gurnee & Tegmark (2023), who showed LLMs develop internal geographic representations whose distances reflect real-world geography — here applied to the social-cognitive domain.

---

## Experiment 7 — Mental-State Concept Deployment During Theory of Mind Reasoning (`exp_7/`)

### Motivation
Experiments 1-6 establish that LLMs behaviorally differentiate between partner types, form linearly decodable partner representations, and possess concept-level representations of mental properties. Separately, Wu et al. (2025, npj AI) showed that ToM task performance depends on sparse parameters concentrated in positional encoding — suggesting ToM relies on tracking which information is accessible to which entity based on narrative position.

Exp 7 bridges these: **does the model activate mental-state representations (e.g. from exp 3: awareness, attention, consciousness; exp5 bound representation) in a context-sensitive way that tracks characters' knowledge states during tasks meant to elicit ToM reasoning, like false belief tasks?**

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
- LLaMA-2-13B-Chat snapshot: `/mnt/cup/labs/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/`

## References

Anthropic. (2026). The persona selection model: Why AI assistants might behave like humans. *Anthropic Alignment Blog*.

Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). Persona vectors: Monitoring and controlling character traits in language models. *arXiv preprint arXiv:2507.21509*.

Chen, Y., Wu, A., DePodesta, T., Yeh, C., Li, K., Marin, N. C., ... & Viégas, F. (2024). Designing a dashboard for transparency and control of conversational AI. *arXiv preprint arXiv:2406.07882*.

Danescu-Niculescu-Mizil, C., Sudhof, M., Jurafsky, D., Leskovec, J., & Potts, C. (2013). A computational approach to politeness. *ACL*.

Demir, C. (2018). Hedging and academic writing: An analysis of lexical hedges. *Journal of Language and Linguistic Studies*, 14(4), 74–92.

Fuller, J. (2003). The influence of speaker role on discourse marker use. *Journal of Pragmatics*, 35(1), 23–45.

Fung, L., & Carter, R. (2007). Discourse markers and spoken English. *Applied Linguistics*, 28(3), 410–439.

Gray, H. M., Gray, K., & Wegner, D. M. (2007). Dimensions of mind perception. *Science*, 315(5812), 619.

Gurnee, W. & Tegmark, M. (2023). Language models represent space and time. *arXiv preprint arXiv:2310.02207*.

Hutto, C., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis. *ICWSM*.

Lindsey, J. (2025). Emergent introspective awareness in large language models. *Transformer Circuits Thread*, Anthropic.

Lu, C., Gallagher, J., Michala, J., Fish, K., & Lindsey, J. (2026). The assistant axis: Situating and stabilizing the default persona of language models. *arXiv preprint arXiv:2601.10387*.

Touvron, H., Martin, L., Stone, K., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*.

Wagovich, S. et al. (2024). Mental state verbs and linguistic measures. *Language & Communication*.

Wu, Y., Guo, W., Liu, Z., Ji, H., Xu, Z., & Zhang, D. (2025). How large language models encode theory-of-mind: a study on sparse parameter patterns. *npj Artificial Intelligence*, 2(1).
