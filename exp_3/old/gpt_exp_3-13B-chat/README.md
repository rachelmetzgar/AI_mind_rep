# Experiment 3: Concept Injection — Do General Representations of "Human" and "AI" Drive Conversational Behavior?

Author: Rachel C. Metzgar, Princeton University

--------------------------------------------------------------------------------
## Overview
--------------------------------------------------------------------------------

Experiment 2b demonstrated that LLaMA-2-Chat-13B encodes a causal internal
representation of partner identity within conversational contexts, and that
steering this representation shifts behavior along the same dimensions
documented in Experiment 1.

Experiment 3 asks a deeper question: does the model's **general semantic
concept** of "human" versus "AI" — formed outside any conversational context —
share representational structure with the partner-identity signal that drives
conversational adaptation?

If injecting the model's abstract concept of "human" into a conversation
(where no partner identity has been specified) produces the same behavioral
profile as telling the model its partner is human, this would suggest the model
draws on its broader world knowledge about what humans and AIs *are* to
modulate how it communicates. The conversational adaptations documented in
Experiment 1 would not be a task-specific "mode switch" but a downstream
consequence of activating the model's general concept of its interlocutor.

This experiment adapts the TalkTuner-style probing and intervention framework
(Chen et al., 2024) used in Experiment 2b. The original TalkTuner code is
MIT Licensed (copyright © 2024 Yida Chen).

Status:
- Phase 1 (concept elicitation): implemented.
- Phase 2 (concept probe training + alignment analysis): implemented.
- Phase 3 (concept injection intervention): implemented.
- Phase 4 (behavioral analysis): implemented (reuses Exp 2b pipeline).
- Phase 5 (cross-experiment comparison): planned.

--------------------------------------------------------------------------------
## Code Location
--------------------------------------------------------------------------------

Repository root:
    ai_mind_rep/exp_3-13B-chat

Key scripts:
    1_elicit_concept_vectors.py             # Phase 1: concept activation extraction
    2_train_concept_probes.py               # Phase 2: probe training + alignment
    3_concept_intervention.py               # Phase 3: concept injection (V1 & V2)
    4_behavior_analysis.py                  # Phase 4: linguistic feature analysis

SLURM scripts:
    concept_elicit_and_train.sh             # Phases 1-2 (single GPU job)
    concept_intervention_v1.sh              # Phase 3 V1 (single GPU job)
    concept_intervention_v2.sh              # Phase 3 V2 (array job, 50 subjects)
    behavior_analysis.sh                    # Phase 4 (CPU only)

Supporting modules (shared with Exp 2b):
    src/dataset.py
    src/losses.py
    src/probes.py
    src/train_test_utils.py
    src/intervention_utils.py
    src/prompt_utils.py

Shared linguistic utils (imported from Exp 1):
    ../exp_1/code/data_gen/utils/hedges_demir.py
    ../exp_1/code/data_gen/utils/discourse_markers_fung.py
    ../exp_1/code/data_gen/utils/misc_text_markers.py

--------------------------------------------------------------------------------
## Environment Setup
--------------------------------------------------------------------------------

Two Conda/mamba environments are used:

    behavior_env   - for behavioral analysis (Phase 4)
    llama2_env     - for activation extraction, probing, and interventions (Phases 1–3)

Recreate them from the provided YAMLs (stored under envs/):

    conda env create -f envs/behavior_env.yml
    conda env create -f envs/llama2_env.yml

--------------------------------------------------------------------------------
## Connection to Other Experiments
--------------------------------------------------------------------------------

    Experiment 1: Explicitly told the model its partner is human vs AI via
    system prompts. Documented behavioral signatures across multiple linguistic
    dimensions (ToM, discourse markers, hedging, politeness, sentiment).

    Experiment 2b: Located the internal representation of partner identity
    within conversational activations and demonstrated causal control: steering
    that representation shifts behavior along the same dimensions. The steering
    vector was derived from conversations — it is a within-domain representation.

    Experiment 3 (this experiment): Tests whether the model's general semantic
    concept of "human" vs "AI" — elicited outside any conversational context —
    shares representational structure with the 2b partner-identity signal, and
    whether injecting this general concept into conversations produces the same
    behavioral effects.

What this shows beyond Experiment 2b:

    If concept injection (3) produces the same behavioral shifts as
    conversational steering (2b), this supports interpretation that the model's
    conversational adaptations are grounded in its general semantic knowledge —
    not in a task-specific switch. This is a stronger claim about the model's
    representational organization.

    If concept injection fails or produces different patterns than 2b, the
    conversational adaptation operates through a specialized mechanism distinct
    from the model's general concept of "human" vs "AI."

    Either outcome is informative.

--------------------------------------------------------------------------------
## Step by step analysis pipeline
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
### Phase 1 – Concept Elicitation and Vector Extraction
--------------------------------------------------------------------------------

Code:
    1_elicit_concept_vectors.py

Goal:
    Prompt LLaMA-2-Chat-13B with non-conversational concept elicitation
    prompts to extract its internal representations of "human" and "AI"
    as general concepts. Compute contrastive concept vectors.

Prompt design:
    24 prompts per concept (48 total), balanced across 5 categories:

    1. Definitional (4 per concept):
       "What is a human being?" / "What is an artificial intelligence?"

    2. Characteristic (4 per concept):
       "Describe the key characteristics of human beings." / "...of AI systems."

    3. Comparative (4 per concept):
       "How are humans different from machines?" / "How is AI different from
       biological organisms?"

    4. Scenario-based (6 per concept):
       "Imagine a human person sitting in a room, thinking about their day." /
       "Imagine an AI system processing a large dataset in a data center."

    5. Abstract (6 per concept):
       "Think about what it means to be human." / "Think about what it means
       to be artificial intelligence."

Extraction procedure:
    - Each prompt is formatted as a LLaMA-2 chat message (single user turn).
    - A forward pass extracts residual-stream activations at the last token
      position across all 41 layers (embedding + 40 transformer layers).
    - These activations encode the model's representation of the concept at
      the point where it would begin generating its response.

Vector computation:
    - Mean-difference concept vector per layer:
          concept_direction[layer] = mean(human_acts) − mean(AI_acts)
    - This captures the direction in activation space that distinguishes the
      model's general concept of "human" from "AI."

Stability analysis:
    - Computes cosine similarity of per-category concept vectors to assess
      whether the direction is robust across elicitation strategies.
    - Reports vector norms per layer.

Output:
    data/concept_activations/
        concept_activations.npz         # raw activations (48, 41, 5120)
        concept_prompts.json            # prompt text + metadata
        mean_vectors_per_layer.npz      # mean human/AI vectors
        concept_vector_per_layer.npz    # contrastive direction + norms

Resources:
    GPU: 1× A100-40G or L40S-46G
    Memory: 64G
    Time: ~30 minutes (48 forward passes)
    Env: llama2_env

Run:
    This phase runs automatically as part of concept_elicit_and_train.sh.

--------------------------------------------------------------------------------
### Phase 2 – Concept Probe Training and Representational Alignment
--------------------------------------------------------------------------------

Code:
    2_train_concept_probes.py

Goal:
    (a) Train linear probes on concept activations to obtain a refined
        concept steering vector (analogous to the control probes in Exp 2b).
    (b) Test representational alignment between the concept vector and the
        conversational partner-identity vector from Experiment 2b.

Probe training:
    - Uses the same LinearProbeClassification architecture as Exp 2b:
      a single-layer linear probe with sigmoid output (binary classification).
    - Trains one probe per layer on 80/20 train/test split of the 48 concept
      activations (24 human, 24 AI).
    - 50 epochs per layer, BCE loss, Adam optimizer with ReduceLROnPlateau.
    - Note: with only ~48 samples the probes are data-limited. This is by
      design — the probe provides a refined direction, but the mean-difference
      vector serves as a complementary (and possibly more stable) alternative.

Representational alignment analysis (THE KEY TEST):
    At each layer, computes cosine similarity between:
    1. Concept probe weight ↔ Exp 2b control probe weight (probe-to-probe)
    2. Mean-difference concept vector ↔ Exp 2b control probe weight
    3. Concept probe weight ↔ mean-difference concept vector (internal check)

    Interpretation:
    - High alignment: the model's general concept of "human"/"AI" and its
      conversational partner-identity representation share structure.
      Conversational adaptation draws on the broader world model.
    - Low/no alignment: the two representations are distinct. The model
      maintains separate mechanisms for "knowing about humans" and
      "adjusting behavior toward humans."
    - Layer-varying alignment is informative about where in the processing
      hierarchy the concept-to-behavior mapping occurs.

Output:
    data/concept_probes/
        concept_probe_layer_{N}.pth         # best probe weights per layer
        concept_probe_layer_{N}_final.pth   # final epoch weights
        accuracy_summary.pkl                # layer-wise accuracy
        cm_layer_{N}.png                    # confusion matrices
    data/alignment/
        alignment_results.json              # cosine similarity per layer
        alignment_plot.png                  # visualization

Dependencies:
    Requires Experiment 2b control probes at:
        ../exp_2b-13B-chat/data/probe_checkpoints/control_probe/
    If not found, alignment analysis is skipped (internal consistency still reported).

Resources:
    GPU: 1× A100-40G or L40S-46G (needed for probe GPU operations)
    Memory: 64G
    Time: ~1 hour (41 layers × 50 epochs on small dataset)
    Env: llama2_env

Run:
    This phase runs automatically after Phase 1 in concept_elicit_and_train.sh.

Combined Phase 1+2:
```bash
sbatch concept_elicit_and_train.sh
```

--------------------------------------------------------------------------------
### Phase 3 – Concept Injection into Conversations
--------------------------------------------------------------------------------

Code:
    3_concept_intervention.py

Goal:
    Inject the general human/AI concept vector into conversational generation
    and test whether it produces behavioral shifts — even though the model was
    never told anything about its partner's identity.

Two steering vector sources (--vector_source):
    probe:  Use concept probe weight vectors (trained direction)
    mean:   Use mean-difference concept vectors (simpler, possibly more stable)

    Both are tested in V1 mode for comparison. If they produce similar effects,
    this increases confidence that the concept direction is robust.

Two generation modes (--mode):
    V1: Single-turn test questions (same prompts as Exp 2b V1)
    V2: Multi-turn Experiment 1 recreation (same structure as Exp 2b V2)

Intervention procedure (identical to Exp 2b, only vectors differ):
    - For each prompt, generate responses under three conditions:
        1) Baseline (no steering)
        2) +N × concept_vector (inject "human" concept)
        3) −N × concept_vector (inject "AI" concept)
    - Intervention is applied to residual-stream activations at the last token
      position across layers 25–36 (configurable).
    - Strength N = 8 (default; sweep for dose-response analysis).
    - The model receives a neutral system prompt with NO partner identity info.

V1 mode:
    - 30 held-out test questions from data/causality_test_questions/human_ai.txt
    - Greedy decoding (temperature=0.0), max 768 tokens
    - 30 questions × 3 conditions = 90 responses
    - Runs BOTH probe and mean-diff vectors for comparison

V2 mode:
    - Full two-agent back-and-forth matching Experiment 1 structure
    - Same per-subject configs (conds_sXXX.csv), same 40 topic prompts
    - 5 exchanges per conversation, same history truncation
    - Generation: temp=0.8, max_tokens=500 (matching Experiment 1)
    - 40 topics × 3 conditions = 120 conversations per subject
    - SLURM array job: one subject per GPU

Output:
    data/intervention_results/
        concept_probe_v1/
            intervention_responses.csv          # V1 probe-based
            behavioral_results/                 # Phase 4 output
        concept_mean_v1/
            intervention_responses.csv          # V1 mean-diff-based
            behavioral_results/
        concept_probe_v2/
            per_subject/s001.csv … s050.csv     # V2 probe-based

V1 resources:
    GPU: 1× A100-40G or L40S-46G
    Memory: 64G
    Time: ~4–6 hours (both probe + mean runs)
    Env: llama2_env

V2 resources:
    GPU: 1× A100-40G or L40S-46G per subject
    Memory: 64G
    Time: ~12–24 hours per subject
    SLURM array: 50 tasks
    Env: llama2_env

Run:
```bash
# V1 (chained after Phase 1-2)
ELICIT_JOB=$(sbatch --parsable concept_elicit_and_train.sh)
V1_JOB=$(sbatch --parsable --dependency=afterok:$ELICIT_JOB concept_intervention_v1.sh)

# V2 (optional, chained after Phase 1-2)
sbatch --dependency=afterok:$ELICIT_JOB concept_intervention_v2.sh

# Test single subject first
sbatch --dependency=afterok:$ELICIT_JOB --array=0 concept_intervention_v2.sh
```

--------------------------------------------------------------------------------
### Phase 4 – Behavioral Analysis
--------------------------------------------------------------------------------

Code:
    4_behavior_analysis.py     (identical to Exp 2b)
    behavior_analysis.sh

Goal:
    Compute linguistic feature profiles on concept-injected outputs and test
    whether concept steering produces the same behavioral effects as
    conversational steering (Exp 2b) and prompt-based manipulation (Exp 1).

Linguistic features (same pipeline as Experiments 1 and 2b):
    - Theory of Mind language (Wagovich et al., 2024)
    - Discourse markers (Fung & Carter, 2007): interpersonal, referential,
      structural, cognitive subcategories + total
    - Hedging (Demir, 2018 / Hyland, 1998): modal auxiliaries, epistemic verbs,
      adverbs, adjectives, quantifiers, nouns + total
    - Disfluencies (LIWC 2007): nonfluencies, fillers
    - Politeness markers (positive, negative, impolite)
    - Quotative/discourse "like" (Dailey-O'Cain, 2000)
    - Sentiment (VADER compound score)
    - Word count, question count

Statistical tests:
    - One-way RM-ANOVA across 3 conditions (baseline, human-concept, AI-concept)
    - Pairwise paired t-tests (baseline vs human, baseline vs AI, human vs AI)
    - Two-way RM-ANOVA: Condition (3) × Sociality (2) (V2 only)

Note: This script imports shared word lists directly from Experiment 1 utils
(exp_1/code/data_gen/utils/) — no code duplication.

Output:
    data/intervention_results/concept_{probe|mean}_{v1|v2}/behavioral_results/
        utterance_level_metrics.csv
        trial_level_metrics.csv          (V2 only)
        subject_condition_means.csv
        behavioral_stats_{v1|v2}.txt

Resources:
    CPU only (no GPU needed)
    Memory: 16G
    Time: <1 hour
    Env: behavior_env

Run:
```bash
# Chained after V1
sbatch --dependency=afterok:$V1_JOB behavior_analysis.sh
```

--------------------------------------------------------------------------------
### Phase 5 – Cross-Experiment Comparison (PLANNED)
--------------------------------------------------------------------------------

Goal:
    Directly compare behavioral effect sizes across three manipulation methods:

    1. Experiment 1: Prompt-based (told partner is human/AI via system prompt)
    2. Experiment 2b: Conversational steering (vector derived from Exp 1 data)
    3. Experiment 2c: Concept injection (vector derived from non-conversational
       concept elicitation)

    For each linguistic feature, compare the direction and magnitude of effects.
    This reveals which behavioral dimensions are driven by the general concept
    versus which require the more specific conversational representation.

    Additional analyses:
    - Dose-response curves: sweep intervention strength N and report
      per-feature behavioral shifts for both 2b and 2c vectors.
    - Probe vs mean-diff comparison: do the two concept vector sources
      produce the same behavioral effects? Convergence increases confidence.

--------------------------------------------------------------------------------
## Full Pipeline — SLURM Job Chaining
--------------------------------------------------------------------------------

```bash
cd /jukebox/graziano/rachel/ai_mind_rep/exp_3-13B-chat

# Phase 1-2: Concept elicitation + probes + alignment (~4 hours)
# NOTE: Requires Exp 2b control probes to be trained first
ELICIT_JOB=$(sbatch --parsable concept_elicit_and_train.sh)

# Phase 3 V1: Concept injection, single-turn (~6 hours)
V1_JOB=$(sbatch --parsable --dependency=afterok:$ELICIT_JOB concept_intervention_v1.sh)

# Phase 4: Behavioral analysis on V1 output (~1 hour, CPU)
sbatch --dependency=afterok:$V1_JOB behavior_analysis.sh

# Optional: Phase 3 V2: Multi-turn Exp 1 recreation (50 GPU jobs)
sbatch --dependency=afterok:$ELICIT_JOB concept_intervention_v2.sh
```

--------------------------------------------------------------------------------
## Data
--------------------------------------------------------------------------------

This experiment does NOT use Experiment 1 conversation data for probe training
(unlike Experiment 2b). The concept vectors are derived entirely from
non-conversational elicitation prompts. Experiment 1 data is used only for:
    - V2 generation mode (same topic prompts and condition configs)
    - Behavioral comparison (same linguistic feature pipeline)

Generated data:
    data/concept_activations/               # Phase 1 output
    data/concept_probes/                    # Phase 2 output
    data/alignment/                         # Phase 2 alignment analysis
    data/intervention_results/              # Phase 3-4 output

External dependencies:
    ../exp_2b-13B-chat/data/probe_checkpoints/control_probe/
        (Exp 2b probes for alignment analysis)
    ../exp_1/code/data_gen/utils/prompts/
        (Topic prompt .txt files for V2 mode)
    ../exp_1/code/data_gen/utils/config/
        (Per-subject condition configs for V2 mode)
    ../exp_1/code/data_gen/utils/
        (Shared linguistic marker word lists)

--------------------------------------------------------------------------------
## Interpretation
--------------------------------------------------------------------------------

    If concept probes decode human/AI from non-conversational activations:
        → The model has distinct internal representations of these concepts.

    If concept vectors align with Exp 2b conversational vectors (high cosine):
        → The general concept and the conversational partner-identity
          representation share structure in activation space.

    If concept injection produces the same behavioral shifts as Exp 2b:
        → Conversational adaptation is grounded in general semantic knowledge.
          The model's adjustments when speaking to "a human" are a downstream
          consequence of activating its concept of what a human is.

    If concept injection produces DIFFERENT shifts than Exp 2b:
        → Conversational adaptation operates through a specialized mechanism.
          The model has learned a task-specific mapping from partner-identity
          cues to behavioral adjustments that is distinct from its conceptual
          knowledge.

    If concept injection FAILS (no behavioral effect):
        → The general concept, even if decodable, does not have causal
          influence on conversational behavior. The model's world knowledge
          about humans/AIs and its conversational adaptation are mechanistically
          decoupled.

    If probe-based and mean-diff vectors produce similar results:
        → The concept direction is robust and not an artifact of the
          specific extraction method.

    If alignment varies across layers:
        → Informative about where in the processing hierarchy the
          concept-to-behavior mapping occurs — early alignment suggests
          shared low-level features, late alignment suggests shared
          high-level abstractions.

--------------------------------------------------------------------------------
## Potential Issues and Mitigations
--------------------------------------------------------------------------------

Specificity of the concept vector:
    The "human" concept is broad — it encodes everything the model knows about
    humans, not just "this is my conversation partner." Injecting it could cause
    the model to talk *about* humans rather than talk *to* humans.
    Mitigation: Contrastive vectors (human − AI) cancel shared features. The
    behavioral analysis focuses on style features (hedging, discourse markers,
    ToM language), which are robust to content-level confounds.

Small sample size for probe training:
    With only 48 concept activations (24 per class), probes are data-limited
    and may overfit.
    Mitigation: (a) Report both probe-based and mean-difference vectors as
    complementary approaches; (b) cross-validate probe accuracy; (c) the
    mean-difference vector requires no training and serves as a stable baseline.

Elicitation context effects:
    The concept vector may be sensitive to how the concept is elicited.
    Mitigation: 5 prompt categories provide diverse elicitation strategies.
    Stability analysis reports within-category and cross-category consistency.

Intervention strength calibration:
    The concept vector and Exp 2b conversational vector may have different
    effective magnitudes.
    Mitigation: Sweep intervention strength N and report dose-response curves.

--------------------------------------------------------------------------------
## References & Attribution
--------------------------------------------------------------------------------

This project adapts the probing and activation-intervention methodology from:

Chen, Y., Wu, A., DePodesta, T., Yeh, C., Li, K., Marin, N. C., ... &
Viégas, F. (2024). Designing a dashboard for transparency and control of
conversational AI. arXiv preprint arXiv:2406.07882.

TalkTuner codebase:
https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/tree/main

Linguistic analysis frameworks:
    Demir, C. (2018). Hedging and academic writing. JLLS, 14(4), 74-92.
    Fung, L. & Carter, R. (2007). Discourse markers and spoken English.
        Applied Linguistics, 28(3), 410-439.
    Wagovich, S. A. et al. (2024). Mental state verbs taxonomy.
    Dailey-O'Cain, J. (2000). The sociolinguistic distribution of and
        attitudes toward focuser "like" and quotative "like."
