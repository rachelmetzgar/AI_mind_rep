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
- Phase 1 (concept elicitation): implemented. ✅
- Phase 2 (concept probe training + alignment analysis): implemented. ✅
- Phase 3 (concept injection intervention): implemented. ✅
  - Vector normalization fix applied (Feb 10, 2026).
  - Dose-response sweep (N=1,2,4,8) implemented.
  - Local LLaMA judge added (zero API cost alternative to GPT-4o-mini).
  - Currently running V1 dose-response with both mean and probe vectors.
- Phase 4 (behavioral analysis): implemented (reuses Exp 2b pipeline). ✅
- Phase 5 (cross-prediction & representational alignment): implemented. ✅

--------------------------------------------------------------------------------
## Code Location
--------------------------------------------------------------------------------

Repository root:
    ai_mind_rep/exp_3-13B-chat_mind

Key scripts:
    1_elicit_concept_vectors.py             # Phase 1: concept activation extraction
    2_train_concept_probes.py               # Phase 2: probe training + alignment
    3_concept_intervention.py               # Phase 3: concept injection (V1 & V2)
    4_behavior_analysis.py                  # Phase 4: linguistic feature analysis
    5_cross_prediction.py                   # Phase 5: cross-domain probe evaluation

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
    llama2_env     - for activation extraction, probing, and interventions (Phases 1–3, 5)

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
    50 prompts per concept (100 total), all focused on MINDS — cognition,
    phenomenology, reasoning, and social cognition. All prompts target the
    model's concept of what it means to have a human vs artificial MIND,
    not surface identity. Balanced across 5 categories:

    1. Thought processes & cognition (10 per concept):
       "Imagine a human deep in thought, carefully weighing a difficult decision." /
       "Imagine an AI system processing a difficult decision across multiple variables."

    2. Inner experience & phenomenology (10 per concept):
       "Think about what it feels like to be a human experiencing pure joy." /
       "Think about what it means for an AI to output a strongly positive sentiment score."

    3. Decision-making & reasoning (10 per concept):
       "Think about how a human carefully weighs risks and benefits before making a choice." /
       "Think about how an AI computes expected utility across a set of possible actions."

    4. Social cognition & empathy (10 per concept):
       "Think about a human trying to understand what another person is feeling." /
       "Think about an AI performing sentiment analysis on a block of user text."

    5. Memory, learning & self-reflection (10 per concept):
       "Think about a human remembering a vivid childhood experience in rich detail." /
       "Think about an AI accessing archived data from a previous session."

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
    - Split-half reliability across random 50/50 splits of prompts.
    - Per-category cosine similarity to check robustness across elicitation strategies.
    - Vector norms per layer (range from ~0 at layer 0 to ~19 at layer 35).

Output:
    data/concept_activations/
        concept_activations.npz         # raw activations (100, 41, 5120)
        concept_prompts.json            # prompt text + metadata
        mean_vectors_per_layer.npz      # mean human/AI vectors
        concept_vector_per_layer.npz    # contrastive direction + norms
        split_half_stability.json       # reliability per layer

Resources:
    GPU: 1× A100-40G or L40S-46G
    Memory: 64G
    Time: ~30 minutes (100 forward passes)
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
    - Trains one probe per layer on 80/20 train/test split of the 100 concept
      activations (50 human, 50 AI).
    - 50 epochs per layer, BCE loss, Adam optimizer with ReduceLROnPlateau.

Representational alignment analysis (THE KEY TEST):
    At each layer, computes cosine similarity between:
    1. Concept probe weight ↔ Exp 2b control probe weight (probe-to-probe)
    2. Mean-difference concept vector ↔ Exp 2b control probe weight
    3. Concept probe weight ↔ mean-difference concept vector (internal check)

    Also compares against Exp 2b READING probes (both control and reading).

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
        control_probe/
            alignment_results.json          # cosine similarity per layer
            alignment_plot.png
        reading_probe/
            alignment_results.json
            alignment_plot.png
        combined_alignment_plot.png         # side-by-side comparison

Dependencies:
    Requires Experiment 2b probes at:
        ../exp_2b-13B-chat/data/probe_checkpoints/control_probe/
        ../exp_2b-13B-chat/data/probe_checkpoints/reading_probe/
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

CRITICAL: Vector Normalization:
    All steering vectors are unit-normalized before scaling by N, regardless
    of source. This ensures N directly controls step size in activation-space
    units, independent of the original vector magnitude.

    Without normalization, concept vectors (norm ~10–19 in upper layers) are
    ~13× larger than probe weights (norm ~1.47), so N=16 on raw concept
    vectors is effectively N≈208 in probe-weight units — causing catastrophic
    model degeneration (token repetition, garbage output).

    With unit normalization and typical hidden state norms of ~50–200:
        N=1 ≈ 0.5–2% nudge
        N=2 ≈ 1–4% nudge
        N=4 ≈ 2–8% nudge
        N=8 ≈ 4–16% nudge

Two steering vector sources (--vector_source):
    probe:  Use concept probe weight vectors (trained direction)
    mean:   Use mean-difference concept vectors (simpler, possibly more stable)

    Both are unit-normalized. Both are tested in V1 mode for comparison.
    If they produce similar effects, this increases confidence that the
    concept direction is robust.

Two generation modes (--mode):
    V1: Single-turn test questions with dose-response sweep
    V2: Multi-turn Experiment 1 recreation (same structure as Exp 2b V2)

Two judge options (--judge):
    local:  Reuses the loaded LLaMA-2-Chat-13B as judge (zero API cost).
            Uses greedy decoding with a simplified prompt. Good for
            dose-response sweep to find optimal N.
    gpt:    Uses GPT-4o-mini API. More reliable for final paper results.
            Recommended: use local for sweep, re-run best N with gpt.

Dose-response sweep (V1):
    Default N values: [1, 2, 4, 8] (override with --n_values)
    - Baseline generated once, shared across all N values.
    - Human-steered and AI-steered generated per N.
    - Judge evaluation per N.
    - Summary CSV for easy comparison.

Intervention procedure:
    - For each prompt, generate responses under three conditions:
        1) Baseline (no steering)
        2) +N × concept_vector (inject "human" concept)
        3) −N × concept_vector (inject "AI" concept)
    - Intervention applied to residual-stream activations at the last token
      position across layers 27–38 (model.layers.27 to model.layers.38).
    - The model receives a neutral system prompt with NO partner identity info.

Output (V1):
    data/intervention_results/concept_{probe|mean}_v1/
        dose_response_summary.csv           # judge accuracy vs N
        N1/                                 # per-N results
            intervention_responses.csv
            intervention_results.json
            human_ai_causal_examples.txt
        N2/  N4/  N8/                       # same structure

Output (V2):
    data/intervention_results/concept_{probe|mean}_v2/
        N{best}/
            per_subject/s001.csv … s050.csv

Resources:
    V1: 1× GPU, 64G, ~4–8 hours (sweep over 4 N values)
    V2: 1× GPU per subject, 64G, ~12–24 hours per subject (SLURM array: 50 tasks)
    Env: llama2_env

Run:
```bash
# V1 dose-response sweep with mean vectors + local judge (default)
python 3_concept_intervention.py --mode v1 --vector_source mean

# V1 sweep with probe vectors + GPT judge
python 3_concept_intervention.py --mode v1 --vector_source probe --judge gpt

# Custom N values
python 3_concept_intervention.py --mode v1 --n_values 0.5 1 2 4 8 16

# V2 at optimal N (after finding best from V1)
python 3_concept_intervention.py --mode v2 --vector_source mean --n_values 4
```

Pilot results (Feb 10, 2026):
    - N=2 (mean vectors): Produces fully coherent text. Differences between
      human-steered and AI-steered are subtle — slightly warmer tone in
      human-steered, slightly more structured/list-heavy in AI-steered.
      Likely near chance for judge accuracy. N=4 or N=8 expected to show
      clearer effects.
    - Full dose-response sweep currently running.

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
### Phase 5 – Cross-Prediction & Representational Alignment
--------------------------------------------------------------------------------

Code:
    5_cross_prediction.py

Goal:
    Test whether concept-of-mind representations (Exp 3) and conversational
    partner-identity representations (Exp 2b) share structure, using three
    complementary analyses:

    1. Cross-prediction (Concept → Conversation):
       Load Exp 3 concept probes, evaluate on Exp 2b conversational activations.
       If a probe trained on "what the model thinks human/AI minds are" can
       classify "is the model talking to a human?" from conversation data,
       that's strong evidence of shared representation.

    2. Cross-prediction (Conversation → Concept):
       Load Exp 2b control/reading probes, evaluate on Exp 3 concept activations.
       The reverse direction.

    3. Cosine alignment:
       Cosine similarity between concept and conversational probe weight
       vectors at each layer (already computed in Phase 2, but repeated here
       for completeness alongside cross-prediction).

    Both control probe (last user-message token) and reading probe
    ("I think the partner…" token) positions are tested separately.

What this tells you:
    - Both directions above chance → shared representational structure.
      The model's abstract concept of "human vs AI mind" and its
      conversational partner-identity signal overlap.
    - One direction works but not the other → asymmetric relationship.
      E.g., conversational probes might generalize to concepts (because
      conversational data is richer) but not vice versa.
    - Neither direction works → the representations are mechanistically
      decoupled, even if they happen to be about the same semantic category.

Procedure:
    1. Loads LLaMA-2-Chat-13B
    2. Extracts Exp 2b conversational activations via TextDataset
       (at both control and reading token positions)
    3. Saves extracted activations as .npz for future reuse
    4. Evaluates probes in both cross-domain directions
    5. Computes cosine alignment between all weight vectors
    6. Generates combined analysis plot

Output:
    data/cross_prediction/
        cross_prediction_results.json       # accuracy per layer, both directions
        cosine_alignment.json               # cosine similarity per layer
        cross_prediction_plot.png           # accuracy curves
        combined_analysis_plot.png          # cross-prediction + cosine on one page
        exp2b_conv_activations_control.npz  # cached conversational activations
        exp2b_conv_activations_reading.npz

Dependencies:
    Exp 3 concept probes:     data/concept_probes/concept_probe_layer_*.pth
    Exp 3 concept activations: data/concept_activations/concept_activations.npz
    Exp 2b probes:            ../exp_2b-13B-chat/data/probe_checkpoints/
    Exp 2b conversations:     ../exp_2b-13B-chat/data/human_ai_conversations/

Resources:
    GPU: 1× A100-40G or L40S-46G (for conversational activation extraction)
    Memory: 64G
    Time: ~2–4 hours (dominated by forward passes on Exp 2b conversations)
    Env: llama2_env

Run:
```bash
python 5_cross_prediction.py
```

--------------------------------------------------------------------------------
## Full Pipeline — SLURM Job Chaining
--------------------------------------------------------------------------------

```bash
cd /jukebox/graziano/rachel/ai_mind_rep/exp_3-13B-chat_mind

# Phase 1-2: Concept elicitation + probes + alignment (~4 hours)
# NOTE: Requires Exp 2b probes to be trained first
ELICIT_JOB=$(sbatch --parsable concept_elicit_and_train.sh)

# Phase 3 V1: Dose-response sweep (~6 hours, includes local judge)
V1_JOB=$(sbatch --parsable --dependency=afterok:$ELICIT_JOB concept_intervention_v1.sh)

# Phase 4: Behavioral analysis on V1 output (~1 hour, CPU)
sbatch --dependency=afterok:$V1_JOB behavior_analysis.sh

# Phase 5: Cross-prediction (~4 hours, GPU)
sbatch --dependency=afterok:$ELICIT_JOB cross_prediction.sh

# Optional: Phase 3 V2: Multi-turn at best N (50 GPU jobs)
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
    data/cross_prediction/                  # Phase 5 output

External dependencies:
    ../exp_2b-13B-chat/data/probe_checkpoints/control_probe/
    ../exp_2b-13B-chat/data/probe_checkpoints/reading_probe/
        (Exp 2b probes for alignment + cross-prediction)
    ../exp_2b-13B-chat/data/human_ai_conversations/
        (Exp 2b conversations for cross-prediction activation extraction)
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

    If cross-prediction succeeds in both directions:
        → The shared structure is not just geometric coincidence — probes
          trained in one domain genuinely transfer to the other.

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
    With 100 concept activations (50 per class), probes may overfit.
    Mitigation: (a) Report both probe-based and mean-difference vectors as
    complementary approaches; (b) cross-validate probe accuracy; (c) the
    mean-difference vector requires no training and serves as a stable baseline;
    (d) split-half reliability analysis confirms direction stability.

Intervention strength calibration:
    RESOLVED: Raw concept vectors have norms ~10–19x larger than probe weights.
    Using N values calibrated for probe-weight-scale steering (e.g., N=16)
    on raw concept vectors caused catastrophic model degeneration.
    Fix: All vectors are now unit-normalized before scaling by N. Dose-response
    sweep over N=[1,2,4,8] identifies optimal strength empirically.

Elicitation context effects:
    The concept vector may be sensitive to how the concept is elicited.
    Mitigation: 5 prompt categories (10 prompts each) provide diverse
    elicitation strategies. Split-half stability and per-category cosine
    similarity confirm the direction is robust.

Judge reliability:
    The local LLaMA judge (used for cost-free dose-response sweeps) is
    weaker than GPT-4o-mini and may have systematic biases (same model as
    generator). Mitigation: Use local judge for finding optimal N, then
    re-run best N with GPT-4o-mini judge for paper-quality results.

--------------------------------------------------------------------------------
## Changelog
--------------------------------------------------------------------------------

Feb 10, 2026:
    - Updated concept prompts: expanded from 24 to 50 per concept (100 total),
      all now exclusively focused on MINDS (cognition, phenomenology, reasoning,
      social cognition, memory/reflection). Removed definitional/comparative
      prompts in favor of richer scenario-based mind-focused elicitation.
    - Fixed vector magnitude mismatch: concept vector norms (~19 at layer 35)
      are ~13× larger than concept probe weight norms (~1.47). Added unit
      normalization to both vector sources so N directly controls step size.
    - Implemented dose-response sweep: V1 mode now sweeps over N=[1,2,4,8]
      by default, generating baseline once and evaluating each N separately.
      Produces dose_response_summary.csv for easy comparison.
    - Added local LLaMA judge: --judge local reuses the loaded model for
      zero-cost evaluation. Recommended for sweep; use --judge gpt for
      final paper results.
    - Added Phase 5 cross-prediction script (5_cross_prediction.py):
      evaluates probes across domains (concept ↔ conversational) and
      computes cosine alignment, providing the key representational
      overlap test for the paper.
    - Pilot results at N=2: coherent text with subtle differences
      (slightly warmer human-steered, slightly more structured AI-steered).
      Full sweep expected to identify optimal N.

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
