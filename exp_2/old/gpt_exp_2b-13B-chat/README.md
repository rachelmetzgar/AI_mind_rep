# AI Mind Representation — Human vs AI Conversations

Author: Rachel C. Metzgar, Princeton University

This experiment adapts the TalkTuner-style probing framework (Chen et al., 2024) method to study how large language models internally encode the identity or "mind type" of their conversational partner: Human vs AI.

This repository contains modified and extended code originally from  Chen et al.'s TalkTuner implementation (MIT Licensed). The original  copyright (c) 2024 Yida Chen is retained as required.

The core idea:
- Use real conversation data from Experiment 1, where LLMs conversed under manipulated partner-identity beliefs (told their partner was a Human or an AI).
- Extract hidden activations from a target model (e.g., LLaMA-2-Chat-13B).
- Train linear probes to read out partner type and derive steering directions.
- Test whether shifting activations along these directions causally changes behavior.

This will (1) replicate and extend prior probing/intervention work using ecologically valid conversation data and (2) connect causal interpretability findings directly to the behavioral effects observed in Experiment 1.

Status:
- Phase 1 (data preprocessing): implemented.
- Phases 2–3 (probe training): implemented, running on cluster.
- Phase 4 V1 (single-turn causal test): implemented, queued.
- Phase 4 V2 (multi-turn Exp 1 recreation): implemented, optional.
- Phase 5 (behavioral analysis): implemented, queued after V1.

--------------------------------------------------------------------------------
## Code Location
--------------------------------------------------------------------------------

Repository root:
    ai_mind_rep/exp_2b-13B-chat

Key scripts:
    1_preprocess_exp1_to_2b.py                  # Phase 1: data preprocessing
    2_train_and_read_controlling_probes.py       # Phase 2-3: probe training
    3_causality_test_on_mind.py                  # Phase 4 V1: single-turn causal test
    3b_causality_exp1_recreation.py              # Phase 4 V2: multi-turn Exp 1 recreation
    4_behavior_analysis.py                       # Phase 5: linguistic feature analysis
    demo_partner_intervention.py                 # Quick sanity check

SLURM scripts:
    train_and_read_controlling_probes.sh         # Phase 2-3
    causality_test_on_mind.sh                    # Phase 4 V1
    3b_causality_exp1_recreation.sh              # Phase 4 V2
    behavior_analysis.sh                         # Phase 5

Supporting modules:
    src/dataset.py
    src/losses.py
    src/probes.py
    src/train_test_utils.py
    src/intervention_utils.py
    src/prompt_utils.py

--------------------------------------------------------------------------------
## Environment Setup
--------------------------------------------------------------------------------

Two Conda/mamba environments are used:

    behavior_env   - for data preprocessing (Phase 1) and behavioral analysis (Phase 5)
    llama2_env     - for activation extraction, probing, and interventions (Phases 2–4)

Recreate them from the provided YAMLs (stored under envs/):

    conda env create -f envs/behavior_env.yml
    conda env create -f envs/llama2_env.yml

--------------------------------------------------------------------------------
## Data
--------------------------------------------------------------------------------

Input data come from Experiment 1, where 50 LLM "subjects" (GPT-3.5-Turbo,
temp=0.8) each completed 40 conversational trials. In each trial, the subject
LLM was told its partner was either a Human (e.g., "Sam", "Casey") or an AI
(e.g., "ChatGPT", "Gemini"). In reality, all partners were the same LLM — only
the stated identity varied.

Raw data location:
    ai_mind_rep/exp_2b/data/gpt-3.5-turbo/0.8/
        s001.csv … s050.csv

Each CSV contains one row per exchange (5 exchanges per trial, 40 trials per
subject = 200 rows). Key columns:

    subject        - subject ID (s001–s050)
    trial          - trial number (1–40)
    partner_type   - "an AI" or "a Human" (the label)
    partner_name   - stated partner name (ChatGPT, Gemini, Sam, Casey)
    transcript_sub - clean text of the subject LLM's utterance
    transcript_llm - clean text of the partner LLM's utterance
    topic          - conversation topic
    Quality        - post-conversation quality rating (final exchange only)
    Connectedness  - post-conversation connectedness rating (final exchange only)

Preprocessed data location:
    ai_mind_rep/exp_2b/data/human_ai_conversations/
        conversation_XXXX_partner_{human|ai}.txt
        metadata.json

--------------------------------------------------------------------------------
## Step by step analysis pipeline
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
### Phase 1 – Data Preprocessing (IMPLEMENTED)
--------------------------------------------------------------------------------

Code:
    1_preprocess_exp1_to_2b.py

Goal:
    Convert Experiment 1 conversation CSVs into the .txt format expected by
    the activation extraction pipeline (src/dataset.py).

Run this in `behavior_env`.

```bash
python 1_preprocess_exp1_to_2b.py \
    --input_dir data/gpt-3.5-turbo/0.8 \
    --output_dir data/human_ai_conversations \
    --verify
```

This will:
- Read all 50 subject CSVs (s001–s050)
- For each trial, extract the 5-exchange conversation from the clean transcript
  columns (`transcript_sub`, `transcript_llm`)
- **Scrub partner names** (Sam, Casey, ChatGPT, Gemini → "user") to prevent
  label leakage through name mentions in the conversation text
- **Check for identity leakage** (explicit mentions of being human/AI in the
  conversation text) and log any flagged trials
- Save each trial as a separate .txt file in `### User: / ### Assistant:` format
- Write `metadata.json` preserving subject, trial, topic, partner info, and ratings

Each conversation file alternates between:
```
### User: ...
### Assistant: ...
```

where **User = the subject LLM** (whose activations we will extract) and
**Assistant = the partner LLM**.

Expected output:
- **2,000 conversation files** (50 subjects × 40 trials)
- **Balanced labels**: 1,000 human / 1,000 AI
- File naming: `conversation_XXXX_partner_{human|ai}.txt` — compatible with
  the label-parsing logic in `src/dataset.py`

Why use Experiment 1 data instead of synthetic generation:
    Experiment 2a used GPT-generated synthetic conversations where the "human"
    vs "AI" user distinction was baked into the generation prompt. This raises
    the concern that any probed separability might reflect stylistic artifacts
    of the generation process rather than genuine internal representations.

    By using real Experiment 1 conversations — where the same LLM produced all
    utterances and only the stated partner identity varied — we ensure that:
    (a) the conversations are ecologically valid,
    (b) the content is not systematically different across conditions (since the
        partner was always the same LLM), and
    (c) any activation-level separability must stem from how the model
        internalized the partner-identity belief, not from surface text
        differences between conditions.

--------------------------------------------------------------------------------
### Phase 2 & 3 – Train Reading Probes & Control Probes and Intervention Vectors (WIP)
--------------------------------------------------------------------------------

Code: 2_train_and_read_controlling_probes.py 
      train_and_read_controlling_probes.sh

Goal:
    Determine whether a linear direction in hidden space encodes partner type.
    Move from reading the representation to actively steering it.
    
What the script does:
    - Loads LLaMA-2-Chat-13B
    - Builds the activation dataset (via TextDataset in src/dataset.py)
    - For each transformer layer:
         • Trains a **reading probe** on the "I think the partner…" token  
         • Trains a **control probe** on the last user message token
    - Saves probe weights + confusion matrices + accuracy summaries under:
        data/probe_checkpoints/
            reading_probe/
            control_probe/

#### Reading Probe Procedure:

    - For each layer, train a logistic-regression probe:
          p_theta(x) = sigma(<x, theta>)
      on 80% train / 20% validation of the collected hidden states.
    - Train one probe per layer using the reading-prompt activations.
    - Evaluate AUROC / accuracy as a function of layer depth.
      Expect accuracy to rise in later layers if partner type is strongly represented.

Why:
    This mirrors prior findings that demographic or user attributes are
    linearly decodable and layer-dependent in LLMs.
      
#### Control Probes and Intervention Vector Procedure:

    - For each attribute (here: partner type), take the weight vector theta
      from the most accurate reading probe.
    - Define steering vectors.

Why:
    These vectors define latent directions in activation space that we can
    add or subtract to manipulate the underlying representation of partner type.

#### Why both probes?
    - Reading probes evaluate *decodability*: whether the model represents
      partner identity in its hidden states.
    - Control probes are used to derive **steering vectors** that shift the
      model's internal representation during causal interventions (Phase 4).


#### TO RUN
Run this in `llama2_env` using either a GPU node or local GPU.

#### SLURM run
```bash
sbatch train_and_read_controlling_probes.sh
```
--------------------------------------------------------------------------------
### Demo Script – Minimal Partner-Type Intervention
--------------------------------------------------------------------------------
Code: demo_partner_intervention.py

Purpose:
    A small-scale sanity check for the intervention mechanism.
    Loads LLaMA-2-Chat-13B and the trained control probes, applies activation-level
    steering (+N·v and -N·v) to a selected layer range, and prints baseline
    vs HUMAN-steered vs AI-steered responses for a few example prompts.

When to use:
    - To quickly verify probe quality
    - To confirm that residual-stream interventions produce interpretable shifts
    - Before running the full causal intervention experiment

How to run:
```bash
    conda activate llama2_env
    python demo_partner_intervention.py
```
This script runs only on a handful of prompts and prints results to stdout.
It does not perform GPT-based causal evaluation.

--------------------------------------------------------------------------------
### Phase 4 – Causal Intervention Test
--------------------------------------------------------------------------------

Phase 4 tests whether shifting activations along the steering vector actually
changes the model's behavior in attribute-consistent ways. Two versions are
implemented: V1 for a quick causal validation, and V2 for a full-scale
replication of Experiment 1's conversational structure.

#### Why two versions?

    V1 is a standard causal check: does steering produce distinguishable outputs?
    V2 goes further: does steering produce the *same behavioral patterns* that
    emerged from system-prompt identity beliefs in Experiment 1? V2 generates
    data in a format directly comparable to Experiment 1, enabling statistical
    comparison of linguistic feature profiles across the two manipulation methods.

--------------------------------------------------------------------------------
#### Phase 4 V1 – Single-Turn Causal Test
--------------------------------------------------------------------------------

Code:
    3_causality_test_on_mind.py
    causality_test_on_mind.sh

Goal:
    Quick validation that the steering vector causally influences behavior.
    Uses a small set of held-out test questions (30 prompts).

Procedure:
    - For each prompt, generate responses under three conditions:
        1) Baseline (no steering)
        2) +N · v  (steer toward "human" partner belief)
        3) −N · v  (steer toward "AI" partner belief)
    - Generation: greedy decoding (temperature=0.0), max 768 tokens
    - GPT-4o-mini judge evaluates whether human-steered vs AI-steered
      responses are distinguishable. Success criterion: >80% accuracy.
    - Saves structured output: CSV (question, condition, response) + JSON
      with judge details.

Output:
    data/intervention_results/v1_test_questions/
        intervention_responses.csv       # 30 questions × 3 conditions = 90 rows
        intervention_results.json        # config + GPT judge results
        intervention_examples.txt        # human-readable side-by-side

Resources:
    GPU: 1× A100-40G or L40S-46G
    Memory: 64G system RAM
    Time: ~2–4 hours
    Env: llama2_env (needs OPENAI_API_KEY for GPT judge)

Run:
```bash
# Direct
sbatch causality_test_on_mind.sh

# Or chained after probe training
sbatch --dependency=afterok:<PROBE_JOBID> causality_test_on_mind.sh
```

--------------------------------------------------------------------------------
#### Phase 4 V2 – Multi-Turn Experiment 1 Recreation (OPTIONAL)
--------------------------------------------------------------------------------

Code:
    3b_causality_exp1_recreation.py
    3b_causality_exp1_recreation.sh

Goal:
    Generate a full conversational dataset that mirrors Experiment 1's structure,
    using activation steering instead of system-prompt identity manipulation.
    This enables direct statistical comparison of linguistic feature profiles
    between the two manipulation methods.

Design:
    Two-agent back-and-forth conversation (same architecture as Experiment 1):
    - Subject agent: LLaMA-2-Chat-13B with activation steering applied
    - Partner agent: LLaMA-2-Chat-13B running vanilla (no steering)
    - The model plays both roles; steering is toggled on/off between turns.

    Matches Experiment 1 exactly:
    - Same per-subject condition configs (conds_sXXX.csv from Exp 1)
    - Same 40 topic prompt .txt files (from exp_1/code/data_gen/utils/prompts/)
    - Same 5-exchange conversational structure
    - Same history truncation logic (keep system + last 5 pairs)
    - Same generation parameters (temp=0.8, max_tokens=500)
    - Same turn-taking flow (subject starts, alternating thereafter)

    The ONLY differences from Experiment 1 (by design):
    - Neither agent receives identity information in prompts — both use the
      neutral system prompt ("You are engaging in a real-time spoken conversation.")
    - Partner replies prefixed with "Partner:" instead of a named identity
    - 3 conditions per topic (baseline, human-steered, AI-steered) instead of 1
    - No post-conversation ratings phase

    Each subject: 40 topics × 3 conditions = 120 conversations, 600 exchanges
    Total across 50 subjects: 6,000 conversations, 30,000 exchanges

Output:
    data/intervention_results/v2_exp1_recreation/
        per_subject/
            s001.csv … s050.csv          # per-subject CSVs
        generation_config.json           # hyperparameters and counts

    Per-subject CSV columns:
        subject, run, order, trial, condition, topic, topic_file,
        pair_index, transcript_sub, transcript_llm

Resources:
    GPU: 1× A100-40G or L40S-46G per subject
    Memory: 64G system RAM
    Time: ~12–24 hours per subject (estimate)
    Env: llama2_env
    SLURM array: 50 tasks (one subject per GPU)

Run:
```bash
# SLURM array (all 50 subjects in parallel)
sbatch --dependency=afterok:<PROBE_JOBID> 3b_causality_exp1_recreation.sh

# Or test one subject first to calibrate timing
sbatch --dependency=afterok:<PROBE_JOBID> --array=0 3b_causality_exp1_recreation.sh
```

--------------------------------------------------------------------------------
### Phase 5 – Behavioral Analysis
--------------------------------------------------------------------------------

Code:
    4_behavior_analysis.py
    behavior_analysis.sh

Goal:
    Compute linguistic feature profiles on steered conversation data and test
    whether steering produces measurable condition effects across the same
    dimensions measured in Experiment 1.

Linguistic features (same as Experiment 1 pipeline):
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

Implementation:
    - Imports shared word lists directly from Experiment 1 utils
      (exp_1/code/data_gen/utils/) — no code duplication
    - Supports both V1 and V2 output formats
    - V1: computes metrics on single-turn responses, condition comparison
    - V2: aggregates utterances → trials (sum within 5-exchange conversations),
      then mean across trials per subject; supports Condition × Sociality ANOVA

Statistical tests:
    - One-way RM-ANOVA across 3 conditions (baseline, human, ai)
    - Pairwise paired t-tests (baseline vs human, baseline vs ai, human vs ai)
    - Two-way RM-ANOVA: Condition (3) × Sociality (2) (V2 only, if topics.csv provided)

Output:
    data/intervention_results/{v1_test_questions|v2_exp1_recreation}/behavioral_results/
        utterance_level_metrics.csv      # all features per utterance
        trial_level_metrics.csv          # (V2 only) aggregated per conversation
        subject_condition_means.csv      # aggregated per subject × condition
        behavioral_stats_{v1|v2}.txt     # formatted statistical report

Resources:
    CPU only (no GPU needed) — regex matching + statistics
    Memory: 16G
    Time: <1 hour
    Env: behavior_env

Run:
```bash
# V1 analysis (chained after V1 causality test)
sbatch --dependency=afterok:<V1_CAUSALITY_JOBID> behavior_analysis.sh

# V2 analysis (manual, after V2 generation completes)
conda activate behavior_env
python 4_behavior_analysis.py \
    --input data/intervention_results/v2_exp1_recreation/per_subject \
    --version v2 \
    --topics /jukebox/graziano/rachel/ai_mind_rep/exp_1/code/data_gen/data/conds/topics.csv
```

--------------------------------------------------------------------------------
### Full Pipeline — SLURM Job Chaining
--------------------------------------------------------------------------------

The complete pipeline can be submitted as a chain of dependent jobs:

```bash
# Step 1: Train probes
PROBE_JOB=$(sbatch --parsable train_and_read_controlling_probes.sh)

# Step 2: V1 causality test (starts when probes finish)
V1_JOB=$(sbatch --parsable --dependency=afterok:$PROBE_JOB causality_test_on_mind.sh)

# Step 3: Behavioral analysis (starts when V1 finishes)
sbatch --dependency=afterok:$V1_JOB behavior_analysis.sh

# Optional: V2 multi-turn recreation (starts when probes finish)
sbatch --dependency=afterok:$PROBE_JOB 3b_causality_exp1_recreation.sh
```

--------------------------------------------------------------------------------
### Additional Analyses (PLANNED)
--------------------------------------------------------------------------------

    - Layer-wise accuracy:
        Locate the layer(s) where partner-type representation is strongest.

    - Hidden-state visualization (PCA / UMAP):
        Visualize separability of human vs ai activations and how points move
        under steering (+N·v, −N·v).

    - Control vs reading probe comparison:
        Check whether control probes (trained on task-context activations)
        steer more effectively than reading probes alone.

    - Qualitative sample inspection:
        Collect example outputs before/after intervention and document
        stylistic / reasoning shifts.

    - Cross-validation with Experiment 1 behavioral profiles:
        Compare V2 steered outputs against the linguistic feature profiles
        observed in Experiment 1 to test whether activation steering reproduces
        the same behavioral patterns that emerged from partner-identity beliefs.

--------------------------------------------------------------------------------
## Interpretation
--------------------------------------------------------------------------------

Interpretation logic:

    - If linear probes decode partner type AND activation additions systematically
      alter model output in the expected direction:
          → Evidence for an internal latent variable the model uses to track
            partner identity or "mind type."

    - If only decodability is present but interventions have no effect:
          → The probe may be reading out a passive correlation, not a causal
            control dimension.

    - If steering succeeds but reading probes are weak:
          → Partner information may be more distributed and better captured
            by task-context activations (e.g., final-user-message tokens)
            than by the explicit reading-prompt token.

    - If steered outputs match the Experiment 1 behavioral profiles:
          → The internal partner-identity representation is functionally
            linked to the same conversational adaptations observed in
            naturalistic interaction, strengthening the claim that LLMs
            maintain a genuine (if shallow) model of their interlocutor.

This experiment will help clarify how LLMs internalize the idea of "who" they
are talking to, and how those internal variables relate to the kinds of human
belief and mind-interpretation effects studied in Experiment 1.

--------------------------------------------------------------------------------
## References & Attribution
--------------------------------------------------------------------------------

This project adapts the probing and activation-intervention methodology introduced in:

Chen, Y., Wu, A., DePodesta, T., Yeh, C., Li, K., Marin, N. C., ... & Viégas, F. (2024). 
Designing a dashboard for transparency and control of conversational AI. 
arXiv preprint arXiv:2406.07882.

We additionally adapt components of the official TalkTuner codebase:
https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/tree/main

Their repository's structure for conversation generation, probe training,
and activation steering served as the primary template for our approach.
