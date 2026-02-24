# AI Mind Representation — Human vs AI Conversations

Author: Rachel C. Metzgar, Princeton University

This experiment adapts the TalkTuner-style probing framework (Chen et al., 2024) method to study how large language models internally encode the identity or "mind type" of their conversational partner: Human vs AI.

This repository contains modified and extended code originally from Chen et al.'s TalkTuner implementation (MIT Licensed). The original copyright (c) 2024 Yida Chen is retained as required.

The core idea:
- Use real conversation data from Experiment 1, where LLMs conversed under manipulated partner-identity beliefs (told their partner was a Human or an AI).
- Extract hidden activations from a target model (LLaMA-2-Chat-13B).
- Train linear probes to read out partner type and derive steering directions.
- Test whether shifting activations along these directions causally changes behavior.

This will (1) replicate and extend prior probing/intervention work using ecologically valid conversation data and (2) connect causal interpretability findings directly to the behavioral effects observed in Experiment 1.

Status:
- Phase 1 (data preprocessing): complete.
- Phase 2 (probe training): complete.
- Phase 3 (causal generation): implemented, parallelized by strength.
- Phase 4 (GPT judge): implemented, standalone from generation.
- Phase 5 (behavioral analysis): implemented, strength-aware.

--------------------------------------------------------------------------------
## Repository Structure
--------------------------------------------------------------------------------

```
ai_mind_rep/exp_2b/llama_exp_2b-13B-chat/
├── data/
│   ├── human_ai_conversations/          # Preprocessed Exp1 conversation files
│   ├── probe_checkpoints/
│   │   ├── control_probe/               # Control probe weights + accuracy_summary.pkl
│   │   └── reading_probe/               # Reading probe weights + accuracy_summary.pkl
│   ├── causality_test_questions/
│   │   └── human_ai.txt                 # Held-out test prompts for V1
│   └── intervention_results/
│       ├── V1/                          # Single-turn causal test outputs
│       │   ├── control_probes/
│       │   │   ├── is_2/                # Strength N=2
│       │   │   │   ├── intervention_responses.csv
│       │   │   │   ├── generation_config.json
│       │   │   │   ├── human_ai_causal_examples.txt
│       │   │   │   └── judge_results.json        # From 4_causality_judge.py
│       │   │   ├── is_4/
│       │   │   ├── is_8/
│       │   │   └── is_16/
│       │   ├── reading_probes/
│       │   │   ├── is_2/ ... is_16/
│       │   └── behavioral_results/      # From 5_behavior_analysis.py
│       └── V2/                          # Multi-turn Exp1 recreation outputs
│           ├── control_probes/
│           │   ├── is_2/
│           │   │   ├── per_subject/
│           │   │   │   ├── s001.csv ... s050.csv
│           │   │   └── judge_s001.json ... judge_s050.json
│           │   ├── is_4/ ... is_16/
│           ├── reading_probes/
│           │   ├── is_2/ ... is_16/
│           └── behavioral_results/
├── logs/
├── src/
│   ├── dataset.py
│   ├── losses.py
│   ├── probes.py
│   ├── train_test_utils.py
│   ├── intervention_utils.py
│   └── prompt_utils.py
├── slurm/
│   ├── 2_train_and_read_controlling_probes.sh
│   ├── 3_causality_generate_V1.sh       # Array[0-3]: 4 strengths
│   ├── 3_causality_generate_V2.sh       # Array[0-199]: 50 subjects × 4 strengths
│   ├── 4_causality_judge_V1.sh          # Array[0-3]: 4 strengths
│   ├── 4_causality_judge_V2.sh          # Array[0-199]: 50 subjects × 4 strengths
│   ├── 5_behavior_analysis_V1.sh        # Array[0-3]: 4 strengths
│   └── 5_behavior_analysis_V2.sh        # Array[0-3]: 4 strengths
├── 1_preprocess_dataset.py
├── 2_train_and_read_controlling_probes.py
├── 3_causality_generate.py
├── 4_causality_judge.py
├── 5_behavior_analysis.py
├── LICENSE
└── README.md
```

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

Each CSV contains one row per exchange (5 exchanges per trial, 40 trials per
subject = 200 rows). Key columns:

    subject, trial, partner_type, partner_name, transcript_sub,
    transcript_llm, topic, Quality, Connectedness

Preprocessed data location:

    data/human_ai_conversations/
        conversation_XXXX_partner_{human|ai}.txt
        metadata.json

--------------------------------------------------------------------------------
## Step-by-Step Analysis Pipeline
--------------------------------------------------------------------------------

### Phase 1 – Data Preprocessing

Code: `1_preprocess_dataset.py`
Env: `behavior_env`

Converts Experiment 1 conversation CSVs into the .txt format expected by
the activation extraction pipeline (src/dataset.py). Scrubs partner names
to prevent label leakage.

```bash
python 1_preprocess_dataset.py \
    --input_dir data/meta-llama-Llama-2-13b-chat-hf/0.8 \
    --output_dir data/human_ai_conversations \
    --verify
```

Expected output: 2,000 conversation files (50 subjects × 40 trials),
balanced 1,000 human / 1,000 AI.

--------------------------------------------------------------------------------

### Phase 2 – Train Reading & Control Probes

Code: `2_train_and_read_controlling_probes.py`
SLURM: `slurm/2_train_and_read_controlling_probes.sh`
Env: `llama2_env`

Loads LLaMA-2-Chat-13B, builds activation datasets, and for each transformer
layer trains:
- **Reading probes** on the "I think the partner…" reflection token
- **Control probes** on the last user message token

Saves probe weights + accuracy summaries under:

    data/probe_checkpoints/{reading_probe,control_probe}/
        human_ai_probe_at_layer_{L}.pth
        accuracy_summary.pkl

Reading probes evaluate *decodability* (whether the model represents partner
identity). Control probes derive **steering vectors** for causal intervention.

```bash
sbatch slurm/2_train_and_read_controlling_probes.sh
```

--------------------------------------------------------------------------------

### Phase 3 – Causal Intervention: Generation

Code: `3_causality_generate.py`
SLURM: `slurm/3_causality_generate_V1.sh`, `slurm/3_causality_generate_V2.sh`
Env: `llama2_env`

Generates steered model outputs under three conditions: baseline (no steering),
human-steered (+N·v), and AI-steered (−N·v). Both probe types (control + reading)
are run within each job.

**Parallelization:** Intervention strengths [2, 4, 8, 16] are parallelized
via SLURM array. Each array job loads the model once and runs both probe types
at a single strength.

Probe accuracies are auto-loaded from `accuracy_summary.pkl` saved during
training — no hardcoded accuracy dictionaries. Probes below 70% validation
accuracy are excluded.

#### V1 – Single-Turn Causal Test

Generates responses to held-out test questions. Each question gets baseline,
human-steered, and AI-steered responses.

```bash
# 4 parallel jobs (one per strength), each runs both probe types
sbatch slurm/3_causality_generate_V1.sh
```

Output per strength/probe: `intervention_responses.csv` (question_idx, question,
condition, response) + `generation_config.json` + human-readable examples.

#### V2 – Multi-Turn Experiment 1 Recreation

Two-agent back-and-forth conversations matching Experiment 1's structure exactly
(same subject configs, topics, 4-exchange format). Steering applied only to the
subject agent.

```bash
# 200 parallel jobs (50 subjects × 4 strengths)
sbatch slurm/3_causality_generate_V2.sh
```

Output: per-subject CSVs with columns: subject, run, order, trial, condition,
topic, topic_file, pair_index, transcript_sub, transcript_llm.

**CLI options:**
```bash
# Run all strengths sequentially (no SLURM)
python 3_causality_generate.py --version V1 --strengths 2 4 8 16

# Run single strength (for SLURM dispatch)
python 3_causality_generate.py --version V1 --strength 16

# V2 single subject
python 3_causality_generate.py --version V2 --subject_idx 0 --strength 8
```

--------------------------------------------------------------------------------

### Phase 4 – Causal Intervention: GPT Judge

Code: `4_causality_judge.py`
SLURM: `slurm/4_causality_judge_V1.sh`, `slurm/4_causality_judge_V2.sh`
Env: any Python env with `openai`, `scipy`, `pandas` (no GPU needed)
Requires: `OPENAI_API_KEY` environment variable

Standalone judge script, separated from generation for independent debugging
and re-running. Reads generation CSVs and evaluates with GPT-4-turbo-preview
(matching the Viegas/TalkTuner methodology).

**Judge methodology (Viegas-style):**
- For each question/topic, both the target type (human or AI) and the response
  presentation order (which is Response 1 vs 2) are independently randomized
  per question (seeded for reproducibility).
- Judge prompt: "Please answer which one of the following responses is more
  consistent with a conversation where the *assistant's conversational partner*
  is a {partner_type}."
- Failed API calls retry with exponential backoff (5 attempts, 2s→32s).
  Persistent failures are marked NA and excluded from the success rate.
- Binomial test against chance (50%) for significance.

```bash
# V1: 4 parallel jobs (one per strength)
sbatch slurm/4_causality_judge_V1.sh

# V2: 200 parallel jobs (50 subjects × 4 strengths)
sbatch slurm/4_causality_judge_V2.sh
```

**CLI options:**
```bash
# Judge a single directory
python 4_causality_judge.py --version V1 --result_dir data/intervention_results/V1/control_probes/is_16

# Walk and judge everything
python 4_causality_judge.py --version V1 --result_root data/intervention_results/V1

# V2 single subject
python 4_causality_judge.py --version V2 --result_dir data/intervention_results/V2/control_probes/is_16 --subject s001

# Override seed
python 4_causality_judge.py --version V1 --result_root data/intervention_results/V1 --seed 123
```

Output: `judge_results.json` (V1) or `judge_{subject_id}.json` (V2), saved
alongside the generation CSVs. Includes success rate, position bias stats,
binomial p-value, and per-question judge details with scratchpad.

--------------------------------------------------------------------------------

### Phase 5 – Behavioral Analysis

Code: `5_behavior_analysis.py`
SLURM: `slurm/5_behavior_analysis_V1.sh`, `slurm/5_behavior_analysis_V2.sh`
Env: `behavior_env` (no GPU needed)

Computes linguistic feature profiles on steered outputs and tests whether
steering produces measurable condition effects across the same dimensions
measured in Experiment 1.

**Linguistic features** (same as Experiment 1 pipeline):
- Theory of Mind language (Wagovich et al., 2024)
- Discourse markers (Fung & Carter, 2007): interpersonal, referential,
  structural, cognitive + total
- Hedging (Demir, 2018 / Hyland, 1998): modals, verbs, adverbs, adjectives,
  quantifiers, nouns + total
- Disfluencies (LIWC 2007): nonfluencies, fillers
- Politeness markers (positive, negative, impolite)
- Quotative/discourse "like" (Dailey-O'Cain, 2000)
- Sentiment (VADER compound score)
- Word count, question count

**Directory-aware:** Auto-discovers `{probe}_probes/is_{N}/` structure and runs
analysis per probe type × strength. Cross-probe comparison tables generated
when both probe types are present.

**Statistical tests:**
- V1: one-way ANOVA + independent t-tests (each question = observation)
- V2: one-way RM-ANOVA + paired t-tests (subject as random effect)
- V2 optional: two-way RM-ANOVA Condition (3) × Sociality (2)

```bash
# 4 parallel jobs (one per strength)
sbatch slurm/5_behavior_analysis_V1.sh
sbatch slurm/5_behavior_analysis_V2.sh
```

**CLI options:**
```bash
# All strengths
python 5_behavior_analysis.py --version v1

# Specific strength
python 5_behavior_analysis.py --version v2 --strength 16 --topics data/topics.csv
```

Output (per probe type × strength):
- `utterance_metrics_{probe}_{strength}.csv`
- `trial_metrics_{probe}_{strength}.csv` (V2 only)
- `subject_condition_{probe}_{strength}.csv` (V2 only)
- `stats_{v1|v2}_{probe}_{strength}.txt`
- `stats_{v1|v2}_combined.txt` (cross-probe comparison)

All saved under `data/intervention_results/V{1,2}/behavioral_results/`.

--------------------------------------------------------------------------------
## Full Pipeline — SLURM Job Chaining
--------------------------------------------------------------------------------

```bash
# Step 1: Train probes
PROBE_JOB=$(sbatch --parsable slurm/2_train_and_read_controlling_probes.sh)

# Step 2: Generate steered outputs (V1: 4 jobs, V2: 200 jobs)
V1_GEN=$(sbatch --parsable --dependency=afterok:$PROBE_JOB slurm/3_causality_generate_V1.sh)
V2_GEN=$(sbatch --parsable --dependency=afterok:$PROBE_JOB slurm/3_causality_generate_V2.sh)

# Step 3: GPT judge (after generation completes)
V1_JUDGE=$(sbatch --parsable --dependency=afterok:$V1_GEN slurm/4_causality_judge_V1.sh)
V2_JUDGE=$(sbatch --parsable --dependency=afterok:$V2_GEN slurm/4_causality_judge_V2.sh)

# Step 4: Behavioral analysis (after generation completes — independent of judge)
sbatch --dependency=afterok:$V1_GEN slurm/5_behavior_analysis_V1.sh
sbatch --dependency=afterok:$V2_GEN slurm/5_behavior_analysis_V2.sh
```

Note: Behavioral analysis depends on generation only (not the judge), so it
can run in parallel with judging.

--------------------------------------------------------------------------------
## Parallelization Summary
--------------------------------------------------------------------------------

| SLURM Script                        | Array Size | Dimension                    | GPU | Env        |
|-------------------------------------|-----------|------------------------------|-----|------------|
| `2_train_and_read_controlling_probes.sh` | 1    | —                            | Yes | llama2_env |
| `3_causality_generate_V1.sh`        | 4         | strengths                    | Yes | llama2_env |
| `3_causality_generate_V2.sh`        | 200       | 50 subjects × 4 strengths   | Yes | llama2_env |
| `4_causality_judge_V1.sh`           | 4         | strengths                    | No  | any        |
| `4_causality_judge_V2.sh`           | 200       | 50 subjects × 4 strengths   | No  | any        |
| `5_behavior_analysis_V1.sh`         | 4         | strengths                    | No  | behavior_env |
| `5_behavior_analysis_V2.sh`         | 4         | strengths                    | No  | behavior_env |

All generation scripts run BOTH probe types (control + reading) within each
job — splitting probe types across jobs would double model load time.

--------------------------------------------------------------------------------
## Interpretation
--------------------------------------------------------------------------------

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
