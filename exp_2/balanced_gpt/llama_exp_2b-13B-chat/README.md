# Experiment 2 (Balanced GPT) — Causal Intervention with GPT-Generated Partners

**Author:** Rachel C. Metzgar, Princeton University
**Repo root:** `ai_mind_rep/exp_2/balanced_gpt/llama_exp_2b-13B-chat`

--------------------------------------------------------------------------------
## Overview
--------------------------------------------------------------------------------

This experiment adapts the TalkTuner-style probing framework (Chen et al., 2024)
to study how LLMs internally encode the identity or "mind type" of their
conversational partner: Human vs AI.

This is the **GPT-generated partner** version of Experiment 2. Unlike
`balanced_names/` where LLaMA-2 was the conversational partner, this version uses
GPT as the partner, allowing us to test whether partner identity representations
generalize across different LLM partners.

**Key difference from `balanced_names/` version:** The conversational partners in
Exp 1 training data are GPT-based agents (generating responses via GPT-3.5-Turbo
or similar), while the participant agent is still LLaMA-2-13B-Chat. This tests
whether probes trained on "human vs AI partner" generalize when the AI partner
is a different model architecture.

**Training data source:** `exp_1/balanced_gpt/data/meta-llama-Llama-2-13b-chat-hf/0.8/`
(50 subjects x 40 conversations, raw uncleaned CSVs).

--------------------------------------------------------------------------------
## Current Status
--------------------------------------------------------------------------------

- **Exp 1 data generation:** In progress (job 3559400, ~28/50 subjects running)
- **Phase 2 (probe training):** Waiting for Exp 1 data completion
- **Phase 2b (probe summary):** Not yet run
- **Phase 3 (causal generation V1/V2):** Not yet run
- **Phase 4 (GPT judge):** Not yet run
- **Phase 5 (behavioral analysis):** Not yet run

--------------------------------------------------------------------------------
## Research Question
--------------------------------------------------------------------------------

**Does partner identity representation depend on partner model architecture?**

If probes trained on GPT-as-partner data show similar accuracy and steering effects
to `balanced_names/` (where partner was also LLaMA-2), this suggests the participant
LLM encodes abstract "human vs AI" identity rather than model-specific features.

If accuracy or steering differs substantially, this could indicate:
- Encoding of partner-specific linguistic style rather than abstract identity
- Sensitivity to inter-model vs. intra-model distinction
- Confounds from architectural differences in partner responses

--------------------------------------------------------------------------------
## Relationship to Other Experiment 2 Variants
--------------------------------------------------------------------------------

| Variant | Human Partners | AI Partners | Participant | Purpose |
|---|---|---|---|---|
| `labels/` | "a human" | "an AI" | LLaMA-2 | Remove name confounds |
| `balanced_names/` | Gregory, Rebecca | ChatGPT, Copilot (LLaMA-2) | LLaMA-2 | Gender-balanced |
| `balanced_gpt/` (this) | Gregory, Rebecca | ChatGPT, Copilot (GPT) | LLaMA-2 | Cross-model generalization |
| `names/` (deprecated) | Sam, Casey | ChatGPT, Copilot | LLaMA-2 | Original (confounded) |

--------------------------------------------------------------------------------
## Repository Structure
--------------------------------------------------------------------------------

```
exp_2/balanced_gpt/llama_exp_2b-13B-chat/
├── data/
│   ├── probe_checkpoints/
│   │   ├── control_probe/               # Probe weights + accuracy_summary.pkl
│   │   └── reading_probe/               # Probe weights + accuracy_summary.pkl
│   ├── causality_test_questions/
│   │   └── human_ai.txt                 # Held-out test prompts for V1
│   ├── conds/
│   │   └── topics.csv                   # Topic list
│   └── intervention_results/
│       ├── V1/{layer_strategy}/{probe_type}/is_{N}/
│       └── V2/{layer_strategy}/{probe_type}/is_{N}/per_subject/
├── results/
│   └── probe_training/                  # Stats, figures, HTML report
├── logs/
├── src/
│   ├── dataset.py
│   ├── losses.py
│   ├── probes.py
│   └── train_test_utils.py
├── slurm/
│   ├── 2_train_and_read_controlling_probes.sh
│   ├── 3_causality_generate_V1.sh
│   ├── 3_causality_generate_V2.sh
│   ├── 4_causality_judge_V1.sh
│   ├── 4_causality_judge_V2.sh
│   ├── 5_behavior_analysis_V1.sh
│   └── 5_behavior_analysis_V2.sh
├── 2_train_and_read_controlling_probes.py
├── 2b_probe_training_summary.py         # Probe training stats + HTML report
├── 3_causality_generate.py
├── 4_causality_judge.py
├── 5_behavior_analysis.py
└── README.md
```

--------------------------------------------------------------------------------
## Pipeline
--------------------------------------------------------------------------------

**Wait for Exp 1 data generation to complete first!**

Check status:
```bash
cd /jukebox/graziano/rachel/ai_mind_rep/exp_1/balanced_gpt
bash check_status.sh
```

Once complete:
```bash
cd /jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_gpt/llama_exp_2b-13B-chat

# Phase 2: Train probes
sbatch slurm/2_train_and_read_controlling_probes.sh

# Phase 2b: Probe training summary
conda activate behavior_env
python 2b_probe_training_summary.py

# Phase 3: Causal generation
sbatch slurm/3_causality_generate_V1.sh
sbatch slurm/3_causality_generate_V2.sh

# Phase 4: GPT judge
sbatch slurm/4_causality_judge_V1.sh

# Phase 5: Behavioral analysis
sbatch slurm/5_behavior_analysis_V1.sh
```

--------------------------------------------------------------------------------
## Expected Results
--------------------------------------------------------------------------------

**If partner identity encoding is abstract/model-agnostic:**
- Probe accuracy comparable to `balanced_names/` (~55-65%)
- Similar steering effects in V1/V2 generation
- No systematic differences in behavioral measures

**If partner identity encoding is model-specific:**
- Probe accuracy may differ (higher if encoding model-specific features)
- Steering effects may be weaker (if probes don't capture abstract identity)
- Behavioral differences may reflect partner model differences rather than human/AI distinction

**Comparison to `balanced_names/` will isolate the effect of partner model architecture.**

--------------------------------------------------------------------------------
## References
--------------------------------------------------------------------------------

Chen, Y., Wu, A., DePodesta, T., Yeh, C., Li, K., Marin, N. C., ... &
Viégas, F. (2024). Designing a dashboard for transparency and control of
conversational AI. arXiv preprint arXiv:2406.07882.
