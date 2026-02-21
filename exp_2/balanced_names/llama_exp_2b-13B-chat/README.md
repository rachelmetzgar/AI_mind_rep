# Experiment 2 (Balanced Names) — Causal Intervention with Gender-Balanced Data

**Author:** Rachel C. Metzgar, Princeton University
**Repo root:** `ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat`

--------------------------------------------------------------------------------
## Overview
--------------------------------------------------------------------------------

This experiment adapts the TalkTuner-style probing framework (Chen et al., 2024)
to study how LLMs internally encode the identity or "mind type" of their
conversational partner: Human vs AI.

This is the **gender-balanced version** of Experiment 2. Partner labels in the
Experiment 1 training data use gender-balanced human names (Gregory/Rebecca)
rather than female-coded names (Sam/Casey), eliminating gender confounds that
could affect probe learning.

**Key difference from `labels/` version:** The `labels/` version uses generic
descriptors ("a human" / "an AI") to avoid name-specific confounds. This version
uses specific names (Gregory, Rebecca for humans; ChatGPT, Copilot for AI) but
balances gender across the human condition to isolate the human/AI distinction
from gender-associated linguistic patterns.

**Training data source:** `exp_1/balanced_names/data/meta-llama-Llama-2-13b-chat-hf/0.8/`
(50 subjects x 40 conversations, raw uncleaned CSVs).

--------------------------------------------------------------------------------
## Current Status
--------------------------------------------------------------------------------

- **Phase 2 (probe training):** Not yet run.
- **Phase 2b (probe summary):** Not yet run.
- **Phase 3 (causal generation V1/V2):** Not yet run.
- **Phase 4 (GPT judge):** Not yet run.
- **Phase 5 (behavioral analysis):** Not yet run.

--------------------------------------------------------------------------------
## Relationship to Other Experiment 2 Variants
--------------------------------------------------------------------------------

| Variant | Human Partner Labels | AI Partner Labels | Purpose |
|---|---|---|---|
| `labels/` | "a human" | "an AI" | Remove all name-specific confounds |
| `balanced_names/` (this) | Gregory, Rebecca | ChatGPT, Copilot | Gender-balanced names |
| `names/` (deprecated) | Sam, Casey | ChatGPT, Copilot | Original (female-coded confound) |

All variants train probes on naturalistic conversation data from Exp 1 to test
whether partner identity is linearly decodable and causally drives behavior.

--------------------------------------------------------------------------------
## Repository Structure
--------------------------------------------------------------------------------

```
exp_2/balanced_names/llama_exp_2b-13B-chat/
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

**Probe configurations** (in `3_causality_generate.py`):
- `control_probes`: Control probes, layers selected by strategy
- `reading_probes_peak`: Reading probes, own best layers per strategy
- `reading_probes_matched` (commented out): Reading probes restricted to control layers

**Layer strategies** (`--layer_strategy`):
- `peak_15` — Top 15 layers by probe accuracy (recommended).
- `narrow` — Best contiguous 10-layer window, Viegas et al. style.
- `wide` / `all_70` — All layers above accuracy threshold.

--------------------------------------------------------------------------------
## Pipeline
--------------------------------------------------------------------------------

```bash
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

If probes successfully learn gender-balanced partner identity (not gender):
- Probe accuracy should be comparable to `labels/` version (baseline: 55-65%)
- No systematic gender artifacts in steered outputs
- Behavioral effects should reflect human/AI distinction, not gender patterns

If gender confounds remain:
- Probe accuracy may be artificially high (learning gender, not partner type)
- Steered outputs may show gendered language patterns
- Results should be compared to `labels/` version to isolate gender effects

--------------------------------------------------------------------------------
## References
--------------------------------------------------------------------------------

Chen, Y., Wu, A., DePodesta, T., Yeh, C., Li, K., Marin, N. C., ... &
Viégas, F. (2024). Designing a dashboard for transparency and control of
conversational AI. arXiv preprint arXiv:2406.07882.
