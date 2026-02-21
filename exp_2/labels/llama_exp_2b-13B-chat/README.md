# Experiment 2 (Labels) — Causal Intervention with Name-Ablated Data

**Author:** Rachel C. Metzgar, Princeton University
**Repo root:** `ai_mind_rep/exp_2/labels/llama_exp_2b-13B-chat`

--------------------------------------------------------------------------------
## Overview
--------------------------------------------------------------------------------

This experiment adapts the TalkTuner-style probing framework (Chen et al., 2024)
to study how LLMs internally encode the identity or "mind type" of their
conversational partner: Human vs AI.

This is the **name-ablated version** of Experiment 2. Partner labels in the
Experiment 1 training data use generic type descriptors ("a human" / "an AI")
rather than specific names, eliminating name-specific and gender-specific
confounds that affected the original `names/` version.

**Key difference from `names/` version:** The original used named partners
(Sam, Casey, ChatGPT, Copilot). Probes trained on those conversations encoded
partner names rather than abstract identity — evidenced by "Copilot" token
artifacts in AI-steered outputs and female-coded language in human-steered
outputs. This version removes that confound entirely.

**Training data source:** `exp_1/labels/data/meta-llama-Llama-2-13b-chat-hf/0.8/`
(50 subjects x 40 conversations, raw uncleaned CSVs).

--------------------------------------------------------------------------------
## Current Status
--------------------------------------------------------------------------------

- **Phase 2 (probe training):** Complete. Reading probe peak 65.2% (layer 33),
  control probe peak 60.5% (layer 31). Weaker than names version (80-90%+),
  confirming names probes were largely encoding partner names.
- **Phase 2b (probe summary):** Complete. Stats, figures, and HTML report in
  `results/probe_training/`.
- **Phase 3 (causal generation V1):** Complete. Strengths 2, 4, 5, 6 with
  `peak_15` strategy. Reading probes show clean steering effects; control probes
  degrade at strengths 5-6.
- Phase 3 (V2): Not yet run.
- Phase 4 (GPT judge): Not yet run.
- Phase 5 (behavioral analysis): Not yet run.

--------------------------------------------------------------------------------
## Probe Training Results
--------------------------------------------------------------------------------

Probes trained with single 80/20 stratified train/test split (n~2000 samples,
n_test~400), matching TalkTuner methodology. 50 epochs per layer, best-epoch
checkpoint selection.

| Metric | Reading Probe | Control Probe |
|---|---|---|
| Mean best test acc | 0.580 | 0.552 |
| Peak best test acc | 0.652 (layer 33) | 0.605 (layer 31) |
| Overall reading > control | t(40)=4.88, p<.0001, d=0.76 | |
| Early layers (0-13) | M=0.532 | M=0.540 (ns) |
| Middle layers (14-27) | M=0.596 | M=0.557 (p<.0001) |
| Late layers (28-40) | M=0.615 | M=0.560 (p<.0001) |

Accuracy is modest but statistically above chance (p<.05 at 0.55 with n=400).
The reading probe advantage over control emerges in middle/late layers,
consistent with deeper layers encoding more abstract representations.

Full stats and figures: `results/probe_training/probe_training_report.html`

--------------------------------------------------------------------------------
## V1 Causality Results (Preliminary)
--------------------------------------------------------------------------------

Strengths tested: 2, 4, 5, 6. Strategy: `peak_15` (top 15 layers by accuracy).

- **MIN_PROBE_ACCURACY = 0.55** (lowered from 0.70 used in names version, since
  labels probes peak at 0.652).
- Reading probes use layers 22-37; control probes use layers 6-37.
- Reading probes are more robust: clean steering through strength 6 (3/60
  degenerate). Control probes degrade badly at strength 5-6 (48/60 degenerate
  at is_6 for human-steering).
- Human-steered responses become warmer/more informal; AI-steered maintain
  formal register. Effects scale with strength.

--------------------------------------------------------------------------------
## Repository Structure
--------------------------------------------------------------------------------

```
exp_2/labels/llama_exp_2b-13B-chat/
├── data/
│   ├── probe_checkpoints/
│   │   ├── control_probe/               # Probe weights + accuracy_summary.pkl
│   │   └── reading_probe/               # Probe weights + accuracy_summary.pkl
│   ├── causality_test_questions/
│   │   └── human_ai.txt                 # Held-out test prompts for V1
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
- `peak_15` — **Current choice.** Top 15 layers by probe accuracy.
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
## References
--------------------------------------------------------------------------------

Chen, Y., Wu, A., DePodesta, T., Yeh, C., Li, K., Marin, N. C., ... &
Viégas, F. (2024). Designing a dashboard for transparency and control of
conversational AI. arXiv preprint arXiv:2406.07882.
