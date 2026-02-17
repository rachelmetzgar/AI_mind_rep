# Session Notes — Feb 15, 2026

## Project Overview

This project investigates whether LLMs internally represent the "mind type" of their conversation partner (human vs AI) and whether steering those internal representations causally changes behavior. The project has multiple experiments:

- **Experiment 1 (`exp_1/`)**: Behavioral data generation — LLaMA-2-Chat-13B has multi-turn conversations on various topics. 50 subjects x 40 trials. Produces conversation transcripts analyzed with linguistic markers (hedges, discourse markers, disfluencies, theory of mind phrases, politeness, sentiment, etc.).

- **Experiment 2 (`exp_2/llama_exp_2b-13B-chat/`)**: Causal intervention experiment. Train linear probes on LLaMA-2-Chat-13B's internal activations to detect partner identity (human vs AI), then use those probes to steer the model's representations during generation.

- **Experiments 3 & 4 (`exp_3/`, `exp_4/`)**: Not explored this session.

---

## Experiment 2 Architecture

### Pipeline (5 phases)

1. **`1_preprocess_dataset.py`** — Preprocess Exp1 conversations into activation extraction format. Combined output: `combined_all.csv` (92K lines).

2. **`2_train_and_read_controlling_probes.py`** — Train two types of linear probes across all 41 transformer layers:
   - **Control probes**: Trained on last-token activations (surface-level partner detection)
   - **Reading probes**: Trained on "reflection token" activations (deeper internal representation)
   - Checkpoints saved in `data/probe_checkpoints/{control,reading}_probe/`
   - Each has `accuracy_summary.pkl` with per-layer validation accuracy

3. **`3_causality_generate.py`** — Generate steered outputs using probe-based interventions:
   - **V1**: Single-turn test questions (60 held-out questions). 3 conditions: baseline, human-steered, AI-steered.
   - **V2**: Multi-turn Exp1 recreation. Two-agent back-and-forth (4 exchange pairs per conversation). Per-subject via SLURM array. 50 subjects x 5 runs x 8 orders x 3 conditions = 120 conversations per subject.

4. **`4_causality_judge.py`** — GPT judge evaluation (not yet run on V2 peak_15 data).

5. **`5_behavior_analysis.py`** — Compute linguistic feature profiles and compare across conditions using RM-ANOVA (V2) or one-way ANOVA (V1). Imports shared linguistic marker utilities from `exp_1/code/analysis/utils/`.

### Intervention Mechanism

- Steering formula: `hidden_state += N * (target @ probe_weight)` applied at last token position
- `target = [1.0]` for human-steered, `target = [-1.0]` (label 0 = AI) for AI-steered
- Applied via forward hooks (`TraceDict` class) during generation
- Intervention strengths tested: N=4 and N=5

### Layer Selection Strategies

Four strategies defined in `LAYER_STRATEGIES`:
- **`narrow`**: Best contiguous 10-layer window (auto-selected by mean accuracy). Matches Viegas et al.
- **`wide`**: All layers with probe accuracy >= 0.70
- **`peak_15`**: Top 15 layers by accuracy (non-contiguous) — **this is the intended strategy for V2**
- **`all_70`**: Same as wide

### Probe Configurations

Defined in `PROBE_CONFIGS` (line 169 of `3_causality_generate.py`):
- **`control_probes`**: Control probes, all layers passing strategy filter (`layer_mode: "all"`)
- **`reading_probes_matched`**: Reading probes, restricted to control probe layers (currently commented out)
- **`reading_probes_peak`**: Reading probes, all layers passing strategy filter (`layer_mode: "all"`) — NOTE: despite the name, this does NOT use "exclude" mode; the label is just a name

---

## What We Found & Fixed This Session

### Problem 1: Wrong layer strategy used for V2 generation

The SLURM script (`slurm/3_causality_generate_V2.sh`) intended to use `peak_15` (stated in comments and IMPORTANT notes) but the actual python command did NOT pass `--layer_strategy peak_15`. The argparse default is `["narrow"]`, so V2 generation ran with the `narrow` strategy instead.

**Result**: Existing V2 data at `data/intervention_results/V2/narrow/reading_probes_peak/` used a 10-layer contiguous window (layers 30-39) instead of the intended top-15 non-contiguous layers.

**Fix**: Added `--layer_strategy peak_15` to line 59 of the SLURM script.

### Problem 2: Control probes commented out

In `PROBE_CONFIGS`, only `reading_probes_peak` was active. `control_probes` was commented out, so V2 generation only produced reading probe results — no control probe condition for comparison.

**Fix**: Uncommented the `control_probes` line in `PROBE_CONFIGS`.

### Problem 3: Wrong path in SLURM script

The SLURM script referenced `exp_2b/` in three places (output log, error log, PROJECT_ROOT) but the actual directory is `exp_2/`.

**Fix**: Replaced all `exp_2b/llama_exp_2b-13B-chat` with `exp_2/llama_exp_2b-13B-chat`.

### Re-submitted job

All three fixes applied. Re-submitted as **SLURM job 3509617** (array 0-99: 50 subjects x 2 strengths). Output will go to `data/intervention_results/V2/peak_15/` with both `control_probes/` and `reading_probes_peak/` subdirectories, each containing `is_4/` and `is_5/` per-subject CSVs.

---

## Qualitative Assessment of Existing V2 Data (narrow strategy)

Even though the existing data used the wrong strategy, the qualitative patterns are informative:

### Text Quality
- Coherent and fluent at both N=4 and N=5 — no token-level degradation
- Some repetition loops in later conversation turns (pairs 3-4), especially in human-steered condition where the model gets stuck echoing itself
- Repetition is not strength-dependent (similar at is_4 and is_5)

### Behavioral Differences (expected directions)
- **Human-steered**: More emotionally expressive ("OH MY GOSH!"), more personal anecdotes, more casual register, more emojis (especially at is_5), more interpersonal discourse markers
- **AI-steered**: More measured/formal tone, more structured/essay-like prose, more informational content, fewer personal details
- **Dose-response**: Subtle increase in effect size from is_4 to is_5, no degradation cliff

### V1 Results (peak_15 strategy, complete)
- `fung_interpersonal_rate` was the standout significant finding: F(2,177) = 4.193, p = 0.0166, with AI significantly lower than human (p = 0.0056)
- `demir_quantifier_rate` was marginal (p = 0.087) with AI lower than baseline/human
- Most other metrics non-significant in V1 (but V1 used independent t-tests with n=60 per condition, not repeated measures)

---

## Key File Paths

| File | Purpose |
|------|---------|
| `exp_2/llama_exp_2b-13B-chat/3_causality_generate.py` | Main generation script (V1 & V2) |
| `exp_2/llama_exp_2b-13B-chat/5_behavior_analysis.py` | Behavioral analysis (V1 & V2) |
| `exp_2/llama_exp_2b-13B-chat/slurm/3_causality_generate_V2.sh` | SLURM script for V2 generation |
| `exp_2/llama_exp_2b-13B-chat/slurm/5_behavior_analysis_V2.sh` | SLURM script for V2 behavioral analysis |
| `exp_2/llama_exp_2b-13B-chat/data/probe_checkpoints/` | Trained probe weights + accuracy summaries |
| `exp_2/llama_exp_2b-13B-chat/data/intervention_results/V1/peak_15/` | V1 results (complete) |
| `exp_2/llama_exp_2b-13B-chat/data/intervention_results/V2/narrow/` | V2 results (wrong strategy, keep for reference) |
| `exp_2/llama_exp_2b-13B-chat/data/intervention_results/V2/peak_15/` | V2 results (correct, generating now) |
| `exp_2/llama_exp_2b-13B-chat/logs/causality_v2/` | Generation logs |
| `exp_1/code/analysis/utils/` | Shared linguistic marker definitions |

---

## Next Steps

1. **Monitor job 3509617** — `squeue -j 3509617`. Previous narrow runs took ~152 min per subject per strength. With peak_15 (more layers), expect longer.
2. **Run V2 behavioral analysis** — Once generation completes: `python 5_behavior_analysis.py --version v2 --input data/intervention_results/V2/peak_15`
3. **Run V2 GPT judge evaluation** — `4_causality_judge.py` and associated SLURM script `4_causality_judge_V2.sh`
4. **Compare control vs reading probes** — The new run produces both, enabling the key dissociation test
5. **Consider**: The `reading_probes_matched` config is still commented out. This would restrict reading probes to only the layers where control probes also work (apples-to-apples comparison). May want to enable if needed.

---

## Environment

- **Model**: LLaMA-2-Chat-13B (41 transformer layers, hidden dim 5120)
- **Conda env**: `llama2_env` (for generation), `behavior_env` (for analysis)
- **Cluster**: SLURM on spockmk2 nodes, 1 GPU per job, 64GB memory
- **Filesystem**: `/jukebox/` and `/mnt/cup/labs/` appear to be the same filesystem at different mount points
