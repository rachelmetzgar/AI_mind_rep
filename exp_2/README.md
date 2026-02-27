# Experiment 2: Naturalistic Conversation Probing & Causal Intervention

Linear probes trained on LLaMA-2-13B-Chat hidden states to classify whether the model's conversation partner is human or AI, followed by causal intervention to test whether these representations drive behavior.

## Data Versions

Each version uses a different system prompt strategy from Experiment 1:

| Version | Description |
|---------|-------------|
| `labels` | Partner labeled as "a Human" / "an AI" (primary version) |
| `balanced_names` | Gender-balanced names instead of labels |
| `balanced_gpt` | Like balanced_names but with GPT-4 replacing Copilot |
| `names` | Original Sam/Casey/Copilot names (deprecated: name confound) |
| `nonsense_codeword` | Token-matched control: "Your session code word is {a Human/an AI}" |
| `nonsense_ignore` | Token-present with ignore instruction |

## Directory Structure

```
exp_2/
├── code/
│   ├── config.py              # Central config with set_version()
│   ├── pipeline/              # Numbered pipeline scripts
│   │   ├── 1_*.py             # Probe training (turn 5 + turn comparison)
│   │   ├── 2_*.py             # Causal intervention generation
│   │   ├── 3_*.py             # GPT judge evaluation
│   │   └── 4_*.py             # Behavioral analysis
│   ├── src/                   # Shared modules (dataset, probes, losses, etc.)
│   ├── slurm/                 # SLURM job scripts (a=V1, b=V2)
│   └── analysis/              # Cross-variant analysis scripts
├── data/
│   ├── {version}/
│   │   ├── probe_checkpoints/ # Trained probe weights (turn_1 through turn_5)
│   │   └── intervention_results/ # V1 and V2 generation outputs
│   └── shared/                # Version-independent data (topics, questions)
├── results/
│   ├── {version}/             # Per-version analysis outputs
│   └── comparisons/           # Cross-variant comparison HTML reports
│       ├── probe_training/    # Probe accuracy, turn layerwise, alt token positions
│       └── causality_qc/      # V1 causal intervention QC summary
├── archive/                   # Old self-contained variant directories
└── logs/                      # SLURM logs by version
```

## Pipeline

All scripts require `--version {version}`. Run from `exp_2/` root.

### 1. Probe Training
```bash
# Turn 5 (full conversation) — reading + control probes
VERSION=labels sbatch code/slurm/1_train_and_read_controlling_probes.sh

# Turns 1-4 (array job, 4 tasks)
VERSION=labels sbatch code/slurm/1b_train_probes_turn_comparison.sh

# Compare across turns
python code/pipeline/1c_compare_turn_probes.py --version labels

# Degradation correlation (requires GPU)
VERSION=labels sbatch code/slurm/1d_degradation_probe_correlation.sh
python code/pipeline/1e_analyze_degradation_results.py --version labels

# Alternative token position probes
VERSION=labels sbatch code/slurm/1f_alternative_position_probes.sh
```

### 2. Causal Intervention
```bash
# V1: Single-prompt generation (a = V1)
VERSION=labels sbatch code/slurm/2a_causality_generate.sh

# V2: Multi-turn recreation (b = V2)
VERSION=labels sbatch code/slurm/2b_causality_generate.sh
```

### 3. GPT Judge Evaluation
```bash
# V1 judge (a = V1)
VERSION=labels sbatch code/slurm/3a_causality_judge.sh

# V2 judge (b = V2)
VERSION=labels sbatch code/slurm/3b_causality_judge.sh
```

### 4. Behavioral Analysis
```bash
# V1 behavioral analysis (a = V1)
VERSION=labels sbatch code/slurm/4a_behavior_analysis.sh

# V2 behavioral analysis (b = V2)
VERSION=labels sbatch code/slurm/4b_behavior_analysis.sh
```

## Config

Central configuration in `code/config.py`. All paths, hyperparameters, and version management:

```python
from config import config, set_version, add_version_argument

set_version("labels")
csv_dir = config.PATHS.csv_dir           # -> exp_1/labels/data/...
probes  = config.PATHS.probe_checkpoints # -> exp_2/data/labels/probe_checkpoints/
```

## Analysis Scripts

Cross-variant reports in `code/analysis/`. Run from `exp_2/` root:

```bash
# Probe training comparison (all 6 versions, unified scale)
python code/analysis/gen_probe_training_comparison.py

# Turn comparison layerwise (all 6 versions, turns 1-5, + alt position probes)
python code/analysis/gen_turn_layerwise_html.py

# Alternative token position comparison (labels, balanced_gpt, nonsense_codeword × 5 turns)
python code/analysis/gen_alt_position_comparison.py

# V1 causality QC summary (4 main versions side by side)
python code/analysis/create_v1_qc_summary.py
```

## Results

### Cross-variant (start here)

- [**Probe training comparison**](results/comparisons/probe_training/probe_training_comparison.html) — All 6 versions on the same scale, makes the labels vs named-partners vs nonsense distinction clear.
- [**Turn comparison layerwise**](results/comparisons/probe_training/turn_comparison_layerwise.html) — Prompt dilution story: perfect at turn 1, degrades monotonically. Includes token position examples.
- [**Alt position comparison**](results/comparisons/probe_training/alt_position_comparison.html) — Where in the token sequence does the signal live? Cross-version x cross-turn.
- [**V1 causal intervention (labels)**](results/labels/v1_analysis_summary.html) — Causal intervention results: does steering the model with probe directions actually change behavior?

### Data degradation

- [Labels](results/labels/degradation_analysis/degradation_probe_report.html) · [Nonsense Codeword](results/nonsense_codeword/degradation_analysis/degradation_probe_report.html) · [Nonsense Ignore](results/nonsense_ignore/degradation_analysis/degradation_probe_report.html) — Text quality metrics + probe confidence across turns.

### Per-version (deep dives)

Each version under `results/{version}/` contains:
- `probe_training/probe_training_report.html` — 9-section probe training analysis
- `v1_analysis_summary.html` — V1 causal intervention QC ([labels](results/labels/v1_analysis_summary.html), [balanced_names](results/balanced_names/v1_analysis_summary.html), [balanced_gpt](results/balanced_gpt/v1_analysis_summary.html), [names](results/names/v1_analysis_summary.html))
- `degradation_analysis/degradation_probe_report.html` — Text degradation vs probe confidence

## Key Findings

**Probe training (turn 5)**
- Named-partner versions (names, balanced_names, balanced_gpt) achieve 75-79% peak accuracy, but control probes match reading probes — the signal comes from partner name tokens, not abstract identity.
- Labels version shows a reading > control advantage (65% vs 60%, d=0.76), indicating probes detect abstract "human vs AI" beyond lexical cues.
- Nonsense controls are at chance (~55%), confirming the signal requires semantic identity framing.

**Turn comparison**
- Turn 1 probes achieve near-perfect accuracy (~97%) across ALL versions including nonsense controls. This reflects proximity to the system prompt, not abstract representation.
- Accuracy degrades monotonically across turns (prompt dilution): the system prompt tokens become proportionally diluted in longer sequences.
- Peak layers shift from early (7-9) at turn 1 to late (33-40) at turn 5.

**Alternative token positions**
- BOS token: at chance (negative control).
- Random mid-sequence: at chance for turns 2+ (partner identity is not broadcast to arbitrary tokens). Elevated at turn 1 due to short sequence length.
- First `</s>` token: perfect accuracy, but this is a causal attention artifact — the preceding context is identical regardless of conversation length.
- Irrelevant (weather) suffix: nearly matches partner-relevant suffix, indicating the representation is accessible from any continuation token.

## Key Design Decisions

- **Probe types**: Reading probe (appends "I think the conversation partner of this user is") vs control probe (probes at pre-generation position with no suffix).
- **Layer strategies**: `peak_15` (top 15 layers), `narrow` (10-layer window), `wide` (all above threshold), `all_70` (all >= 0.70).
- **`--mode V1/V2`**: V1 = single-prompt causality test. V2 = multi-turn conversation recreation. Not to be confused with `--version` (data variant).
- **Conversation data**: Raw conversations are stored in `exp_1/{version}/data/` and referenced via `config.py`. Exp 2 stores only trained probes and intervention outputs.
