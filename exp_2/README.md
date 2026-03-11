# Experiment 2: Naturalistic Conversation Probing & Causal Intervention

Linear probes trained on LLaMA-2-13B-Chat hidden states to classify whether the model's conversation partner is human or AI, followed by causal intervention to test whether these representations drive behavior.

## Quick Start

```bash
# Train probes (turn 5)
VERSION=balanced_gpt sbatch code/slurm/1_train_and_read_controlling_probes.sh

# Generate V1 causal interventions
VERSION=balanced_gpt sbatch code/slurm/2a_causality_generate.sh

# Judge with GPT-4
VERSION=balanced_gpt sbatch code/slurm/3a_causality_judge.sh

# Behavioral analysis
VERSION=balanced_gpt sbatch code/slurm/4a_behavior_analysis.sh
```

All scripts accept `MODEL` env var (default: `llama2_13b_chat`).

## Results

### Cross-version comparisons (start here)

- [**Probe training comparison**](results/llama2_13b_chat/comparisons/probe_training/probe_training_comparison.html) — All versions on the same scale
- [**Turn comparison layerwise**](results/llama2_13b_chat/comparisons/probe_training/turn_comparison_layerwise.html) — Prompt dilution story: perfect at turn 1, degrades monotonically
- [**Alt position comparison**](results/llama2_13b_chat/comparisons/probe_training/alt_tokens/combined.html) — Where in the token sequence does the signal live?
- [**V1 QC summary**](results/llama2_13b_chat/comparisons/v1_causality/causality_qc/v1_qc_summary_all_variants.html) — Cross-version causal intervention quality check
- [**Judge comparison**](results/llama2_13b_chat/comparisons/v1_causality/judge/judge_comparison.html) — Cross-version judge accuracy

### Per-version reports

For each active version (`balanced_gpt`, `nonsense_codeword`):

- Probe training: `results/llama2_13b_chat/{version}/probe_training/probe_training_report.html`
- Degradation: `results/llama2_13b_chat/{version}/probe_training/degradation/degradation_probe_report.html`
- V1 judge: `results/llama2_13b_chat/{version}/V1_causality/judge/judge_report.html`
- V1 behavioral: `results/llama2_13b_chat/{version}/V1_causality/behavioral/peak_15/behavioral_summary.html`
- V1 summary: `results/llama2_13b_chat/{version}/V1_causality/v1_analysis_summary.html`
- Steered samples: `results/llama2_13b_chat/{version}/V1_causality/steered_samples.html`

## Data Versions

| Version | Description |
|---------|-------------|
| `labels` | Partner labeled as "a Human" / "an AI" |
| `balanced_names` | Gender-balanced names instead of labels |
| `balanced_gpt` | Like balanced_names but with GPT-4 replacing Copilot |
| `names` | Original Sam/Casey/Copilot names (deprecated: name confound) |
| `nonsense_codeword` | Token-matched control: "Your session code word is {a Human/an AI}" |
| `nonsense_ignore` | Token-present with ignore instruction |
| `labels_turnwise` | Labels version with turn-wise probe extraction |
| `you_are_labels` | "You are talking to a Human/an AI" framing |
| `you_are_labels_turnwise` | You-are framing with turn-wise extraction |
| `you_are_balanced_gpt` | You-are framing with GPT-4 |

Active versions being rerun: `balanced_gpt`, `nonsense_codeword`.

## Directory Structure

```
exp_2/
├── code/
│   ├── config.py                               # Central config (--model, --version)
│   ├── src/                                    # Shared modules
│   ├── shared/                                 # Version-independent input files
│   │   ├── conds/topics.csv
│   │   └── causality_test_questions/human_ai.txt
│   ├── 1_train_probes.py                       # Probe training (turn 5)
│   ├── 1a_probe_training_summary_generator.py  # Per-version probe report
│   ├── 1b_train_probes_turn_comparison.py      # Turns 1-4
│   ├── 1c_compare_turn_probes.py               # Turn comparison analysis
│   ├── 1c_turn_comparison_summary_generator.py # Turn comparison report
│   ├── 1d_degradation_probe_correlation.py     # Degradation analysis
│   ├── 1e_analyze_degradation_results.py       # Degradation report
│   ├── 1f_alternative_position_probes.py       # Alt token positions
│   ├── 1f_alt_position_summary_generator.py    # Alt position report
│   ├── 1f_investigate_weather_suffix.py        # Weather suffix investigation
│   ├── 1g_vocab_asymmetry_check.py             # Vocab asymmetry check
│   ├── 1g_operational_summary_generator.py     # Operational probe comparison
│   ├── 1h_probe_training_comparison_summary_generator.py  # Cross-version comparison
│   ├── 2_causality_generate.py                 # Causal intervention generation
│   ├── 2b_steered_samples_generator.py         # Steered sample viewer
│   ├── 3_causality_judge.py                    # GPT judge evaluation
│   ├── 3b_judge_summary_generator.py           # Per-version judge report
│   ├── 3c_judge_comparison_summary_generator.py # Cross-version judge comparison
│   ├── 4_behavior_analysis.py                  # Behavioral analysis
│   ├── 4b_behavioral_summary_generator.py      # Per-version behavioral report
│   ├── 4c_v1_qc_summary_generator.py           # Cross-version QC summary
│   └── slurm/                                  # SLURM wrappers (15 scripts)
├── results/
│   ├── {model}/{version}/                      # e.g., llama2_13b_chat/balanced_gpt/
│   │   ├── probe_training/
│   │   │   ├── data/                           # Probe checkpoints (gitignored)
│   │   │   ├── degradation/                    # Degradation analysis
│   │   │   │   └── data/                       # Degradation CSVs
│   │   │   ├── figures/
│   │   │   └── probe_training_report.html/.md
│   │   ├── V1_causality/
│   │   │   ├── data/                           # Generated CSVs (gitignored)
│   │   │   ├── judge/                          # Judge reports + figures
│   │   │   ├── behavioral/                     # Behavioral analysis reports
│   │   │   ├── steered_samples.html            # Steered sample viewer
│   │   │   └── v1_analysis_summary.html/.md    # Per-version V1 summary
│   │   └── V2_causality/
│   │       └── data/                           # Generated CSVs (gitignored)
│   └── {model}/comparisons/                    # Within-model cross-version comparisons
│       ├── probe_training/                     # Probe training comparisons + figures
│       │   └── alt_tokens/                     # Alt position comparisons
│       └── v1_causality/
│           ├── judge/                          # Cross-version judge comparison
│           └── causality_qc/                   # V1 QC summary across versions
├── logs/{model}/{version}/{step}/              # SLURM logs (gitignored)
│   # step = probe_training | V1_causality | V2_causality
└── write_up/
```

## Pipeline

All scripts accept `--version` (required) and `--model` (default: `llama2_13b_chat`).

### 1. Probe Training
```bash
# Turn 5 (full conversation) — metacognitive + operational probes
VERSION=balanced_gpt sbatch code/slurm/1_train_and_read_controlling_probes.sh

# Turns 1-4 (array job, 4 tasks)
VERSION=balanced_gpt sbatch code/slurm/1b_train_probes_turn_comparison.sh

# Compare across turns
python code/1c_compare_turn_probes.py --version balanced_gpt

# Degradation correlation (requires GPU)
VERSION=balanced_gpt sbatch code/slurm/1d_degradation_probe_correlation.sh
python code/1e_analyze_degradation_results.py --version balanced_gpt

# Alternative token position probes
VERSION=balanced_gpt sbatch code/slurm/1f_alternative_position_probes.sh
```

### 2. Causal Intervention
```bash
# V1: Single-prompt generation
VERSION=balanced_gpt sbatch code/slurm/2a_causality_generate.sh

# V2: Multi-turn recreation
VERSION=balanced_gpt sbatch code/slurm/2b_causality_generate.sh
```

### 3. GPT Judge Evaluation
```bash
# V1 judge
VERSION=balanced_gpt sbatch code/slurm/3a_causality_judge.sh

# V2 judge
VERSION=balanced_gpt sbatch code/slurm/3b_causality_judge.sh
```

### 4. Behavioral Analysis
```bash
# V1
VERSION=balanced_gpt sbatch code/slurm/4a_behavior_analysis.sh

# V2
VERSION=balanced_gpt sbatch code/slurm/4b_behavior_analysis.sh
```

## Config

Central configuration in `code/config.py`:

```python
from config import config, set_version, set_model, add_version_argument, add_model_argument

set_model("llama2_13b_chat")
set_version("balanced_gpt")
csv_dir = config.PATHS.csv_dir                  # -> exp_1/results/llama2_13b_chat/balanced_gpt/data/
probes  = config.PATHS.probe_checkpoints        # -> exp_2/results/llama2_13b_chat/balanced_gpt/probe_training/data/
v1_data = config.PATHS.intervention_results_v1  # -> exp_2/results/llama2_13b_chat/balanced_gpt/V1_causality/data/
```

## Key Design Decisions

- **Probe types**: Metacognitive probe (appends "I think my partner is") vs operational probe (probes at pre-generation position with no suffix).
- **Layer strategies**: `peak_15` (top 15 layers), `narrow` (10-layer window), `wide` (all above threshold), `all_70` (all >= 0.70).
- **`--mode V1/V2`**: V1 = single-prompt causality test. V2 = multi-turn conversation recreation. Not to be confused with `--version` (data variant).
- **Conversation data**: Raw conversations are stored in `exp_1/results/llama2_13b_chat/{version}/data/` and referenced via `config.py`. Exp 2 stores only trained probes and intervention outputs.
