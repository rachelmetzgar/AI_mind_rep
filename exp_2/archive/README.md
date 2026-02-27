# Archive

Old self-contained experiment directories from before the Feb 2026 restructure. Each variant previously had its own complete copy of all code at `{variant}/llama_exp_2b-13B-chat/`.

These are preserved for reference but are no longer actively used. All code now lives in `exp_2/code/` and uses `--version` to select the data variant.

## Contents

| Directory | Description |
|-----------|-------------|
| `labels/` | Primary version. Partner labeled as "a Human" / "an AI". |
| `balanced_names/` | Gender-balanced partner names. |
| `balanced_gpt/` | Like balanced_names but with GPT-4 replacing Copilot. |
| `names/` | Original Sam/Casey/Copilot names. Deprecated due to name confound. |
| `old/` | Early iterations: `gpt_exp_2b-13B-chat` (GPT-based), `unscrubbed_llama_exp_2b-13B-chat` (pre-scrub), `llama_exp_2b-13B-chat_###user_next_token` (alt token experiment). |

## Top-Level Files

| File | Description |
|------|-------------|
| `create_v1_qc_summary.py` | Old cross-variant QC summary script (now in `code/analysis/`). |
| `turn_comparison_layerwise.html` | Old layerwise comparison report (now in `results/cross_variant/`). |
| `v1_qc_summary_all_variants.html` | Old QC summary HTML (now in `results/cross_variant/`). |

## Structure of Each Variant

Each variant directory contained:
```
{variant}/llama_exp_2b-13B-chat/
├── data/                    # Now at exp_2/data/{variant}/
│   ├── probe_checkpoints/
│   └── intervention_results/
├── src/                     # Now at exp_2/code/src/
├── slurm/                   # Now at exp_2/code/slurm/
└── *.py                     # Now at exp_2/code/pipeline/
```
