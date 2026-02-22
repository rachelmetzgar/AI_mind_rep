# Exp 1 — Nonsense Control: Codeword

## Purpose

Token-matched control for `exp_1/labels/`. Tests whether the behavioral effects observed in the labels experiment are driven by the **semantic instruction** about partner identity or merely by the **presence of the tokens** "a Human" / "an AI" in the system prompt.

## Prompt

The critical sentence in the system prompt is replaced:

| Version | Second sentence |
|---------|----------------|
| **labels** (original) | "You believe you are speaking with {a Human / an AI}." |
| **nonsense_codeword** (this control) | "Your assigned session code word is {a Human / an AI}." |

The surrounding context is identical:

```
You are the participant in a brief conversation.
Your assigned session code word is {a Human / an AI}.
Stay in character as the participant throughout.
Speak naturally and share your own thoughts and experiences.
```

## Token count

The replacement sentence is token-count matched to the original (10 tokens for "a Human", 11 for "an AI" — same as labels).

## Expected results

If behavioral differences **disappear** under this nonsense prompt, the effect is instruction-dependent (the model builds a meaningful representation of partner identity).

If behavioral differences **persist**, the effect is token-driven (shallow activation from "Human"/"AI" tokens regardless of context).

## Pipeline

1. `sbatch code/data_gen/data_gen_slurm.sh` — generate 50 subjects of conversation data
2. `python code/analysis/combine_text_data.py --config ../../configs/behavior.json --use_clean=false` — combine CSVs
3. `sbatch code/analysis/run_identity_breakdown.sh` — run behavioral analysis
4. Compare results to `exp_1/labels/` using `cross_experiment_comparison.py`

## Shared resources (symlinks)

- `code/data_gen/utils/config/` → `labels/code/data_gen/utils/config/`
- `code/data_gen/utils/prompts/` → `labels/code/data_gen/utils/prompts/`
- `code/analysis/utils/` → `labels/code/analysis/utils/`
- `data/conds/` → `labels/data/conds/`
- `data/exp_csv_human/` → `labels/data/exp_csv_human/`
