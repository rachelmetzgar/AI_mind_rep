# AI Mind Rep — Simulation & Behavioral Analyses

**Author:** Rachel C. Metzgar  
**Repo root:** `ai_mind_rep/exp_1`  
**Python:** 3.11+  

This project generates **synthetic human–AI conversations** and runs a suite of **behavioral text analyses** (sentiment, politeness, hedging, ToM, questions, word count, etc.). It supports OpenAI chat models and base LLaMA models, and standardizes outputs for reproducibility.

---
## Quick start

1) **(Optional) Generate data**

From `exp_1/`:
```bash
export OPENAI_API_KEY=...
bash code/data_gen/data_gen_slurm.sh
python code/data_gen/gpt_data_generation.py
```
Generated CSVs: `data/temp_<temperature>/<model>/sXXX.csv`

2) **Run behavioral analyses**
```bash
bash code/analysis/run_all_behavior.sh
```

Or run individual scripts:

First: **Combine per-subject files**
```bash
python code/analysis/combine_text_data.py --config configs/behavior.json
```
Then run individual scripts:
```bash
python code/analysis/politeness.py --config configs/behavior.json
python code/analysis/sentiment_vader.py --config configs/behavior.json
python code/analysis/wordcount.py --config configs/behavior.json
```
Outputs: `results/<model>/<temperature>/<analysis_name>/`

---

## Configuration

`configs/behavior.json` defines:
- subject_ids
- model (e.g., gpt-3.5-turbo)
- temperature (e.g., 0.8)

---

## Key analysis scripts

- politeness.py — polite markers ("please", "thank you", "sorry").
- hedging.py — hedges ("maybe", "might", "I think").
- filler.py — fillers ("um", "uh", "like").
- empath_tom.py — ToM score via Empath.
- questions.py — question frequency (? and regex).
- sentiment_vader.py — VADER sentiment analysis.
- tom_words.py — partner-referential ToM phrases.
- wordcount.py — word counts + Condition × Sociality ANOVA.
- qual_connect.py, tom_ai_ratings.py — conversation quality & connectedness.
- semantic_diversity.py, sentiment_transformer.py — lexical and transformer-based metrics.

Each analysis produces:
- per-interaction CSV
- subject-level CSV
- stats text file
- standardized plots

---

## Plotting & palette

Shared in `plot_helpers.py`:
- plot_violin_basic() — clean violins
- main_effect_violin_lines() — Bot left, Human right, subject lines + stars
- barplot_with_lines() — mean ± SD bars + significance stars

Palette:
- Human = steelblue
- Bot = sandybrown
- Social = darker tone, Nonsocial = lighter tone

---

## Stats

- Within-subject paired t-tests (Human vs Bot).
- 2×2 repeated-measures ANOVA (Condition × Sociality in wordcount.py).
- Effect sizes: Cohen’s d.
- Environment info logged via run_logger.py.

---

## Project structure

exp_1/
├─ code/
│  ├─ analysis/
│  │  ├─ combine_text_data.py
│  │  ├─ empath_tom.py
│  │  ├─ filler.py
│  │  ├─ hedging.py
│  │  ├─ politeness.py
│  │  ├─ qual_connect.py
│  │  ├─ questions.py
│  │  ├─ run_all_behavior.sh
│  │  ├─ semantic_diversity.py
│  │  ├─ sentiment_transformer.py
│  │  ├─ sentiment_vader.py
│  │  ├─ tom_ai_ratings.py
│  │  ├─ tom_words.py
│  │  └─ wordcount.py
│  │
│  │  └─ utils/
│  │     ├─ cli_helpers.py
│  │     ├─ data_helpers.py
│  │     ├─ globals.py
│  │     ├─ plot_helpers.py
│  │     ├─ print_helpers.py
│  │     ├─ run_logger.py
│  │     ├─ stats_helpers.py
│  │     └─ subject_utils.py
│  │
│  ├─ data_gen/
│  │  ├─ gpt_data_generation.py
│  │  ├─ llm_data_generation.py
│  │  ├─ data_gen_slurm.sh
│  │  └─ utils/
│  │     ├─ config/                 # per-subject condition CSVs (conds_sXXX.csv)
│  │     ├─ prompts/                # topic text files
│  │     ├─ conversation_helpers.py
│  │     ├─ gpt_client.py
│  │     ├─ llama_client.py
│  │     ├─ log_helpers.py
│  │     ├─ prompts_config.py
│  │     ├─ randomize_conds.py
│  │     └─ sim_helpers.py
│  │
│  └─ (notebook checkpoint folders omitted)
│
├─ configs/
│  └─ behavior.json                 # model/temp/subject list for analyses
│
├─ data/
│  ├─ conds/                        # topic ↔ social/nonsocial map (e.g., topics.csv)
│  ├─ behavior/                     # (optional) behavior aggregations
│  └─ temp_<temperature>/<model>/   # generated transcripts: s001.csv … s050.csv
│
├─ logs/
│
└─ results/
   └─ <model>/<temperature>/
      ├─ empath_tom/
      ├─ filler/
      ├─ hedging/
      ├─ politeness/
      ├─ questions/
      ├─ sentiment_vader/
      ├─ tom_words/
      └─ wordcount/

---