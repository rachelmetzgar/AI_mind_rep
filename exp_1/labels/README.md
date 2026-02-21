# Experiment 1 (Labels) — Name-Ablated Behavioral Analysis

**Author:** Rachel C. Metzgar, Princeton University
**Repo root:** `ai_mind_rep/exp_1/labels`
**Python:** 3.11+

## What Changed from `names/`

This is a rerun of Experiment 1 with **generic type labels** replacing specific
partner names in the system prompt:

| | `names/` (original) | `labels/` (this version) |
|---|---|---|
| Human partners | Sam, Casey | "a human" |
| AI partners | ChatGPT, Copilot | "an AI" |
| N conditions | 4 (2 human, 2 AI names) | 2 (human, AI) |

### Why the change

The original named partners introduced confounds for downstream probe training
(Exps 2-4):
1. **Gender confound:** LLaMA-2 associates "Sam" and "Casey" with female
   identity, so probes conflate gender with the human/AI distinction.
2. **Name encoding:** The token "Copilot" appears directly in AI-steered
   probe outputs ("cop" repetition loops), showing probes learn the literal
   partner name rather than abstract AI identity.

By using only "a human" or "an AI" in the system prompt, any differences in
the model's internal representations must reflect the human/AI type distinction
itself, not name-specific or gender-specific associations.

### What stayed the same

- 50 independent GPT-3.5-Turbo participant agents
- 40 conversations per agent (2 partner types x 2 sociality x 10 topics)
- 20 social + 20 nonsocial topics (same topic set)
- Same conversation structure (5-turn exchanges)
- Same linguistic analysis pipeline and statistical framework

---
## Data cleaning

The `names/` version used `clean_transcripts.py` to scrub partner names, emoji,
emotes, meta-narration openers, and stuck-loop turns before behavioral analysis.

For the `labels/` version, **no cleaning is performed**. Rationale:
- **Partner names** — not present (only generic "a human" / "an AI" labels).
- **Emoji, emotes, meta-narration** — these are genuine behavioral markers that
  may differ by condition and are part of the signal we want to measure.
- **Stuck-loop turns** — rare (~3% of rows) and roughly balanced across
  conditions (125 AI, 174 human). Leaving them in preserves the raw signal.

The `clean_transcripts.py` and `clean_transcripts_0.py` files have been removed
from `code/analysis/` to avoid confusion. Only the `names/` version uses cleaning.

## Quick start

1) **Generate data**

```bash
sbatch code/data_gen/data_gen_slurm.sh
```
Generated CSVs: `data/meta-llama-Llama-2-13b-chat-hf/0.8/sXXX.csv`

2) **Combine per-subject files** (uses raw CSVs, no cleaning)
```bash
module load pyger
conda activate behavior_env
cd /jukebox/graziano/rachel/ai_mind_rep/exp_1/labels
export PROJECT_ROOT=/jukebox/graziano/rachel/ai_mind_rep/exp_1/labels/
python code/analysis/combine_text_data.py --config configs/behavior.json --use_clean false
```

3) **Run behavioral analyses**
```bash
python code/analysis/cross_experiment_comparison.py --config configs/behavior.json
```

## Configuration

`configs/behavior.json` defines:
- subject_ids
- model (e.g., gpt-3.5-turbo)
- temperature (e.g., 0.8)

---

```
exp_1/labels/
├─ code/
│  ├─ analysis/
│  ├─ data_gen/
│  │  ├─ gpt_data_generation.py
│  │  ├─ llm_data_generation.py
│  │  ├─ data_gen_slurm.sh
│  │  └─ utils/
│  │     ├─ config/                 # per-subject condition CSVs
│  │     ├─ prompts/                # topic text files
│  │     ├─ conversation_helpers.py
│  │     ├─ gpt_client.py
│  │     ├─ llama_client.py
│  │     ├─ log_helpers.py
│  │     ├─ prompts_config.py
│  │     ├─ randomize_conds.py
│  │     └─ sim_helpers.py
│  │
├─ configs/
│  └─ behavior.json
│
├─ data/
│  ├─ conds/                        # topic <-> social/nonsocial map
│  └─ <model>/temp_<temperature>/   # generated transcripts: s001.csv ... s050.csv
│
└─ logs/
```
