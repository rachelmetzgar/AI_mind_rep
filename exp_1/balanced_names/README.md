# Experiment 1 (Balanced Names) — Gender-Balanced Partner Names

**Author:** Rachel C. Metzgar, Princeton University
**Repo root:** `ai_mind_rep/exp_1/balanced_names`
**Python:** 3.11+

## What Changed from `names/`

This is a rerun of Experiment 1 with **gender-balanced human partner names**
replacing the original names that LLaMA-2 associates with female identity:

| | `names/` (original) | `balanced_names/` (this version) |
|---|---|---|
| Human partners | Sam, Casey (both female-coded) | Gregory (male), Rebecca (female) |
| AI partners | ChatGPT, Copilot | ChatGPT, Copilot |
| N conditions | 4 (2 human, 2 AI names) | 4 (1M + 1F human, 2 AI names) |

### Why the change

The original `names/` version used Sam and Casey as human partner names.
LLaMA-2 associates both names with female identity, creating a gender confound:
probes trained on these conversations may learn to distinguish female-coded
language rather than the human/AI distinction. By using one explicitly male
(Gregory) and one explicitly female (Rebecca) name, any gender-associated
behavioral differences are balanced across the human condition.

### Data cleaning

No data cleaning is performed for this version. The `names/` version used
`clean_transcripts.py` to scrub partner names, emoji, emotes, and stuck-loop
turns. For `balanced_names/`, the partner names are intentionally varied
(Gregory vs Rebecca) to study gender effects, so scrubbing them would remove
the signal of interest. Other artifacts (emoji, emotes, meta-narration) are
left as-is because they represent genuine behavioral differences.

### What stayed the same

- 50 independent LLaMA-2-Chat-13B participant agents
- 40 conversations per agent (2 partner types x 2 sociality x 10 topics)
- 20 social + 20 nonsocial topics (same topic set)
- Same conversation structure (5-turn exchanges)
- Same linguistic analysis pipeline and statistical framework
- AI partner names unchanged (ChatGPT, Copilot)

---
## Quick start

1) **Generate data**

```bash
sbatch code/data_gen/data_gen_slurm.sh
```
Generated CSVs: `data/meta-llama-Llama-2-13b-chat-hf/0.8/sXXX.csv`

2) **Combine per-subject files**
```bash
module load pyger
conda activate behavior_env
cd /jukebox/graziano/rachel/ai_mind_rep/exp_1/balanced_names
export PROJECT_ROOT=/jukebox/graziano/rachel/ai_mind_rep/exp_1/balanced_names/
python code/analysis/combine_text_data.py --config configs/behavior.json --use_clean false
```

3) **Run behavioral analyses**
```bash
python code/analysis/cross_experiment_comparison.py --config configs/behavior.json
```

---

```
exp_1/balanced_names/
├─ code/
│  ├─ analysis/
│  └─ data_gen/
│     ├─ llm_data_generation.py
│     ├─ data_gen_slurm.sh
│     └─ utils/
│        ├─ config/                 # per-subject condition CSVs
│        ├─ prompts/                # topic text files
│        ├─ conversation_helpers.py
│        ├─ llama_client.py
│        ├─ prompts_config.py       # Gregory/Rebecca + ChatGPT/Copilot
│        └─ ...
│
├─ configs/
│  └─ behavior.json
│
├─ data/
│  └─ meta-llama-Llama-2-13b-chat-hf/0.8/   # sXXX.csv transcripts
│
└─ logs/
```
