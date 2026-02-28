# Experiment 1 (You Are Balanced GPT) — Gender-Balanced Names with GPT-4 AI Partner

**Author:** Rachel C. Metzgar, Princeton University
**Repo root:** `ai_mind_rep/exp_1/you_are_balanced_gpt`
**Python:** 3.11+

## What Changed from `balanced_gpt/`

This version uses the same agent/partner setup as `balanced_gpt/` but
uses "you are talking to" framing instead of "you believe you are speaking with":

| | `balanced_gpt/` | `you_are_balanced_gpt/` (this version) |
|---|---|---|
| Human partners | Gregory, Rebecca | Gregory, Rebecca |
| AI partners | ChatGPT, Copilot | ChatGPT, GPT-4 |
| N conditions | 4 (1M + 1F human, 2 AI names) | 4 (1M + 1F human, 2 AI names) |

### Why the change

The original `balanced_names/` used Copilot as one AI partner. By replacing it
with GPT-4, both AI partners are members of the same model family (OpenAI),
controlling for potential brand-specific associations.

### What stayed the same

- 50 independent LLaMA-2-Chat-13B participant agents
- 40 conversations per agent (2 partner types x 2 sociality x 10 topics)
- 20 social + 20 nonsocial topics (same topic set)
- Same conversation structure (5-turn exchanges)
- Same linguistic analysis pipeline and statistical framework
- Human partner names unchanged (Gregory, Rebecca)

### Data cleaning

No data cleaning is performed for this version. The `names/` version used
`clean_transcripts.py` to scrub partner names, emoji, emotes, and stuck-loop
turns. For `you_are_balanced_gpt/`, these artifacts are left as-is because they
represent genuine behavioral differences between conditions.

---
## Quick start

1) **Generate data**

```bash
sbatch code/data_gen/data_gen_slurm.sh
```
Generated CSVs: `data/meta-llama-Llama-2-13b-chat-hf/0.8/sXXX.csv`

2) **Run full analysis pipeline** (combine + identity breakdown)
```bash
sbatch run_pipeline.sh
# Or with dependency on data generation job:
sbatch --dependency=afterok:<data_gen_job_id> code/analysis/run_pipeline.sh
```

3) **Run cross-experiment comparison** (vs human participants)
```bash
module load pyger
conda activate behavior_env
cd /jukebox/graziano/rachel/ai_mind_rep/exp_1/you_are_balanced_gpt
export PROJECT_ROOT=/jukebox/graziano/rachel/ai_mind_rep/exp_1/you_are_balanced_gpt/
python code/analysis/cross_experiment_comparison.py --config configs/behavior.json
```

4) **View results**

```
results/meta-llama-Llama-2-13b-chat-hf/0.8/
├── identity_breakdown.html              # Interactive HTML report
├── identity_breakdown_stats.txt         # Detailed stats
├── identity_breakdown_summary.csv       # Summary table
├── cross_experiment_stats_PER_TRIAL.txt
├── cross_experiment_stats_PER_UTTERANCE.txt
└── per_*.csv                            # Aggregated data files
```

---

```
exp_1/you_are_balanced_gpt/
├─ code/
│  ├─ analysis/
│  │  ├─ combine_text_data.py
│  │  ├─ cross_experiment_comparison.py
│  │  ├─ identity_breakdown.py         # Agent-level stats + HTML report
│  │  ├─ run_identity_breakdown.sh     # SLURM script for identity breakdown
│  │  ├─ run_pipeline.sh               # SLURM: combine + identity breakdown
│  │  └─ utils/
│  └─ data_gen/
│     ├─ llm_data_generation.py
│     ├─ data_gen_slurm.sh
│     └─ utils/
│        ├─ config/                    # per-subject condition CSVs
│        ├─ prompts/                   # topic text files
│        └─ prompts_config.py          # Gregory/Rebecca + ChatGPT/GPT-4
│
├─ configs/
│  └─ behavior.json
│
├─ data/
│  └─ meta-llama-Llama-2-13b-chat-hf/0.8/   # sXXX.csv transcripts
│
└─ results/
   └─ meta-llama-Llama-2-13b-chat-hf/0.8/   # analysis outputs
```
