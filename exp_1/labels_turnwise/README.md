# Experiment 1 (Labels Turnwise) — Label-Only Belief + Turn-Level Identity Prefix

**Author:** Rachel C. Metzgar, Princeton University
**Repo root:** `ai_mind_rep/exp_1/labels_turnwise`
**Python:** 3.11+

## What Changed from `labels/` and `balanced_gpt/`

This version combines two approaches:
- Like `labels/`: the system prompt says "You believe you are speaking with a Human/an AI" (no individual names)
- Like `balanced_gpt/`: each partner turn is prefixed with the identity label ("Human:" or "AI:") instead of the generic "Partner:"

| | `labels/` | `balanced_gpt/` | `labels_turnwise/` (this version) |
|---|---|---|---|
| System prompt | "...speaking with a Human/an AI" | "...speaking with Gregory (a Human)" | "...speaking with a Human/an AI" |
| Turn prefix | "Partner:" | "Gregory:" / "ChatGPT:" | "Human:" / "AI:" |
| Identity reinforcement | System prompt only | Every turn (via name) | Every turn (via label) |

### Why the change

The `labels/` version mentions identity only once in the system prompt — as conversations grow, this signal fades. `labels_turnwise` reinforces the Human/AI label at every partner turn, testing whether stronger identity cueing amplifies behavioral effects.

### What stayed the same

- 50 independent LLaMA-2-Chat-13B participant agents
- 40 conversations per agent (2 partner types x 2 sociality x 10 topics)
- 20 social + 20 nonsocial topics (same topic set)
- Same conversation structure (5-turn exchanges)
- Same linguistic analysis pipeline and statistical framework

---
## Quick start

1) **Generate data**

```bash
sbatch code/data_gen/data_gen_slurm.sh
```
Generated CSVs: `data/meta-llama-Llama-2-13b-chat-hf/0.8/sXXX.csv`

2) **Run full analysis pipeline** (combine + identity breakdown)
```bash
sbatch --dependency=afterok:<data_gen_job_id> code/analysis/run_pipeline.sh
```

3) **View results**

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
exp_1/labels_turnwise/
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
│        └─ prompts_config.py          # Human/AI labels (no individual names)
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
