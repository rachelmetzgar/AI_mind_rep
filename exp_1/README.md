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



2) **Combine per-subject files and put combined file in analysis dir**
```bash
python code/analysis/combine_text_data.py --config configs/behavior.json
```

3) **Run behavioral analyses**
```bash
pyger
conda activate behavior_env
export PATH=~/.conda/envs/behavior_env/bin:$PATH
python code/analysis/cross_experiment_comparison.py --config configs/behavior.json
```

## Configuration

`configs/behavior.json` defines:
- subject_ids
- model (e.g., gpt-3.5-turbo)
- temperature (e.g., 0.8)

---

exp_1/
├─ code/
│  ├─ analysis/
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
│  ├─ exp_csv_human/                # human data
│  └─ <model>/temp_<temperature>/   # generated transcripts: s001.csv … s050.csv
│        └─ combine_text_data.csv   # combined LLM transcripts
│
├─ logs/
│
└─ results/
   └─ <model>/<temperature>/ # results should save here
---