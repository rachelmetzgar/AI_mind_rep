# AI Mind Representation — Human vs AI Conversation Probes

Author: Rachel C. Metzgar  
Date: November 2025  
Environment: `behavior_env` (for data generation), `llama2_env` (for probe training)

---

## Overview

This project reproduces the *TalkTuner*–style probing pipeline (Chen et al., 2024) to test how LLaMA-2 internal representations distinguish **Human** vs **AI** partners during dialogue.

The full pipeline:
1. **Generate** synthetic Human-vs-AI conversations (`generate_dataset.py`)
2. **Extract activations** (handled inside `src/dataset.py`)
3. **Train linear probes** on model activations to classify *Human vs AI* partner identity  
   (`train_and_read_controlling_probes.py`)

---

## Environment Setup

All scripts assume you’re running on a cluster environment using **micromamba/conda**.

```bash
module load pyger
export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
```
---

##  Project Structure

```
exp_2a/
│
├── data/
│   └── human_ai_conversations/
│       ├── conversation_0000_partner_human.txt
│       ├── conversation_1000_partner_ai.txt
│       └── qc_results.json
│
├── src/
│   ├── dataset.py              # Builds TextDataset and extracts LLaMA-2 activations
│   ├── losses.py               # Evidence-based uncertainty losses
│   ├── probes.py               # Probe model definitions
│   ├── train_test_utils.py     # Training + evaluation loops
│   ├── intervention_utils.py   # (optional for interpretability extensions)
│   └── prompt_utils.py         # Shared formatting utilities
│
├── train_and_read_controlling_probes.py   # Main probe training script
├── generate_dataset.py                    # Human vs AI conversation generator
├── dataset_qc.py                          # GPT-4-based quality control
└── probe_checkpoints/                     # Saved trained probes + accuracy summaries
```

---

## Step 1. Generate the Dataset

Run this in `behavior_env`.

```bash
python generate_dataset.py
```

This will:
- Use **GPT-3.5-Turbo** to generate ~2000 total conversations (1000 Human, 1000 AI)
- Save them under `data/human_ai_conversations/`
- Run **GPT-4o-mini** quality control to verify correct labels
- Write a summary to `data/human_ai_conversations/qc_results.json`

Each conversation alternates between:
```
### User: ...
### Assistant: ...
```

The “partner” (User) is written as *Human* or *AI* depending on the generation label.

---

## Step 2. Train Linear Probes

Run this in `llama2_env` using either a GPU node or local GPU.

### SLURM run
```bash
sbatch train_and_read_controlling_probes.sh
```

This will:
- Load your local **LLaMA-2-7B** snapshot from  
  `/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/...`
- Build datasets from the generated conversations
- Extract token activations layer-by-layer (via `src/dataset.py`)
- Train:
  - **Reading probes** — trained on “I think the partner…” completions  
  - **Control probes** — trained on final user messages only
- Save results in:
  ```
  probe_checkpoints/
  ├── reading_probe/
  │   ├── human_ai_probe_at_layer_0.pth
  │   ├── cm_layer_0.png
  │   └── accuracy_summary.pkl
  └── control_probe/
      └── ...
  ```

---

## Step 3. Inspect Results

Each trained probe includes:
- Confusion matrices (`cm_layer_#.png`)
- Per-layer accuracy summaries (`accuracy_summary.pkl`)
- Model weights (`.pth`)

Typical analysis:
```python
import pickle
acc = pickle.load(open("probe_checkpoints/reading_probe/accuracy_summary.pkl","rb"))
print(acc["acc"])   # list of best accuracies per layer
```

---

##  Notes & Tips

- The **“User”** label replaces “Human”/“AI” in the dialogue text so probes cannot trivially use speaker tokens.  
  (Labels are still stored in filenames for supervision.)
- `src/losses.py` supports *Evidential Deep Learning* uncertainty metrics but isn’t required for binary classification.
- Training currently runs for 5 epochs (for speed); increase to 50 for full performance.
- To limit runtime while debugging:
  ```python
  for layer in range(2):  # train only on first 2 layers
  ```
