# Experiment 0: TalkTuner Replication

Baseline replication of the TalkTuner probing and intervention methodology (Chen et al. 2024) applied to conversation partner identity (Human vs AI). Serves as a proof-of-concept for the broader project and provides hands-on experience with causal interpretability methods on conversation data.

## Overview

- Generate synthetic conversations where the partner is explicitly labeled as Human or AI (GPT-3.5-Turbo)
- Extract hidden activations from LLaMA-2 and train linear probes to decode partner type
- Derive steering vectors and test whether shifting activations causally changes model behavior
- Evaluate interventions with GPT-4o-mini judge

## Models

Two parallel implementations:
- **`exp_2a-13B-chat/`** — LLaMA-2-13B-Chat (primary)
- **`exp_2a-7B-base/`** — LLaMA-2-7B-Base (comparison)

Each contains the same pipeline with identical structure.

## Pipeline

```
Phase 1: Synthetic conversation generation    [CPU, behavior_env]
  → 2,001 labeled conversations (GPT-3.5-Turbo + GPT-4o-mini QC)

Phase 2-3: Probe training + steering vectors  [GPU, llama2_env]
  → Reading probes (at "I think the partner..." token)
  → Control probes (at last user message token)
  → Per-layer accuracy, confusion matrices

Phase 4: Causal intervention test             [GPU, llama2_env]
  → Baseline, human-steered, AI-steered outputs
  → GPT-4 pairwise evaluation

Phase 5: Analysis                             [planned]
  → Layer-wise accuracy, PCA/UMAP, qualitative inspection
```

## File Structure

```
exp_0/
  exp_2a-13B-chat/              # LLaMA-2-13B-Chat
    1_generate_human_ai_dataset.py
    2_train_and_read_controlling_probes.py
    3_causality_test_on_mind.py
    demo_partner_intervention.py
    src/
      dataset.py                # conversation loading, activation extraction
      probes.py                 # linear probe classifier
      losses.py                 # EDL MSE loss
      train_test_utils.py       # training/validation loops
      intervention_utils.py     # steering direction computation
      prompt_utils.py           # prompt formatting
    slurm/
      train_and_read_controlling_probes.sh
      causality_test_on_mind.sh
    data/
      human_ai_conversations/   # 2,001 synthetic conversations
      probe_checkpoints/        # trained probes (reading + control)
      intervention_results/     # steered outputs + GPT evaluations
  exp_2a-7B-base/               # LLaMA-2-7B-Base (same structure)
  README.md
```

## Key Findings

Intervention examples show interpretable behavioral shifts:
- **Baseline:** Formal, neutral, professional tone
- **Human-steered:** Warm, personal language, emojis, encouragement
- **AI-steered:** Clinical, formal, explicit disclaimers

## Attribution

Adapts the probing and activation-intervention methodology from:

Chen, Y., Wu, A., DePodesta, T., et al. (2024). Designing a dashboard for transparency and control of conversational AI. *arXiv:2406.07882*.

Code adapted from the [TalkTuner repository](https://github.com/yc015/TalkTuner-chatbot-llm-dashboard) (MIT License, copyright 2024 Yida Chen).
