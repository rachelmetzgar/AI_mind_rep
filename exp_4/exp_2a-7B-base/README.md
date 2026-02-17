# AI Mind Representation — Human vs AI Conversations

Author: Rachel C. Metzgar, Princeton University

This experiment adapts the TalkTuner-style probing framework (Chen et al., 2024) method to study how large language models internally encode the identity or "mind type" of their conversational partner: Human vs AI.

This repository contains modified and extended code originally from  Chen et al.’s TalkTuner implementation (MIT Licensed). The original  copyright (c) 2024 Yida Chen is retained as required.

The core idea:
- Use synthetic dialogues where the user is explicitly specified as a Human or an AI.
- Extract hidden activations from a target model (e.g., LLaMA-2-Chat-13B).
- Train linear probes to read out partner type and derive steering directions.
- Test whether shifting activations along these directions causally changes behavior.

This will (1) replicate and extend prior probing/intervention work and (2) give hands-on experience with causal interpretability methods on conversation data linked to exp_1.

Status:
- Phase 1 (synthetic conversation generation): implemented.
- Phases 2–5: planned / work in progress.

--------------------------------------------------------------------------------
## Code Location
--------------------------------------------------------------------------------

Repository root:
    ai_mind_rep/exp_2a

Key scripts:
    1_generate_human_ai_dataset.py
    2_train_and_read_controlling_probes.py
    3_run_partner_intervention_experiment.pytest
    demo_partner_intervention.py            


Supporting modules:
    src/dataset.py
    src/losses.py
    src/probes.py
    src/train_test_utils.py
    src/intervention_utils.py
    src/prompt_utils.py

--------------------------------------------------------------------------------
## Environment Setup
--------------------------------------------------------------------------------

Two Conda/mamba environments are used:

    behavior_env   - for synthetic data generation and QC
    llama2_env     - for activation extraction, probing, and interventions

Recreate them from the provided YAMLs (stored under envs/):

    conda env create -f envs/behavior_env.yml
    conda env create -f envs/llama2_env.yml

--------------------------------------------------------------------------------
## Step by step analysis pipeline
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
### Phase 1 – Synthetic Conversation Generation (IMPLEMENTED)
--------------------------------------------------------------------------------

Code:
    1_generate_human_ai_dataset.py

Goal:
    Create a labeled conversation corpus that reliably expresses the target
    attribute: user = human vs user = AI. This corpus will be used later to
    read out and control that attribute in a separate model.

Run this in `behavior_env`.

```bash
python 1_generate_dataset.py
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

Why:
    We need a high-consistency synthetic dataset so that any separability in
    activations is actually tracking the intended label (human vs ai), not noise.

--------------------------------------------------------------------------------
### Phase 2 & 3  Train Reading Probes & Control Probes and Intervention Vectors  (WIP)
--------------------------------------------------------------------------------

Code: 2_train_and_read_controlling_probes.py 
      train_and_read_controlling_probes.sh

Goal:
    Determine whether a linear direction in hidden space encodes partner type.
    Move from reading the representation to actively steering it.
    
What the script does:
    - Loads LLaMA-2-7B
    - Builds the activation dataset (via TextDataset in src/dataset.py)
    - For each transformer layer:
         • Trains a **reading probe** on the “I think the partner…” token  
         • Trains a **control probe** on the last user message token
    - Saves probe weights + confusion matrices + accuracy summaries under:
        data/probe_checkpoints/
            reading_probe/
            control_probe/

#### Train Reading Procedure:

    - For each layer, train a logistic-regression probe:
          p_theta(x) = sigma(<x, theta>)
      on 80 % train / 20 % validation of the collected hidden states.
    - Train one probe per layer using the reading-prompt activations.
    - Evaluate AUROC / accuracy as a function of layer depth.
      Expect accuracy to rise in later layers if partner type is strongly represented.

Why:
    This mirrors prior findings that demographic or user attributes are
    linearly decodable and layer-dependent in LLMs.
      
#### Control Probes and Intervention Vector Procedure:

    - For each attribute (here: partner type), take the weight vector theta
      from the most accurate reading probe.
    - Define steering vectors.

Why:
    These vectors define latent directions in activation space that we can
    add or subtract to manipulate the underlying representation of partner type.

#### Why both probes?
    - Reading probes evaluate *decodability*: whether the model represents
      partner identity in its hidden states.
    - Control probes are used to derive **steering vectors** that shift the
      model’s internal representation during causal interventions (Phase 3).


#### TO RUN
Run this in `llama2_env` using either a GPU node or local GPU.

#### SLURM run
```bash
sbatch train_and_read_controlling_probes.sh
```
--------------------------------------------------------------------------------
### Demo Script – Minimal Partner-Type Intervention
--------------------------------------------------------------------------------
Code: demo_partner_intervention.py

Purpose:
    A small-scale sanity check for the intervention mechanism.
    Loads LLaMA-2-7B and the trained control probes, applies activation-level
    steering (+N·v and -N·v) to a selected layer range, and prints baseline
    vs HUMAN-steered vs AI-steered responses for a few example prompts.

When to use:
    - To quickly verify probe quality
    - To confirm that residual-stream interventions produce interpretable shifts
    - Before running the full causal intervention experiment

How to run:
```bash
    conda activate llama2_env
    python demo_partner_intervention.py
```
This script runs only on a handful of prompts and prints results to stdout.
It does not perform GPT-based causal evaluation.

--------------------------------------------------------------------------------
### Phase 4 – Causal Intervention Test (WIP)
--------------------------------------------------------------------------------
Code: 
    3_causality_test_on_mind.py      # full causal evaluation (GPT judge)
    causality_test_on_mind.sh 

Goal:
    Test whether shifting activations along the steering vector actually
    changes the model’s behavior in attribute-consistent ways.

Procedure:
    - Pick a held-out set of attribute-sensitive questions (e.g., for human vs ai,
      or the human/AI tasks from exp_1).
    - For each input, run inference under three conditions:
        1) Baseline (no shift)
        2) +N · v  (steer toward "human")
        3) −N · v  (steer toward "ai")
    - Generate full responses (e.g., with greedy or low-temperature decoding).
    - Ask GPT-4 or human annotators to classify which variant corresponds to
      which attribute. Success criterion: classifier matches > 80 % of pairs.

Why:
    This reproduces the causal check from Chen/TalkTuner: it is not enough
    that probes decode a feature; we want the direction to have behavioral
    consequences when applied to activations.

--------------------------------------------------------------------------------
### Phase 5 – Analyses (PLANNED)
--------------------------------------------------------------------------------

Analysis components:

    - Layer-wise accuracy:
        Locate the layer(s) where partner-type representation is strongest.

    - Hidden-state visualization (PCA / UMAP):
        Visualize separability of human vs ai activations and how points move
        under steering (+N·v, −N·v).

    - Control vs reading probe comparison:
        Check whether control probes (trained on task-context activations)
        steer more effectively than reading probes alone.

    - Qualitative sample inspection:
        Collect example outputs before/after intervention and document
        stylistic / reasoning shifts (e.g., changes in self-references,
        politeness, or descriptions of the partner).

--------------------------------------------------------------------------------
## Interpretation
--------------------------------------------------------------------------------

Interpretation logic:

    - If linear probes decode partner type AND activation additions systematically
      alter model output in the expected direction:
          → Evidence for an internal latent variable the model uses to track
            partner identity or "mind type."

    - If only decodability is present but interventions have no effect:
          → The probe may be reading out a passive correlation, not a causal
            control dimension.

    - If steering succeeds but reading probes are weak:
          → Partner information may be more distributed and better captured
            by task-context activations (e.g., final-user-message tokens)
            than by the explicit reading-prompt token.

This experiment will help clarify how LLMs internalize the idea of "who" they
are talking to, and how those internal variables relate to the kinds of human
belief and mind-interpretation effects studied in exp_1.

--------------------------------------------------------------------------------
## References & Attribution
--------------------------------------------------------------------------------

This project adapts the probing and activation-intervention methodology introduced in:

Chen, Y., Wu, A., DePodesta, T., Yeh, C., Li, K., Marin, N. C., ... & Viégas, F. (2024). 
Designing a dashboard for transparency and control of conversational AI. 
arXiv preprint arXiv:2406.07882.

We additionally adapt components of the official TalkTuner codebase:
https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/tree/main

Their repository’s structure for conversation generation, probe training,
and activation steering served as the primary template for our approach.