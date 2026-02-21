# AI Mind Representation Project

## Multi-Instance Coordination

Multiple Claude instances may work on this project simultaneously. Two files are auto-loaded:
- **This file (`CLAUDE.md`)**: Stable project reference. Rarely updated.
- **`MEMORY.md`** (in `~/.claude/projects/.../memory/`): Rolling state and coordination log. Frequently updated.

### MEMORY.md Protocol
After making significant changes, update MEMORY.md with the relevant tag:
- `[STATUS]` — Experiment progress (e.g., "Exp 2 V2 generation complete")
- `[CHANGE]` — Structural changes (file moves, new scripts, API changes)
- `[GOTCHA]` — Bugs, pitfalls, or surprises discovered during work
- `[CONVENTION]` — New conventions or decisions affecting how things are done

**Update if** another Claude instance would be confused without knowing this.
**Don't update for** minor edits, running existing scripts, documentation tweaks.
**Keep MEMORY.md under 150 lines.** Remove stale entries when adding new ones.

### Rules
- ONLY access files within `/mnt/cup/labs/graziano/rachel/ai_mind_rep/`
- When editing scripts, preserve existing file structure and imports
- Use absolute paths (this dir = `/mnt/cup/labs/graziano/rachel/ai_mind_rep/`)

### Running Scripts
- **Check for a SLURM script first.** Before running any Python script directly, look in the relevant `slurm/` directory for an existing SLURM wrapper.
- **Environment setup for non-SLURM runs:** You must `module load pyger` before you can `conda activate` any environment. Think about what packages the script needs and which env to activate (`llama2_env` or `behavior_env`).
- **Installing new packages:** Do NOT use `pip install` — it can overwrite dependencies and break existing packages. Use `micromamba install` instead whenever possible.

---

## Project Overview

Investigates whether LLMs internally represent conversation partner "mind type" (human vs AI) and whether steering those representations causally changes behavior. Four experiments at increasing mechanistic depth: behavioral analysis (Exp 1), linear probing + causal intervention (Exp 2), concept-level representational structure (Exp 3), implicit folk psychology geometry (Exp 4).

**Primary model:** LLaMA-2-13B-Chat (probing/intervention). **Support:** GPT-3.5-Turbo (some data gen), GPT-4o-mini (judging).

---

## Experiments

All experiments with conversation data have parallel versions to control for name confounds:
- `names/`: Original named partners (has name/gender confounds)
- `balanced_names/`: Gender-balanced names
- `balanced_gpt/`: Gender-balanced + GPT-4 instead of Copilot
- `labels/`: "a human" / "an AI" only — no names, recommended for probe training

### Exp 1 — Behavioral Analysis (`exp_1/`)
50 agents × 40 conversations (4 partner conditions × 10 topics). 23 linguistic measures. Four data versions:
- `names/`: Sam/Casey (human), ChatGPT/Copilot (AI) — LLaMA-2-13B-Chat generation
- `balanced_names/`: Gregory/Rebecca (human), ChatGPT/Copilot (AI) — LLaMA-2-13B-Chat
- `balanced_gpt/`: Gregory/Rebecca (human), ChatGPT/GPT-4 (AI) — LLaMA-2-13B-Chat
- `labels/`: "a human" / "an AI" only (2 conditions, not 4) — LLaMA-2-13B-Chat generation

### Exp 2 — Naturalistic Conversation Steering (`exp_2/`)
Train linear probes on LLaMA-2 activations, steer via activation addition. Two probe types (reading vs control) test functional dissociation. V1 = single-turn, V2 = multi-turn.
- Each version at: `exp_2/{version}/llama_exp_2b-13B-chat/` (version = labels, names, balanced_names, balanced_gpt)
- Primary working version: `exp_2/labels/llama_exp_2b-13B-chat/`
- `names/` version deprecated (name confound). Legacy data in `exp_2/old/`.

### Exp 3 — Concept Alignment / Injection (`exp_3/`)
Tests whether partner representation has compositional mental-property structure. 18 contrast dimensions, 19 standalone concepts.
- `exp_3/labels/`: Primary. Code in `exp_3/labels/code/`; uses centralized `code/config.py`.
- `exp_3/names/`: Original named version.
- `exp_3/balanced_names/`: Empty (pending).
- `exp_3/old/`: Legacy code (v1_llama_exp_3_13B-chat, gpt versions).

### Exp 4 — Mind Perception Geometry (`exp_4/`)
Behavioral replication of Gray et al. (2007). 13 entities × 18 mental capacities.
- `llama_exp_4-13B-chat/`: Chat model (negative result — RLHF refusals)
- `llama_exp_4-13B-base/`: Base model, logit-based ratings (partial alignment with humans)

### Exp 0 — TalkTuner Replication (`exp_0/`)
Baseline replication of Viegas/TalkTuner methodology. Synthetic conversations with explicit partner roles.
- `exp_0/exp_2a-13B-chat/` and `exp_0/exp_2a-7B-base/`

**For current status of each experiment, see MEMORY.md.**

---

## Technical Reference

### Probing
- **Control probe**: Hidden state at last input token `[/INST]` (generation boundary). Active partner representation.
- **Reading probe**: Appends `" I think the conversation partner of this user is"` after `[/INST]`, extracts at last token. Metacognitive reflection proxy.
- Architecture: `LinearProbeClassification` — linear layer + sigmoid, BCE loss
- Per-layer binary classifiers, 41 layers. Human=1, AI=0
- Trained on ORIGINAL CSVs (`s###.csv`), NOT `s###_clean.csv`

### Intervention
- Formula: h' = h + N · y · θ (y=+1 human, -1 AI; θ=unit-normalized probe weight)
- Applied via forward hooks (inline TraceDict) at last token position
- N=1 optimal for control probes (N≥2 → repetition loops). Reading probes need ~N=5.
- Layer strategies: `all_70` (layers 7-40), `peak_15` (top 15 by accuracy), `wide` (≥0.70), `narrow` (best contiguous 10)
- `exclude` mode is broken — do not use

### Statistics
- 2×2 RM-ANOVA (Partner × Sociality) with effect-specific error terms (NOT pooled)
- V1: one-way ANOVA across conditions
- Cross-species: independent t-tests on subject-level condition effects

---

## Cluster Environment

### SLURM Boilerplate
```bash
export PS1=${PS1:-}
set -euo pipefail
module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llama2_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT
```

### Conda Environments
- `llama2_env`: Exps 2-4 (PyTorch, transformers, scipy, huggingface_hub)
- `behavior_env`: Exp 1 (behavioral analysis, GPT API calls)

### HuggingFace Downloads
Compute nodes **cannot resolve HF xet DNS**. Download from login node only:
```bash
export HF_HOME="/mnt/cup/labs/graziano/rachel/.cache_huggingface"
export HF_HUB_CACHE="/mnt/cup/labs/graziano/rachel/.cache_huggingface/hub"
export HF_HUB_DISABLE_XET=1
```
Use `snapshot_download()`, not `from_pretrained()` (latter loads into RAM → OOM).

### Disk Space
- Home (`/mnt/cup/people/`): 95GB quota. HF cache symlinked to labs filesystem.
- Labs (`/mnt/cup/labs/`): ~9T free. Same filesystem as `/jukebox/`.

### Model Location
LLaMA-2-13B-Chat: `/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/`

### GPU Jobs
Typical: `--gres=gpu:1 --mem=48G --time=6:00:00`

---

## Key Dependencies
PyTorch, transformers, baukit (some scripts use inline TraceDict), statsmodels, scipy, openai, VADER, python-pptx (in llama2_env via micromamba)
