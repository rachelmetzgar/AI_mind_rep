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
- ONLY access files within `/mnt/cup/labs/graziano/rachel/mind_rep/`
- When editing scripts, preserve existing file structure and imports
- Use absolute paths (this dir = `/mnt/cup/labs/graziano/rachel/mind_rep/`)
- **Reports require regeneration code.** When producing an HTML/MD report, always create a standalone Python script that can regenerate it from saved data (`.npz`/`.csv`). Place it in the relevant `comparisons/code/` directory. Exception: user says "quick analysis" or explicitly says not to.

### Running Scripts
- **Check for a SLURM script first.** Before running any Python script directly, look in the relevant `slurm/` directory for an existing SLURM wrapper.
- **ALWAYS use SLURM for heavy computation.** Any script expected to run longer than ~5 minutes MUST be submitted via SLURM, not run directly on the login node. This includes bootstrapping, loading/processing large activation files, training probes, generation, judging, and any iterative analysis over many layers/dimensions. If no SLURM wrapper exists, **create one** in the relevant `slurm/` directory before running.
  - **OK to run directly on login node:** Quick analysis scripts (<5 min), report/figure generation from pre-computed CSVs, file I/O utilities, small data inspection.
  - **Must go through SLURM:** Probe training, activation extraction, bootstrap analyses, causality generation/judging, behavioral analysis, alignment analysis over activations, anything loading model weights or large `.npz`/`.pkl` activation files.
  - When creating a new SLURM script, use the boilerplate from the "SLURM Boilerplate" section below. For CPU-only jobs (no GPU), use `--mem=16G --time=2:00:00` as a starting point (adjust as needed).
- **Environment setup for non-SLURM runs:** You must `module load pyger` before you can `conda activate` any environment. Think about what packages the script needs and which env to activate (`llama2_env` or `behavior_env`).
- **Installing new packages:** Do NOT use `pip install` — it can overwrite dependencies and break existing packages. Use `micromamba install` instead whenever possible.

---

## Project Overview

Investigates whether LLMs internally represent conversation partner "mind type" (human vs AI) and whether steering those representations causally changes behavior. Four experiments at increasing mechanistic depth: behavioral analysis (Exp 1), linear probing + causal intervention (Exp 2), concept-level representational structure (Exp 3), implicit folk psychology geometry (Exp 4).

**Primary model:** LLaMA-2-13B-Chat (probing/intervention). **Support:** GPT-3.5-Turbo (some data gen), GPT-4o-mini (judging).

---

## Experiments

All experiments with conversation data have parallel versions to control for confounding factors.

### Exp 1 — Behavioral Analysis (`exp_1/`)
50 agents × 40 conversations (4 partner conditions × 10 topics). 23 linguistic measures. Multiple versions.

### Exp 2 — Naturalistic Conversation Steering (`exp_2/`)
Train linear probes on LLaMA-2 activations, steer via activation addition. Two probe types (reading vs control) test functional dissociation. V1 = single-turn, V2 = multi-turn.

### Exp 3 — Concept Alignment / Injection (`exp_3/`)
Tests whether partner representation has compositional mental-property structure. contrast dimensions, standalone concepts.

### Exp 4 — Mind Perception Geometry (`exp_4/`)
Behavioral + geometric replication of Gray et al. (2007). 13 entities × 18 mental capacities.

### Exp 0 — TalkTuner Replication (`exp_0/`)
Baseline replication of Viegas/TalkTuner methodology. Synthetic conversations with explicit partner roles.

**For current status of each experiment, see MEMORY.md.**

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
- Labs (`/mnt/cup/labs/`): Same filesystem as `/jukebox/`.

### Model Location
LLaMA-2-13B-Chat: `/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/`

### GPU Jobs
Typical: `--gres=gpu:1 --mem=48G --time=6:00:00`

---

## Key Dependencies
PyTorch, transformers, baukit (some scripts use inline TraceDict), statsmodels, scipy, openai, VADER, python-pptx (in llama2_env via micromamba)
