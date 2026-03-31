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

Investigates whether LLMs internally represent conversation partner "mind type" (human vs AI) and whether steering those representations causally changes behavior. Eight experiments at increasing mechanistic depth: TalkTuner-style probing baseline (Exp 0), behavioral analysis across partner conditions (Exp 1), linear probing + causal intervention on naturalistic conversations (Exp 2), concept-level representational structure of mental properties (Exp 3), implicit folk psychology geometry replicating Gray et al. (Exp 4), RSA on mental state attribution sentences (Exp 5), multi-agent belief propagation tracking via representational geometry (Exp 6), and mental-state concept deployment during theory of mind reasoning (Exp 7, planned)

**Primary models:** LLaMA-2-13B-Chat,LLaMA-2-13B-Base, LLaMA-3-8B-Instruct, LLaMA-3-8B-Base. **Support:**  (exp5), GPT-3.5-Turbo (some data gen), GPT-4o-mini (judging).

---

## Experiments

**For current status of each experiment, see README.md and MEMORY.md.**

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
Typical: `--gres=gpu:1 --mem=64G --time=6:00:00`

**SLURM `--mem` is CPU RAM, not GPU VRAM.** `from_pretrained()` loads the full model into CPU RAM before `.half().to(device)` moves it to GPU. This means you need enough `--mem` to hold the model weights in their on-disk dtype (usually bf16/fp32) *plus* Python overhead. Rule of thumb: **`--mem` should be at least 3× the on-disk model size** (e.g., a 20 GB model needs ~64G). Using 32G for models larger than ~8 GB on disk will OOM during checkpoint loading.

---

## Key Dependencies
PyTorch, transformers, baukit (some scripts use inline TraceDict), statsmodels, scipy, openai, VADER, python-pptx (in llama2_env via micromamba)

## Target Experiment Directory Structure

```
exp_N/
  code/
    config.py           # paths, model, versions — single source of truth
    utils/              # shared utilities for this experiment
    slurm/              # SLURM wrappers for every pipeline script
    1_step_name.py      # numbered pipeline scripts
    1a_substep.py       # sub-steps labeled a, b, c...
    1a_step_summary_generator.py  # summary/report generators named after their step
  results/
    {model}/            # e.g. llama2_13b_chat/
      {version}/        # e.g. balanced_gpt/, nonsense_codeword/; does not apply to exp6
        {analysis}/     # e.g. probe_training/, alignment/
          data/         # generated artifacts (npz, pkl, csv) used by later steps
          figures/      # figures from summary generators
      comparisons/      # within-model cross-version comparisons
    comparisons/        # cross-model comparisons (only if multiple models)
  logs/
    {analysis}/         # log files organized by analysis step
  writeup/              # paper text, notes
  archive/              # archived old structure (pre-refactor snapshots)
  README.md             # purpose, conditions, decisions, pipeline description
```

## Naming Conventions

- **Pipeline scripts**: `N_short_description.py` (e.g. `1_elicit_concept_vectors.py`)
- **Sub-steps**: `Na_description.py` (e.g. `1a_probe_summary.py`)
- **Summary generators**: `Na_description_summary_generator.py`
- **Figures**: saved in `figures/` subfolder wherever the summary lives
- **Names minimal given context**: no redundant prefixes (file inside `V1_causality/` doesn't need `_v1` in name)

## Structural Principles

1. **Model-scoped results**: results/{model}/ — comparisons nest under model unless cross-model
2. **Files near logical parent**: artifacts live inside the analysis that produced them
3. **Version flags**: scripts that depend on generated data take `--version` flag, save to version subfolders
4. **Config is truth**: every script imports paths from config.py, never hardcodes
5. **Cross-experiment references**: if script A references exp_B data, import exp_B's config to find the path
6. **Archive, don't delete**: old structure goes to `archive/pre_refactor/`

## What Goes Where

- **results/**: everything generated by the pipeline (data + human-readable output)
  - **results/.../data/**: machine artifacts (npz, pkl, csv) — gitignored
  - **results/.../figures/**: generated figures
  - Human-readable summaries (html, md) live at the analysis level (directory top)
  - **Convention**: Reports (HTML/MD) at directory top level; computed data (CSV/NPZ/JSON) in `data/` subfolder; figures in `figures/` subfolder. Variant outputs use filename suffixes (e.g. `summary_top_align.json`) rather than separate directories.
- **code/utils/**: tracked input data or utility scripts
- **logs/**: SLURM logs, organized by analysis step — gitignored
- **archive/**: old code/results snapshots — gitignored

## .gitignore Rules

Tracked: results/, code/, writeup/, README.md
Not tracked: archive/, logs/, **/data/, **/activations/, **/checkpoints/, etc.
This is in the project level gitignore file.
