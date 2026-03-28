# Plan: Add Gemma-2-9B-IT, Qwen-2.5-7B-Instruct, and Qwen3-8B to Exp 4

**Created:** 2026-03-28
**Status:** Not started

## Context

Exp 4 currently runs 4 LLaMA models (2×13B, 2×8B) across 4 experimental branches. We're adding 3 new models to broaden the model diversity. All 3 are already downloaded to the HF cache. Three LLaMA-3 jobs are currently running — we won't interfere with those.

Storage impact: ~2 GB results per model × 4 branches = ~8 GB per model, ~24 GB total. Negligible on the 8.8 TB labs filesystem.

## New Models

| Key | HF ID | Family | is_chat | hidden_dim | layers | Disk |
|-----|--------|--------|---------|------------|--------|------|
| `gemma2_9b_it` | `google/gemma-2-9b-it` | `gemma2` | True | 3584 | 42 | ~20 GB |
| `qwen25_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | `qwen2` | True | 3584 | 28 | ~17 GB |
| `qwen3_8b` | `Qwen/Qwen3-8B` | `qwen3` | True | 4096 | 36 | ~18 GB |

HF cache snapshots:
- `models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819`
- `models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28`
- `models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`

---

## Steps

### Step 0: Commit Current State
- [ ] Commit all exp_4 changes to git so the current working state is saved before modifications.

### Step 1: Update `config.py`
- [ ] Add 3 entries to `VALID_MODELS` tuple
- [ ] Add 3 entries to `MODELS` dict (paths, families, dims, layers)

### Step 2: Add Chat Templates to `utils.py`
- [ ] `gemma2_prompt()` — no system role, fold into first user msg (like LLaMA-2)
  ```
  <start_of_turn>user\n{system_prefix}{content}<end_of_turn>\n<start_of_turn>model\n
  ```
- [ ] `qwen2_prompt()` — ChatML format, native system role
  ```
  <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
  ```
- [ ] `qwen3_prompt()` — same ChatML but append `/no_think` to system msg to disable thinking mode
- [ ] Extend `format_chat_prompt()` dispatcher for `"gemma2"`, `"qwen2"`, `"qwen3"`

### Step 3: Update `report_utils.py`
- [ ] Add to `MODEL_COLORS`: gemma2=#4daf4a, qwen25=#a65628, qwen3=#f781bf
- [ ] Add to `MODEL_LABELS`: "Gemma-2-9B-IT", "Qwen-2.5-7B-Instruct", "Qwen3-8B"

### Step 4: Verify No Analysis Scripts Need Changes
- [ ] Grep for hardcoded "llama" conditionals in analysis scripts
- [ ] Confirm all dispatch through `config.MODEL_FAMILY` / `format_chat_prompt()`

### Step 5: Create SLURM Scripts (~33 new files)
Template from existing `*_llama3_instruct.sh`, change job-name, logs, `--model` arg.

**gray_replication/** (6):
- [ ] `1_pairwise_{gemma2_it,qwen25_instruct,qwen3}.sh`
- [ ] `3_individual_{gemma2_it,qwen25_instruct,qwen3}.sh`

**gray_simple/** (6):
- [ ] `1_extract_entities_{gemma2_it,qwen25_instruct,qwen3}.sh` (GPU)
- [ ] `2_neural_pca_{gemma2_it,qwen25_instruct,qwen3}.sh` (CPU)

**human_ai_adaptation/** (6):
- [ ] `1_gray_chars_{gemma2_it,qwen25_instruct,qwen3}.sh`
- [ ] `2_gray_names_only_{gemma2_it,qwen25_instruct,qwen3}.sh`

**expanded_mental_concepts/** (15):
- [ ] `behavioral_pca_{gemma2_it,qwen25_instruct,qwen3}.sh`
- [ ] `activation_rsa_{gemma2_it,qwen25_instruct,qwen3}.sh`
- [ ] `concept_rsa_{gemma2_it,qwen25_instruct,qwen3}.sh` (may need --time=8:00:00)
- [ ] `standalone_alignment_{gemma2_it,qwen25_instruct,qwen3}.sh`
- [ ] `contrast_alignment_{gemma2_it,qwen25_instruct,qwen3}.sh`

### Step 6: Smoke Test
- [ ] `python config.py` — verify all 7 models
- [ ] Test chat templates for each new family
- [ ] Load tokenizers, verify special tokens in vocab
- [ ] Optional: single-prompt dry run per model

### Step 7: Submit Jobs

**Wave 1** (independent GPU jobs — 24 total):
- [ ] gray_replication/1_pairwise × 3 models
- [ ] gray_simple/1_extract_entities × 3 models
- [ ] human_ai_adaptation/1_gray_chars × 3 models
- [ ] expanded_mental_concepts/{behavioral_pca, activation_rsa, concept_rsa, standalone_alignment, contrast_alignment} × 3 models

**Wave 2** (depends on Wave 1 — 9 total):
- [ ] gray_replication/3_individual × 3
- [ ] gray_simple/2_neural_pca × 3 (CPU)
- [ ] human_ai_adaptation/2_gray_names_only × 3

**Wave 3** (CPU post-processing per model):
- [ ] compute_excl_pca, compute_human_comparisons, make_loadings_bar_chart, make_condition_reports
- [ ] matched_behavioral_pca, activation_pca
- [ ] Per-model report generators

**Wave 4** (cross-model reports):
- [ ] Re-run all 7 comparison scripts (auto-discover new models)

---

## Files Modified (3)
- `exp_4/code/config.py`
- `exp_4/code/utils/utils.py`
- `exp_4/code/utils/report_utils.py`

## Files Created (~33)
- SLURM scripts in `exp_4/code/slurm/{branch}/`

## Files NOT Modified
- All ~20 analysis Python scripts (model-agnostic by design)
- All entity/character definition files
- All report generator scripts (use `ALL_MODELS` dynamically)

## Risks
- **Qwen3 thinking mode**: If `/no_think` doesn't suppress, strip `<think>...</think>` in parse_rating
- **Gemma-2 special tokens**: If manual template tokens aren't in vocab, fall back to `tokenizer.apply_chat_template()`
- **float16 numerics**: `.half()` on bfloat16 models is fine but monitor first job outputs for NaN
