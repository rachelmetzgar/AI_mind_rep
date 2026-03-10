# Exp 5 — Implementation Plan

## Directory Structure

```
exp_5/
  code/
    config.py                              # Paths, model config, constants
    stimuli.py                             # STIMULI list + condition/category metadata
    utils/
      __init__.py
      rsa.py                               # RDM computation, RSA, partial RSA, permutation tests
    slurm/
      1_extract_activations.sh             # GPU job
      2_rsa_analysis.sh                    # CPU job (simple + partial + category RSA)
    1_extract_activations.py               # Phase 1: forward pass, save activations
    2_rsa_analysis.py                      # Phase 2: all three RSA analyses
    2a_rsa_report_generator.py             # Generate HTML report from saved results
  results/
    llama2_13b_chat/
      activations/data/                    # .npz files (336 x 41 x 5120)
      rsa/data/                            # RSA results (.csv, .npz)
      rsa/figures/                         # Layer profile plots
      rsa/rsa_report.html                  # Summary report
  logs/
    activations/                           # SLURM logs for phase 1
    rsa/                                   # SLURM logs for phase 2
  writeup/
  README.md
  IMPLEMENTATION_PLAN.md
```

## Pipeline Scripts

### `config.py` — Configuration
- `ROOT_DIR`, `PROJECT_ROOT`, model path, hidden dim (5120), n_layers (41)
- `set_model()` / `add_model_argument()` (default: `llama2_13b_chat`)
- Output path helpers: `ensure_dir()`, `get_results_dir(phase)`
- Constants: `N_ITEMS = 56`, `N_CONDITIONS = 6`, `N_SENTENCES = 336`, `CONDITION_LABELS`, `CATEGORY_LABELS`

### `stimuli.py` — Stimulus Definitions
- Full `STIMULI` list (56 dicts, each with 6 condition sentences)
- Helper functions:
  - `get_all_sentences()` -> list of (item_id, condition, category, sentence) tuples, ordered
  - `get_condition_indices(condition)` -> indices into the 336-sentence list
  - `get_category_indices(category)` -> indices within condition 1 (56 items)

### `utils/rsa.py` — RSA Utilities
- `compute_rdm_correlation(activations)` — correlation distance RDM (1 - Pearson r)
- `compute_rdm_cosine(activations)` — cosine distance RDM (alternative)
- `extract_lower_triangle(rdm)` — vectorize lower triangle
- `build_model_rdms(n_items, n_conditions)` — construct all 8 model RDMs (A, B, C, D, E, F, G, H)
- `build_category_rdm(n_items_per_cat, n_cats)` — Model Cat for Analysis 3
- `simple_rsa(neural_rdm_vec, model_rdm_vec)` — Spearman correlation
- `partial_rsa_regression(neural_rdm_vec, model_rdm_vecs)` — multiple regression, return betas + semi-partial r
- `permutation_test_simple(neural_rdm, model_rdm, n_perms, item_structure)` — permute condition labels within items
- `permutation_test_partial(neural_rdm, model_rdm_dict, hypothesis_key, n_perms, item_structure)` — same for partial RSA
- `fdr_correct(p_values)` — BH-FDR correction

### `1_extract_activations.py` — Phase 1 (GPU)
- Load LLaMA-2-13B-Chat (float16, device_map="auto", local_files_only=True)
- For each of 336 sentences:
  - Tokenize (no chat formatting — these are bare sentences, not conversations)
  - Forward pass with `output_hidden_states=True`
  - Extract activations at **last token** and **mean across tokens** for all 41 layers
- Save: `activations/data/activations_last_token.npz` (336 x 41 x 5120, float16)
- Save: `activations/data/activations_mean_token.npz` (336 x 41 x 5120, float16)
- Save: `activations/data/stimuli_metadata.csv` (item_id, condition, category, sentence, n_tokens)
- Estimated time: ~5-10 min (336 short sentences, single forward pass each, no generation)
- **SLURM**: GPU job, `--gres=gpu:1 --mem=48G --time=1:00:00`

### `2_rsa_analysis.py` — Phase 2 (CPU)
- Load activations from phase 1
- For each layer (41 layers) x each token position mode (last, mean):

  **Analysis 1 — Simple RSA:**
  - Compute 336x336 neural RDM (correlation distance)
  - Build Model A RDM
  - Spearman correlation
  - Permutation test (10,000 iterations, shuffle condition labels within items)

  **Analysis 2 — Partial RSA (primary: Model A):**
  - Build all 7 model RDMs (A, B, C, D, F, G, H)
  - Multiple regression: neural ~ A + B + C + D + F + G + H
  - Extract beta_A, semi-partial r for A, and betas for all confounds
  - Permutation test for each beta

  **Analysis 2b — Partial RSA (secondary: Model E):**
  - Swap A for E in the regression
  - Same procedure

  **Analysis 3 — Category Structure RSA:**
  - Extract condition 1 only (56 sentences)
  - Compute 56x56 neural RDM
  - Build Model Cat RDM (7 categories x 8 items)
  - Spearman correlation + permutation test
  - Optional: run on each condition separately for comparison

- FDR correction across 41 layers for each analysis
- Save all results to `rsa/data/`:
  - `simple_rsa_results.csv` (layer, token_mode, rho, p, p_fdr)
  - `partial_rsa_primary_results.csv` (layer, token_mode, model, beta, semi_partial_r, p, p_fdr)
  - `partial_rsa_secondary_results.csv` (same with Model E)
  - `category_rsa_results.csv` (layer, token_mode, condition, rho, p, p_fdr)
  - `permutation_nulls.npz` (null distributions for key analyses)
- Estimated time: 10,000 permutations x 41 layers x multiple analyses — likely 30-60 min
- **SLURM**: CPU job, `--mem=32G --time=4:00:00` (conservative; RDMs are 336x336 = ~900K floats, regression is fast, but 10K permutations adds up)

### `2a_rsa_report_generator.py` — Report (login node)
- Load CSVs from `rsa/data/`
- Generate `rsa/rsa_report.html` with:
  - Layer profile plots for each analysis (simple RSA rho, partial RSA betas, category RSA rho)
  - Significance markers (FDR-corrected)
  - Summary table of peak layers and effect sizes
  - Comparison across token position modes (last vs mean)
  - Confound model betas across layers (to show what each confound captures)
- Standalone Python script that regenerates from saved CSVs

## Execution Order

1. **Write code** (config.py, stimuli.py, utils/rsa.py, 1_extract_activations.py, SLURM wrappers)
2. **Submit phase 1** via SLURM: `sbatch code/slurm/1_extract_activations.sh`
3. **After phase 1 completes**, submit phase 2: `sbatch code/slurm/2_rsa_analysis.sh`
4. **After phase 2 completes**, run report generator on login node: `python code/2a_rsa_report_generator.py`

## Key Design Decisions

- **No chat formatting**: These are bare sentences, not conversations. No system prompt, no [INST] tags. Just raw text -> tokenizer -> forward pass. This is different from exp_2/3/4 which use `llama_v2_prompt()`.
- **Both token positions**: Last token captures sentence-final representation; mean captures full-sentence representation. Report both, let the data decide.
- **Correlation distance for RDMs**: Standard in RSA literature (Kriegeskorte 2008). More robust to scale differences than cosine distance. Will also compute cosine as a robustness check.
- **Permutation within items**: Shuffling condition labels within items preserves item structure (shared object nouns) while breaking condition assignments. This is the correct null for this design.
- **Float16 storage**: 336 x 41 x 5120 x 2 bytes = ~140 MB per file. Manageable.
- **Single model**: LLaMA-2-13B-Chat only (matches exp_2/3). No base model variant needed since this is about representational structure, not RLHF effects.
