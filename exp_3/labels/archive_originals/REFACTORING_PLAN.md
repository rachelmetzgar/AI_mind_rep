# Exp 3 Pipeline Refactoring Plan

## Current State Analysis

After analyzing all 20 Python scripts in the exp_3/labels directory, here's what the pipeline does and where it needs improvement.

---

## Pipeline Structure (Current)

### **Main Pipeline (Sequential):**
```
1_elicit_concept_vectors.py (17K)  → Extract concept activations
2_train_concept_probes.py (14K)    → Train probes on activations
3_concept_intervention.py (28K)    → Generate steered conversations
4_behavior_analysis.py (25K)       → Linguistic feature analysis
5_cross_prediction.py (20K)        → Cross-domain prediction tests
```

### **Analysis Scripts (Parallel, run after Phase 1 or 2):**

**Phase 1 analyses:**
- `1b_alignment_analysis.py` (24K) — Alignment with Exp 2 probes (raw/residual/standalone)
- `1c_layer_profile_analysis.py` (39K) — Layer-wise alignment profiles
- `1d_elicit_sysprompt_vectors.py` (11K) — System prompt variant extraction
- `1e_sysprompt_alignment.py` (16K) — System prompt alignment analysis

**Phase 2 analyses:**
- `2b_summarize_concept_probes.py` (37K) — Summarize probe training results
- `2c_permutation_tests.py` (18K) — Permutation testing for alignment
- `2d_concept_probe_stats.py` (37K) — Statistical analysis (contrasts)
- `2e_concept_probe_figures.py` (37K) — Generate figures (contrasts)

**Phase 3 analyses (standalone mode):**
- `3a_standalone_stats.py` (33K) — Stats for standalone activations
- `3b_standalone_figures.py` (41K) — Figures for standalone activations
- `3_summarize_alignment.py` (7.4K) — **CONFUSING NAME** Cross-dimension summary

### **Reporting/Documentation:**
- `lexical_distinctiveness.py` (9.6K) — Lexical overlap investigation
- `build_lexical_overlap_report.py` (64K) — HTML report builder
- `build_lexical_overlap_pptx.py` (36K) — PowerPoint report builder
- `make_pub_figures.py` (26K) — Publication-ready figures

---

## Key Problems

### 1. **Naming Confusion** 🔴 HIGH PRIORITY
- **Problem**: `3_summarize_alignment.py` looks like it's part of the main pipeline (like 1, 2, 3, 4, 5), but it's actually an analysis script
- **Impact**: Hard to tell what order to run scripts in
- **Recommendation**: Rename to `analysis/summarize_alignment_cross_dimension.py`

### 2. **Script Proliferation** 🟡 MEDIUM PRIORITY
- **Problem**: 20 scripts in a flat directory, 7 of which start with "1", 5 with "2", 4 with "3"
- **Impact**: Overwhelming, hard to navigate
- **Recommendation**: Organize into subdirectories

### 3. **Code Duplication** 🟡 MEDIUM PRIORITY
- **Duplicated across 2d/2e and 3a/3b:**
  - Bootstrap confidence interval computation
  - Permutation testing logic
  - Figure generation templates (bar plots, heatmaps, strip plots)
  - Statistical test wrappers
- **Impact**: Bug fixes must be applied multiple times, inconsistency risk
- **Recommendation**: Extract shared utilities

### 4. **Contrast/Standalone Mode Split** 🟡 MEDIUM PRIORITY
- **Problem**: Nearly identical analysis pipelines for contrasts vs standalone, creating parallel script pairs (2d/3a, 2e/3b)
- **Impact**: 2× the code to maintain, harder to ensure consistency
- **Recommendation**: Merge into unified scripts with `--mode` flag

### 5. **Monolithic Analysis Scripts** 🟢 LOW PRIORITY
- **Problem**: Scripts like `1c_layer_profile_analysis.py` (39K), `2e_concept_probe_figures.py` (37K), `build_lexical_overlap_report.py` (64K) are huge
- **Impact**: Hard to read, test, and modify
- **Recommendation**: Split into modular components

### 6. **Unclear Dependencies** 🟡 MEDIUM PRIORITY
- **Problem**: No clear indication of what must run before what
- **Impact**: Users don't know what order to run scripts
- **Recommendation**: Add dependency diagram, pipeline runner script

---

## Recommended Refactoring

### **Phase 1: Quick Wins (1-2 hours)**

#### A. Reorganize into directories
```
exp_3/labels/
├── pipeline/                    # Main 5-phase pipeline
│   ├── 1_elicit_concepts.py
│   ├── 2_train_probes.py
│   ├── 3_generate_interventions.py
│   ├── 4_analyze_behavior.py
│   └── 5_cross_prediction.py
│
├── analysis/                    # Parallel analyses
│   ├── alignment/
│   │   ├── compute_alignment.py         # Merges 1b + 1e
│   │   ├── analyze_layer_profiles.py    # Was 1c
│   │   └── summarize_cross_dimension.py # Was 3_summarize_alignment.py
│   │
│   ├── probes/
│   │   ├── summarize_probe_training.py  # Was 2b
│   │   ├── permutation_tests.py         # Was 2c
│   │   ├── compute_stats.py             # Merges 2d + 3a
│   │   └── generate_figures.py          # Merges 2e + 3b
│   │
│   └── lexical/
│       ├── compute_distinctiveness.py   # Was lexical_distinctiveness.py
│       └── generate_reports.py          # Merges HTML + PPTX builders
│
├── reports/
│   └── make_pub_figures.py
│
├── utils/                       # Shared utilities (NEW)
│   ├── __init__.py
│   ├── stats.py                 # Bootstrap, permutation tests
│   ├── plotting.py              # Shared plot templates
│   └── io_helpers.py            # File loading/saving
│
├── src/                         # Already exists
│   ├── dataset.py
│   ├── probes.py
│   ├── intervention_utils.py
│   └── ...
│
└── run_pipeline.py              # NEW: Master runner script
```

#### B. Create `run_pipeline.py` master script
```python
#!/usr/bin/env python3
"""
Master pipeline runner for Experiment 3.

Usage:
    # Run full pipeline
    python run_pipeline.py --mode contrasts --dim_id 1

    # Run specific phase
    python run_pipeline.py --phase 2 --dim_id 1

    # Run analysis only
    python run_pipeline.py --analysis alignment --mode standalone
"""
```

#### C. Add `PIPELINE.md` documentation
- Clear dependency diagram
- What each script does in 1-2 sentences
- When to run each script
- Expected inputs/outputs

---

### **Phase 2: Consolidation (4-6 hours)**

#### D. Merge duplicate analysis pairs

**Merge 2d + 3a → `analysis/probes/compute_stats.py`:**
```python
python analysis/probes/compute_stats.py \
    --mode contrasts \  # or standalone
    --dim_id 1
```
- Extract shared bootstrap/permutation code to `utils/stats.py`
- Single codebase, cleaner logic

**Merge 2e + 3b → `analysis/probes/generate_figures.py`:**
```python
python analysis/probes/generate_figures.py \
    --mode contrasts \  # or standalone
    --dim_id 1
```
- Extract plot templates to `utils/plotting.py`
- Reduce from 78K → ~30K lines

#### E. Consolidate alignment analyses

**Merge 1b + 1e → `analysis/alignment/compute_alignment.py`:**
```python
python analysis/alignment/compute_alignment.py \
    --analysis raw \        # or residual, standalone
    --dim_id 1
```
- Both do similar cosine similarity + bootstrap
- Sysprompt is just a special case (dim 18-22)

#### F. Merge report builders

**Merge HTML + PPTX → `analysis/lexical/generate_reports.py`:**
```python
python analysis/lexical/generate_reports.py \
    --format html \  # or pptx, or both
```
- Share figure generation code
- Output both formats from same analysis

---

### **Phase 3: Refactor Internals (Optional, 8-12 hours)**

#### G. Extract shared utilities

**`utils/stats.py`:**
- `bootstrap_ci(data, stat_fn, n_iter=1000, ci=95)`
- `permutation_test(group1, group2, stat_fn, n_perm=10000)`
- `split_half_reliability(data, n_splits=100)`

**`utils/plotting.py`:**
- `plot_bar_with_ci(data, labels, title, ...)`
- `plot_heatmap(matrix, row_labels, col_labels, ...)`
- `plot_strip_with_overlay(data, conditions, ...)`
- `setup_figure_style()` — consistent fonts, colors

**`utils/io_helpers.py`:**
- `load_concept_activations(dim_name, mode)`
- `load_probe_weights(dim_name, layer_idx)`
- `save_results_json(data, output_path)`

#### H. Split monolithic scripts

**`build_lexical_overlap_report.py` (64K) → 3 modules:**
- `lexical/compute_metrics.py` — Analysis logic
- `lexical/generate_html.py` — HTML templating
- `lexical/main.py` — Orchestration

---

## Migration Plan

### Step 1: No-Risk Reorganization (Do First)
1. Create new directory structure
2. **Copy** (don't move) scripts to new locations with new names
3. Update imports in copied scripts
4. Test copied scripts work
5. Leave originals in place until confident

### Step 2: Add Documentation
1. Create `PIPELINE.md` with dependency diagram
2. Add docstrings to all scripts (if missing)
3. Create `run_pipeline.py` wrapper

### Step 3: Consolidation (Requires Testing)
1. Merge 2d + 3a (most redundant, biggest win)
2. Test on a few dimensions
3. If successful, merge 2e + 3b
4. Then merge alignment scripts

### Step 4: Extract Utilities (Optional)
1. Create `utils/` package
2. Extract bootstrap code
3. Extract plotting templates
4. Update all scripts to use shared utilities

---

## Testing Strategy

For each refactored script:
1. **Smoke test**: Run on dim_id=0 (baseline), check outputs exist
2. **Comparison test**: Compare outputs to original script (should be identical or very close)
3. **Edge case**: Run on dim_id=15 (shapes, control dimension)
4. **Full test**: Run on dim_id=1 (phenomenology, representative dimension)

---

## Priority Recommendations

### **MUST DO (High Impact, Low Risk):**
1. ✅ Rename `3_summarize_alignment.py` → `analysis/summarize_alignment_cross_dimension.py`
2. ✅ Create `PIPELINE.md` documentation with dependency diagram
3. ✅ Create directory structure (pipeline/, analysis/, utils/)
4. ✅ Move/copy scripts to new locations

### **SHOULD DO (High Impact, Medium Risk):**
5. ✅ Merge 2d + 3a → `analysis/probes/compute_stats.py` (eliminates ~35K duplicate code)
6. ✅ Merge 2e + 3b → `analysis/probes/generate_figures.py` (eliminates ~40K duplicate code)
7. ✅ Create `run_pipeline.py` master runner

### **NICE TO HAVE (Medium Impact, Higher Risk):**
8. Extract `utils/stats.py` and `utils/plotting.py`
9. Merge HTML + PPTX report builders
10. Split `build_lexical_overlap_report.py` into modules

---

## Estimated Time Investment

| Task | Time | Impact | Risk |
|------|------|--------|------|
| Reorganize directories | 1 hr | High | Low |
| Create PIPELINE.md | 1 hr | High | None |
| Create run_pipeline.py | 2 hrs | Medium | Low |
| Merge 2d+3a (stats) | 3 hrs | High | Medium |
| Merge 2e+3b (figures) | 3 hrs | High | Medium |
| Extract utils/stats.py | 4 hrs | Medium | Medium |
| Extract utils/plotting.py | 4 hrs | Medium | Medium |
| **TOTAL (Essential)** | **7 hrs** | — | — |
| **TOTAL (Full refactor)** | **18 hrs** | — | — |

---

## Questions to Resolve

1. **Naming**: For "a human"/"an AI" labels version, should we rename scripts to reflect this? (e.g., `1_elicit_concepts_labels.py` vs `1_elicit_concepts_names.py`)

2. **Config**: Should we use a central config file for paths (Exp 2b probe locations, model paths) instead of hardcoding in each script?

3. **SLURM**: Should SLURM scripts move to `slurm/` in the root, or stay in each subdirectory?

4. **Standalone mode**: Is this still actively used, or should it be deprioritized in refactoring?

5. **Lexical analysis**: Is this a one-time investigation or ongoing? If one-time, maybe move to `reports/lexical_investigation/` to reduce clutter.

---

## Next Steps

Let me know which phase you want to tackle:
- **Phase 1 (Quick Wins)**: Reorganize + document (2 hrs)
- **Phase 2 (Consolidation)**: Merge duplicate scripts (6 hrs)
- **Phase 3 (Deep Refactor)**: Extract utilities, split monoliths (12 hrs)

I can execute any of these with your approval. What's your priority?
