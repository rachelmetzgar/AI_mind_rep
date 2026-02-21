# Script Consolidation Plan

## Current State: 19 Scripts

### Pipeline Scripts (5) - KEEP SEPARATE ✅
These are sequential phases - should NOT be merged:

1. **`pipeline/1_elicit_concept_vectors.py`** (17K)
   - Extracts activation vectors from LLaMA for all concept prompts
   - Creates: `data/concept_activations/`

2. **`pipeline/2_train_concept_probes.py`** (14K)
   - Trains binary classifiers to predict human vs AI from activations
   - Creates: `data/concept_probes/`

3. **`pipeline/3_concept_intervention.py`** (28K)
   - Steers model generation by adding concept vectors to activations
   - Creates: `results/interventions/`

4. **`pipeline/4_behavior_analysis.py`** (25K)
   - Analyzes text outputs from interventions (lexical markers, hedging, etc.)
   - Creates: `results/behavioral/`

5. **`pipeline/5_cross_prediction.py`** (20K)
   - Tests if concept probes predict Exp 2 conversational data
   - Creates: `results/cross_prediction/`

**Total: 5 scripts, 104K code**
**Decision: KEEP AS-IS** - these are distinct pipeline phases

---

## Analysis Scripts (14) - OPPORTUNITIES FOR CONSOLIDATION ⚠️

### Alignment Analysis (5 scripts)

6. **`analysis/alignment/1b_alignment_analysis.py`** (24K)
   - Computes cosine similarity between concept vectors and Exp 2 probe weights
   - Three modes: raw, residual (entity-subtracted), standalone
   - Creates: `results/alignment/`

7. **`analysis/alignment/1c_layer_profile_analysis.py`** (39K)
   - Analyzes PER-LAYER alignment profiles (where does alignment emerge?)
   - Depends on: 1b's outputs
   - Creates: `results/alignment/layer_profiles/`

8. **`analysis/alignment/1d_elicit_sysprompt_vectors.py`** (11K)
   - Extracts activations for system prompt variations (bare names, labeled names)
   - Creates: `data/concept_activations/` (dims 18, 20-23)

9. **`analysis/alignment/1e_sysprompt_alignment.py`** (16K)
   - Tests alignment between system prompt vectors and concept vectors
   - Creates: `results/alignment/sysprompt/`

10. **`analysis/alignment/summarize_cross_dimension.py`** (7K)
    - Aggregates alignment results across all dimensions into summary tables
    - Creates: `results/alignment/summary/`

**Potential consolidation:**
- Scripts 1d and 1e could merge: `elicit_and_align_sysprompt.py --step elicit|align|both`
- **BUT** these are investigative analyses, not core pipeline
- **Decision: KEEP FOR NOW** - minor savings, increases complexity

---

### Probe Analysis (6 scripts) - **HIGH PRIORITY CONSOLIDATION** 🔥

#### GROUP A: Contrast Analysis

11. **`analysis/probes/2b_summarize_concept_probes.py`** (37K)
    - Loads probe accuracy and alignment results
    - Computes summary stats (bootstrap, t-tests)
    - Creates CSV tables and bar charts
    - **Function**: Quick summary of all dimensions

12. **`analysis/probes/2c_permutation_tests.py`** (18K)
    - Permutation tests for alignment significance
    - Faster than t-tests, accounts for layer correlation
    - **Function**: Statistical significance testing

13. **`analysis/probes/2d_concept_probe_stats.py`** (37K) 🔥
    - **COMPREHENSIVE stats**: permutation tests, bootstrap CIs, pairwise comparisons, FDR correction
    - Per-dimension, per-layer, and category-level analyses
    - Creates: `results/probes/alignment/summaries/` (JSON + CSV)
    - **Function**: COMPLETE statistical analysis for CONTRAST mode

14. **`analysis/probes/2e_concept_probe_figures.py`** (37K) 🔥
    - **Publication-quality figures** for contrast analysis
    - Bar charts, heatmaps, layer profiles, category comparisons
    - Creates: `results/probes/alignment/figures/` (50+ PNG files)
    - **Function**: COMPLETE visualization for CONTRAST mode

#### GROUP B: Standalone Analysis

15. **`analysis/probes/3a_standalone_stats.py`** (33K) 🔥
    - **COMPREHENSIVE stats** for standalone mode (no human/AI labels)
    - Bootstrap tests, pairwise comparisons, FDR correction
    - Creates: `results/probes/standalone_alignment/summaries/`
    - **Function**: COMPLETE statistical analysis for STANDALONE mode

16. **`analysis/probes/3b_standalone_figures.py`** (41K) 🔥
    - **Publication-quality figures** for standalone analysis
    - Bar charts, entity comparisons, sysprompt variants
    - Creates: `results/probes/standalone_alignment/figures/`
    - **Function**: COMPLETE visualization for STANDALONE mode

**Overlap Analysis:**
- **2d vs 3a**: ~65% code duplication (bootstrap, FDR, category comparisons)
- **2e vs 3b**: ~70% code duplication (plotting functions, style, layouts)
- **2b vs 2c**: Partially redundant (2d supersedes both)

**Consolidation Opportunity:**
```
BEFORE (6 scripts):
- 2b_summarize_concept_probes.py     (37K) → REDUNDANT with 2d
- 2c_permutation_tests.py            (18K) → REDUNDANT with 2d
- 2d_concept_probe_stats.py          (37K) → KEEP (comprehensive)
- 2e_concept_probe_figures.py        (37K) → KEEP (comprehensive)
- 3a_standalone_stats.py             (33K) → MERGE into 2d
- 3b_standalone_figures.py           (41K) → MERGE into 2e

AFTER (2 scripts):
- compute_alignment_stats.py         (50K) → contrast + standalone stats
- generate_alignment_figures.py      (55K) → contrast + standalone figures
```

**Lines saved: ~150K → ~105K (30% reduction)**
**Scripts saved: 6 → 2 (67% reduction)**

---

### Lexical Analysis (3 scripts) - **MEDIUM PRIORITY** ⚠️

17. **`analysis/lexical/lexical_distinctiveness.py`** (10K)
    - Computes Jaccard similarity between human/AI prompt word sets
    - Tests correlation with alignment
    - Creates: `results/lexical/lexical_distinctiveness.csv`

18. **`analysis/lexical/build_lexical_overlap_report.py`** (64K) 🔥
    - Generates interactive HTML report with embedded figures
    - Scatter plots, correlations, dimension comparisons
    - Creates: `results/lexical/LEXICAL_OVERLAP_REPORT.html`

19. **`analysis/lexical/build_lexical_overlap_pptx.py`** (36K) 🔥
    - Generates PowerPoint presentation with same content
    - Creates: `results/lexical/LEXICAL_OVERLAP_REPORT.pptx`

**Overlap:**
- Scripts 18 and 19 share ~45% code (data loading, statistics, figure generation)
- Different output formats (HTML vs PPTX)

**Consolidation Opportunity:**
```
BEFORE (3 scripts):
- lexical_distinctiveness.py          (10K) → KEEP (data generation)
- build_lexical_overlap_report.py     (64K) → MERGE
- build_lexical_overlap_pptx.py       (36K) → MERGE

AFTER (2 scripts):
- lexical_distinctiveness.py          (10K) → unchanged
- generate_lexical_report.py          (70K) → HTML + PPTX + both
```

**Lines saved: ~110K → ~80K (27% reduction)**
**Scripts saved: 3 → 2 (33% reduction)**

---

## Consolidation Summary

### Option 1: Conservative (Keep Investigative Scripts)

**Merge only probe analysis scripts:**
- 6 probe scripts → 2 unified scripts
- Saves: 4 scripts, ~45K lines of code
- **Final count: 15 scripts** (down from 19)

### Option 2: Moderate (Recommended) ✅

**Merge probe + lexical scripts:**
- 6 probe scripts → 2 unified
- 3 lexical scripts → 2 unified
- Saves: 5 scripts, ~65K lines of code
- **Final count: 14 scripts** (down from 19)

### Option 3: Aggressive (Maximum Consolidation)

**Merge everything possible:**
- 6 probe scripts → 2 unified
- 3 lexical scripts → 2 unified
- 2 alignment scripts merged (1d + 1e)
- Saves: 6 scripts, ~70K lines of code
- **Final count: 13 scripts** (down from 19)

---

## Recommendation: Option 2 (Moderate) ✅

**Why:**
1. **High-impact merges** (probe + lexical scripts are heavily redundant)
2. **Clear structure** (stats scripts generate numbers, figure scripts make plots)
3. **Easy to use** (single --mode flag switches between contrast/standalone)
4. **Preserves flexibility** (alignment scripts stay separate for investigation)

**New Structure:**
```
pipeline/ (5 scripts - unchanged)
├── 1_elicit_concept_vectors.py
├── 2_train_concept_probes.py
├── 3_concept_intervention.py
├── 4_behavior_analysis.py
└── 5_cross_prediction.py

analysis/
├── alignment/ (5 scripts - unchanged)
│   ├── 1b_alignment_analysis.py
│   ├── 1c_layer_profile_analysis.py
│   ├── 1d_elicit_sysprompt_vectors.py
│   ├── 1e_sysprompt_alignment.py
│   └── summarize_cross_dimension.py
│
├── probes/ (2 scripts - MERGED FROM 6)
│   ├── compute_alignment_stats.py        ← NEW (2d + 3a merged)
│   └── generate_alignment_figures.py     ← NEW (2e + 3b merged)
│
└── lexical/ (2 scripts - MERGED FROM 3)
    ├── lexical_distinctiveness.py         ← unchanged
    └── generate_lexical_report.py         ← NEW (HTML + PPTX merged)
```

**Total: 14 scripts (down from 19, -26%)**

---

## Implementation Plan

### Phase 1: Merge Probe Stats Scripts (2d + 3a)

**New script: `analysis/probes/compute_alignment_stats.py`**

```bash
# Usage (replaces both 2d and 3a):
python analysis/probes/compute_alignment_stats.py --mode contrast
python analysis/probes/compute_alignment_stats.py --mode standalone
python analysis/probes/compute_alignment_stats.py --mode both
```

**Changes needed:**
1. Add `--mode` argument
2. Load data from different paths based on mode
3. Adjust test statistic computation (contrast uses permutation, standalone uses bootstrap)
4. Keep all existing outputs (JSON, CSV, methods.md)

**Estimated effort:** 2-3 hours
**Risk:** Low (output formats identical, just different input data)

### Phase 2: Merge Probe Figure Scripts (2e + 3b)

**New script: `analysis/probes/generate_alignment_figures.py`**

```bash
# Usage (replaces both 2e and 3b):
python analysis/probes/generate_alignment_figures.py --mode contrast
python analysis/probes/generate_alignment_figures.py --mode standalone
python analysis/probes/generate_alignment_figures.py --mode both
```

**Changes needed:**
1. Add `--mode` argument
2. Load stats from different JSON files based on mode
3. Adjust axis labels (contrast: "Human-AI alignment", standalone: "Projection onto probe")
4. Keep dimension/category plots, add mode-specific plots (entity, sysprompt)

**Estimated effort:** 2-3 hours
**Risk:** Low (plotting functions reusable, just different data/labels)

### Phase 3: Merge Lexical Report Scripts

**New script: `analysis/lexical/generate_lexical_report.py`**

```bash
# Usage (replaces both HTML and PPTX builders):
python analysis/lexical/generate_lexical_report.py --format html
python analysis/lexical/generate_lexical_report.py --format pptx
python analysis/lexical/generate_lexical_report.py --format both
```

**Changes needed:**
1. Add `--format` argument
2. Extract shared data loading and figure generation into functions
3. Keep separate rendering functions for HTML vs PPTX
4. Keep both output formats available

**Estimated effort:** 3-4 hours
**Risk:** Medium (two different output formats require careful refactoring)

### Phase 4: Clean Up

1. **Move obsolete scripts to archive:**
   - 2b_summarize_concept_probes.py
   - 2c_permutation_tests.py
   - 2d_concept_probe_stats.py → replaced by compute_alignment_stats.py
   - 3a_standalone_stats.py → replaced by compute_alignment_stats.py
   - 2e_concept_probe_figures.py → replaced by generate_alignment_figures.py
   - 3b_standalone_figures.py → replaced by generate_alignment_figures.py
   - build_lexical_overlap_report.py → replaced by generate_lexical_report.py
   - build_lexical_overlap_pptx.py → replaced by generate_lexical_report.py

2. **Update documentation:**
   - PIPELINE.md: Update script names and usage examples
   - MIGRATION_COMPLETE.md: Add consolidation section

3. **Test consolidated scripts:**
   - Verify outputs match original scripts
   - Test all modes (contrast, standalone, both)
   - Check all output formats

**Total effort:** 7-10 hours
**Total reduction:** 5 scripts, ~65K duplicate code

---

## Benefits

### Immediate
- ✅ **26% fewer scripts** (19 → 14)
- ✅ **Clearer purpose** ("compute stats" vs "generate figures")
- ✅ **Single usage pattern** (--mode flag instead of different scripts)
- ✅ **Less duplication** (~65K lines of duplicate code eliminated)

### Long-term
- ✅ **Easier maintenance** (fix a bug once, not twice)
- ✅ **Faster development** (add a feature once, not twice)
- ✅ **Better testing** (fewer scripts = more thorough tests possible)
- ✅ **Lower cognitive load** (fewer files to remember)

---

## Questions to Consider

1. **Do you want me to implement this consolidation now?**
   - Yes → I'll start with Phase 1 (merge stats scripts)
   - No → Keep current structure, use CONSOLIDATION_PLAN.md as reference

2. **Should I keep the old scripts as backup during consolidation?**
   - Yes → Move to `archive_originals/` after testing
   - No → Delete immediately after verifying new scripts work

3. **Any scripts you DEFINITELY want to keep separate?**
   - Example: Maybe you frequently run HTML reports but never PPTX
   - We can adjust the plan based on your actual usage patterns

---

Last updated: Feb 19, 2026
