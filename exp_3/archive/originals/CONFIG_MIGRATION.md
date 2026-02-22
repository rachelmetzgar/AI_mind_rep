# Migrating Scripts to Use config.py

## Overview

Scripts in `pipeline/` and `analysis/` have been **copied** from the root directory. To use the centralized `config.py`, each script needs minimal updates.

**Status**:
- ✅ `config.py` created with all paths and hyperparameters
- ✅ `pipeline/1_elicit_concept_vectors.py` updated (DONE)
- ✅ `pipeline/2_train_concept_probes.py` updated (DONE)
- ⏳ Remaining 15+ scripts need updating (follow pattern below)

---

## Migration Pattern

### Step 1: Update imports

**OLD** (when script was in root):
```python
sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import llama_v2_prompt
```

**NEW** (when script is in pipeline/ or analysis/):
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import llama_v2_prompt
from config import config
```

### Step 2: Replace hardcoded config values

| Old Hardcoded Value | New config.py Reference |
|---------------------|-------------------------|
| `MODEL_NAME = "/jukebox/..."` | `MODEL_NAME = config.MODEL_NAME` |
| `INPUT_DIM = 5120` | `INPUT_DIM = config.INPUT_DIM` |
| `DEVICE = "cuda" if ...` | `DEVICE = config.get_device()` |
| `torch.manual_seed(12345)` | `torch.manual_seed(config.ANALYSIS.seed)` |
| `EPOCHS = 50` | `EPOCHS = config.TRAINING.epochs` |
| `BATCH_SIZE_TRAIN = 16` | `BATCH_SIZE_TRAIN = config.TRAINING.batch_size_train` |
| `N_BOOTSTRAP = 1000` | `N_BOOTSTRAP = config.ANALYSIS.n_bootstrap` |
| `N_PERMUTATIONS = 10000` | `N_PERMUTATIONS = config.ANALYSIS.n_permutations` |

### Step 3: Replace hardcoded paths

#### Input Paths (Concepts, Exp 2, Exp 1)

| Old Hardcoded Path | New config.py Reference |
|--------------------|-------------------------|
| `"concepts"` | `str(config.PATHS.concepts_root)` |
| `"concepts/contrasts"` | `str(config.PATHS.concepts_contrasts)` |
| `"concepts/standalone"` | `str(config.PATHS.concepts_standalone)` |
| `"/jukebox/.../exp_2b/.../probe_checkpoints"` | `str(config.PATHS.exp2_probes)` |
| `"/jukebox/.../exp_2b/.../control_probe"` | `str(config.PATHS.exp2_control_probe)` |
| `"/jukebox/.../exp_2b/.../reading_probe"` | `str(config.PATHS.exp2_reading_probe)` |
| `"/jukebox/.../exp_1/.../prompts"` | `str(config.PATHS.exp1_prompts)` |

#### Data Paths (Intermediate data - stays in data/)

| Old Hardcoded Path | New config.py Reference |
|--------------------|-------------------------|
| `"data/concept_activations"` | `str(config.PATHS.concept_activations)` |
| `"data/concept_activations/contrasts"` | `str(config.PATHS.concept_activations_contrasts)` |
| `"data/concept_activations/standalone"` | `str(config.PATHS.concept_activations_standalone)` |
| `"data/concept_probes"` | `str(config.PATHS.concept_probes)` |

#### Output Paths (Final results - goes to results/)

| Old Hardcoded Path | New config.py Reference |
|--------------------|-------------------------|
| `"data/alignment"` → | `str(config.RESULTS.alignment)` |
| `"data/alignment/contrasts"` → | `str(config.RESULTS.alignment_contrasts)` |
| `"data/alignment/contrasts/raw"` → | `str(config.RESULTS.alignment_contrasts_raw)` |
| `"data/alignment/contrasts/residual"` → | `str(config.RESULTS.alignment_contrasts_residual)` |
| `"data/alignment/standalone"` → | `str(config.RESULTS.alignment_standalone)` |
| `"data/intervention_results"` → | `str(config.RESULTS.interventions)` |
| `"data/intervention_results/V1"` → | `str(config.RESULTS.interventions_v1)` |
| `"data/intervention_results/V2"` → | `str(config.RESULTS.interventions_v2)` |
| `"results/behavioral"` → | `str(config.RESULTS.behavioral)` |
| `"results/lexical"` → | `str(config.RESULTS.lexical)` |

**Key principle**: Intermediate data (activations, trained probes) stays in `data/`. Final results (alignment stats, figures, reports) go to `results/`.

---

## Example: Before and After

### BEFORE (original script in root)

```python
#!/usr/bin/env python3
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.probes import LinearProbeClassification

# Config
MODEL_NAME = "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/..."
INPUT_DIM = 5120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 16
OUTPUT_DIR = "data/alignment/contrasts/raw"
EXP2B_PROBES = "/jukebox/.../exp_2b/llama_exp_2b-13B-chat/data/probe_checkpoints"
torch.manual_seed(12345)

# ... rest of script ...
```

### AFTER (updated script in pipeline/)

```python
#!/usr/bin/env python3
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.probes import LinearProbeClassification
from config import config

# Config
MODEL_NAME = config.MODEL_NAME
INPUT_DIM = config.INPUT_DIM
DEVICE = config.get_device()
EPOCHS = config.TRAINING.epochs
BATCH_SIZE = config.TRAINING.batch_size_train
OUTPUT_DIR = str(config.RESULTS.alignment_contrasts_raw)
EXP2_PROBES = str(config.PATHS.exp2_probes)
torch.manual_seed(config.ANALYSIS.seed)

# ... rest of script ...
```

---

## Scripts Needing Migration

### Pipeline (High Priority)

- ✅ `pipeline/1_elicit_concept_vectors.py` (DONE)
- ✅ `pipeline/2_train_concept_probes.py` (DONE)
- ⏳ `pipeline/3_concept_intervention.py`
- ⏳ `pipeline/4_behavior_analysis.py`
- ⏳ `pipeline/5_cross_prediction.py`

### Analysis - Alignment (Medium Priority)

- ⏳ `analysis/alignment/1b_alignment_analysis.py`
- ⏳ `analysis/alignment/1c_layer_profile_analysis.py`
- ⏳ `analysis/alignment/1d_elicit_sysprompt_vectors.py`
- ⏳ `analysis/alignment/1e_sysprompt_alignment.py`
- ⏳ `analysis/alignment/summarize_cross_dimension.py`

### Analysis - Probes (Medium Priority)

- ⏳ `analysis/probes/2b_summarize_concept_probes.py`
- ⏳ `analysis/probes/2c_permutation_tests.py`
- ⏳ `analysis/probes/2d_concept_probe_stats.py`
- ⏳ `analysis/probes/2e_concept_probe_figures.py`
- ⏳ `analysis/probes/3a_standalone_stats.py`
- ⏳ `analysis/probes/3b_standalone_figures.py`

### Analysis - Lexical (Low Priority)

- ⏳ `analysis/lexical/lexical_distinctiveness.py`
- ⏳ `analysis/lexical/build_lexical_overlap_report.py`
- ⏳ `analysis/lexical/build_lexical_overlap_pptx.py`

### Results - Figures (Low Priority)

- ⏳ `results/figures/make_pub_figures.py`

---

## Testing After Migration

After updating a script, test it:

```bash
# Test config import
cd /mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_3/labels
python3 -c "from config import config; print('✅ Config loads')"

# Test script import (no execution)
python3 -c "import pipeline.1_elicit_concept_vectors as m; print('✅ Script imports')"

# Run script (if applicable)
python pipeline/1_elicit_concept_vectors.py --mode contrasts --dim_id 0 --help
```

---

## Common Issues

### Issue 1: ImportError: No module named 'config'

**Cause**: `sys.path` not updated correctly
**Fix**: Ensure you have:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import config
```

### Issue 2: ImportError: No module named 'src'

**Cause**: `sys.path` points to wrong directory
**Fix**: For scripts in `pipeline/` or `analysis/*/`, add parent:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
```

For scripts in `analysis/probes/` (2 levels deep), add grandparent:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
```

### Issue 3: FileNotFoundError: Path not found

**Cause**: Relative paths may not work from new locations
**Fix**: Use absolute paths from `config`:
```python
# DON'T: concepts_dir = "concepts/contrasts"
# DO:    concepts_dir = str(config.PATHS.concepts_contrasts)
```

### Issue 4: Attribute Error: 'Path' object has no attribute '...'

**Cause**: Trying to use Path methods on strings
**Fix**: Convert Path to string when needed:
```python
OUTPUT_DIR = str(config.RESULTS.alignment)
```

---

## Benefits After Migration

✅ **Single source of truth** - Update paths once in `config.py`, not 20 scripts
✅ **Easier transitions** - Switch to `results/` output by changing config
✅ **Consistent hyperparameters** - All scripts use same epochs, batch sizes, seeds
✅ **Better portability** - Easy to adapt for different machines/clusters
✅ **Clearer intent** - `config.RESULTS.alignment` vs `"data/alignment"` shows where it goes

---

## Quick Reference Card

```python
# ============ IMPORTS ============
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import config

# ============ MODEL ============
MODEL_NAME = config.MODEL_NAME
INPUT_DIM = config.INPUT_DIM
DEVICE = config.get_device()
torch.manual_seed(config.ANALYSIS.seed)

# ============ TRAINING ============
EPOCHS = config.TRAINING.epochs
BATCH_SIZE_TRAIN = config.TRAINING.batch_size_train
BATCH_SIZE_TEST = config.TRAINING.batch_size_test
LOGISTIC = config.TRAINING.logistic
MIN_PROBE_ACC = config.TRAINING.min_probe_accuracy

# ============ ANALYSIS ============
N_BOOTSTRAP = config.ANALYSIS.n_bootstrap
N_PERMUTATIONS = config.ANALYSIS.n_permutations
N_SPLIT_HALF = config.ANALYSIS.n_split_half
CI_ALPHA = config.ANALYSIS.ci_alpha

# ============ GENERATION ============
# V1
MAX_TOKENS_V1 = config.GEN_V1.max_new_tokens
TEMPERATURE_V1 = config.GEN_V1.temperature
TOP_P_V1 = config.GEN_V1.top_p

# V2
MAX_TOKENS_V2 = config.GEN_V2.max_new_tokens
TEMPERATURE_V2 = config.GEN_V2.temperature
PAIRS_TOTAL = config.GEN_V2.pairs_total

# ============ INPUT PATHS ============
CONCEPTS_ROOT = str(config.PATHS.concepts_root)
CONCEPTS_CONTRASTS = str(config.PATHS.concepts_contrasts)
CONCEPTS_STANDALONE = str(config.PATHS.concepts_standalone)
EXP2_PROBES = str(config.PATHS.exp2_probes)
EXP2_CONTROL = str(config.PATHS.exp2_control_probe)
EXP2_READING = str(config.PATHS.exp2_reading_probe)
EXP1_PROMPTS = str(config.PATHS.exp1_prompts)

# ============ DATA PATHS (intermediate) ============
CONCEPT_ACTIVATIONS = str(config.PATHS.concept_activations)
CONCEPT_ACTIVATIONS_CONTRASTS = str(config.PATHS.concept_activations_contrasts)
CONCEPT_ACTIVATIONS_STANDALONE = str(config.PATHS.concept_activations_standalone)
CONCEPT_PROBES = str(config.PATHS.concept_probes)

# ============ RESULTS PATHS (final outputs) ============
RESULTS_ROOT = str(config.RESULTS.root)
ALIGNMENT = str(config.RESULTS.alignment)
ALIGNMENT_CONTRASTS_RAW = str(config.RESULTS.alignment_contrasts_raw)
ALIGNMENT_CONTRASTS_RESIDUAL = str(config.RESULTS.alignment_contrasts_residual)
ALIGNMENT_STANDALONE = str(config.RESULTS.alignment_standalone)
INTERVENTIONS_V1 = str(config.RESULTS.interventions_v1)
INTERVENTIONS_V2 = str(config.RESULTS.interventions_v2)
BEHAVIORAL = str(config.RESULTS.behavioral)
LEXICAL = str(config.RESULTS.lexical)
FIGURES = str(config.RESULTS.figures)
```

---

## Next Steps

1. Copy this pattern to each remaining script (15 scripts)
2. Test each script after migration
3. Once all scripts updated, can delete originals in root
4. Update SLURM scripts to call `pipeline/` and `analysis/` versions

Estimated time: ~15-30 minutes per script = ~4-8 hours total
