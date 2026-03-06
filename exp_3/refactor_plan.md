# Exp 3 Refactoring Plan

## Current Structure (non-archive)

```
exp_3/
  code/
    config.py
    analysis/
      alignment/       # 2a-2h (alignment analysis, 10 scripts)
      lexical/         # lexical distinctiveness (2 scripts)
      probes/          # 4a-4b (probe alignment stats, 2 scripts)
    pipeline/          # 1, 3, 4, 4b, 5, 6, 7 (main pipeline, 7 scripts)
    slurm/             # SLURM wrappers (16 scripts)
    src/               # Shared modules: dataset, probes, losses, etc. (7 files)
    utils/             # EMPTY
  concepts/            # Concept prompt definitions (top-level, not under code/)
    contrasts/         # 18 numbered .py files (0-17)
    standalone/        # 20 numbered .py files (1-19)
  data/
    causality_test_questions/  # human_ai.txt (static input)
    concept_activations/       # (appears empty — .npz files gitignored or deleted)
  results/
    alignment/
      versions/{version}/turn_{N}/{contrasts|standalone}/   # per-version alignment .npz
      comparisons/
        code/           # 4 GENERATOR SCRIPTS LIVING IN RESULTS (should be in code/)
        turn_{1-5}/{raw|residual|standalone}/               # comparison reports + figures
      concept_overlap/{contrasts|standalone}/                # overlap analysis
    versions/                                                # concept steering results
      balanced_gpt/concept_steering/v1/{dim}/{strategy}/    # steering output
  logs/
    alignment/
    concept_steering_v1/
  write_up/            # exp3_methods.html
  .ipynb_checkpoints/  # STALE
  LICENSE
  plan.md              # Phase 8 plan (future work)
  README.md
```

## Issues Found

1. **code/ not flattened**: `analysis/` and `pipeline/` subdirs (same issue exp_2 had)
2. **`code/src/` should be `code/utils/`**: utils/ exists but is empty, src/ has the modules
3. **`concepts/` is top-level**: Should be under `code/` (tracked input definitions)
4. **`data/` top-level**: `causality_test_questions/` is static input (move to code/); `concept_activations/` is pipeline output (move to results/)
5. **Results not model-scoped**: No `results/llama2_13b_chat/` level
6. **Two separate `versions/` trees**: `results/alignment/versions/` AND `results/versions/` — confusing
7. **Generator scripts in results/**: `results/alignment/comparisons/code/` has 4 Python scripts
8. **`write_up` → `writeup`**: Naming consistency
9. **Numbering collision**: `4a/4b` exists in both `analysis/probes/` (probe stats) and `pipeline/` (steering). Need to resolve when flattening.
10. **Exp 2 paths in config are STALE**: Points to `exp_2/data/{version}/probe_checkpoints/` but exp_2 refactored to `exp_2/results/llama2_13b_chat/{version}/probe_training/data/`
11. **Cleanup needed**: TODELETE dir, .ipynb_checkpoints at multiple levels
12. **`plan.md`**: Future work notes — archive or move to writeup

## Pipeline Map (current script → what it does)

### Phase 1: Concept Elicitation (no dependencies)
| Script | Purpose |
|--------|---------|
| `pipeline/1_elicit_concept_vectors.py` | Extract concept activation vectors from LLaMA |

### Phase 2: Alignment Analysis (needs Phase 1 + exp_2 probes)
| Script | Purpose |
|--------|---------|
| `analysis/alignment/2a_alignment_analysis.py` | Cosine alignment: concept vectors vs exp_2 probes |
| `analysis/alignment/2b_layer_profile_analysis.py` | Layer-resolved alignment profiles |
| `analysis/alignment/2c_elicit_sysprompt_vectors.py` | Extract system prompt concept vectors |
| `analysis/alignment/2d_sysprompt_alignment.py` | System prompt alignment analysis |
| `analysis/alignment/2e_summarize_cross_dimension.py` | Cross-dimension summary (REPORT GENERATOR) |
| `analysis/alignment/2f_concept_overlap.py` | Concept vector overlap (contrasts) |
| `analysis/alignment/2f_concept_overlap_report.py` | Overlap report (REPORT GENERATOR) |
| `analysis/alignment/2g_concept_overlap_standalone.py` | Concept vector overlap (standalone) |
| `analysis/alignment/2g_concept_overlap_standalone_report.py` | Overlap report (REPORT GENERATOR) |
| `analysis/alignment/2h_concept_aligned_layers.py` | Identify concept-aligned layers |
| `results/alignment/comparisons/code/generate_comparison_reports.py` | Cross-version comparison (REPORT GENERATOR, misplaced) |
| `results/alignment/comparisons/code/generate_raw_comparison.py` | Raw comparison (REPORT GENERATOR, misplaced) |
| `results/alignment/comparisons/code/generate_pairwise_report.py` | Pairwise report (REPORT GENERATOR, misplaced) |
| `results/alignment/comparisons/code/generate_pairwise_tests.py` | Pairwise tests (REPORT GENERATOR, misplaced) |

### Phase 3: Concept Probes (needs Phase 1)
| Script | Purpose |
|--------|---------|
| `pipeline/3_train_concept_probes.py` | Train binary probes on concept activations |
| `analysis/probes/4a_compute_alignment_stats.py` | Probe alignment statistics |
| `analysis/probes/4b_generate_alignment_figures.py` | Alignment figures (REPORT GENERATOR) |

### Phase 4: Concept Steering V1 (needs Phases 1 + 3)
| Script | Purpose |
|--------|---------|
| `pipeline/4_concept_steering_generate.py` | Single-turn steered generation |
| `pipeline/4b_concept_steering_behavior.py` | Behavioral analysis of steered output |

### Phase 5: Concept Intervention V2 (needs Phases 1 + 3)
| Script | Purpose |
|--------|---------|
| `pipeline/5_concept_intervention.py` | Multi-turn concept intervention |
| `pipeline/6_behavior_analysis.py` | Behavioral analysis of intervention output |

### Phase 6: Cross-Prediction (needs Phase 1 + exp_2)
| Script | Purpose |
|--------|---------|
| `pipeline/7_cross_prediction.py` | Cross-prediction between concepts and conversations |

### Standalone: Lexical Analysis
| Script | Purpose |
|--------|---------|
| `analysis/lexical/lexical_distinctiveness.py` | Lexical analysis of concept prompts |
| `analysis/lexical/generate_lexical_report.py` | Lexical report (REPORT GENERATOR) |

## Questions for Rachel

1. **Numbering collision resolution**: When flattening, the current `analysis/probes/4a` and `analysis/probes/4b` collide with `pipeline/4_concept_steering` and `pipeline/4b_concept_steering_behavior`. Should I:
   - (a) Renumber probe stats as **3a/3b** (since they're sub-analyses of step 3, train probes)?
   - (b) Keep current numbers and just disambiguate somehow?

2. **`concepts/` placement**: These are Python files with prompt lists — input definitions used by step 1. Move to `code/concepts/`?

3. **`data/causality_test_questions/human_ai.txt`**: Static input file. Move to `code/` (e.g. `code/concepts/causality_questions.txt`) or `code/utils/`?

4. **Model scoping**: Exp 3 only uses one model (LLaMA-2-13B-Chat). Still add `results/llama2_13b_chat/` level for consistency with exp_2, or skip it since there's only one model?

5. **`plan.md`**: This describes a future Phase 8 (concept-conversation alignment). Archive it, move to writeup, or keep at top level?

6. **`LICENSE`**: Keep at exp_3 top level or move somewhere?

7. **Results reorganization**: Currently there are two separate trees:
   - `results/alignment/versions/{version}/` (alignment .npz files)
   - `results/versions/{version}/concept_steering/` (steering output)

   Should these unify under one model-scoped tree? E.g.:
   ```
   results/llama2_13b_chat/
     {version}/
       alignment/
       concept_steering/
   ```

8. **Alignment comparisons**: Currently `results/alignment/comparisons/turn_{N}/` holds cross-version comparison reports. These span versions but not models. Should they go in:
   - `results/llama2_13b_chat/comparisons/alignment/`? (like exp_2 pattern)
   - Keep roughly where they are?
