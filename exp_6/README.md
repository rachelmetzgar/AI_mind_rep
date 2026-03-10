# Experiment 6: Multi-Agent Belief Propagation

## Purpose

Tests whether LLaMA-2-13B-Chat internally represents the epistemic states (who-believes-what) of multiple agents in false-belief narratives, and whether this representational geometry tracks ground-truth belief structure rather than surface features like communication topology or textual position.

Extends classical 2-agent Theory of Mind probing (Zhu et al. 2024, Bortoletto et al. 2024) to 4-agent networks with divergent beliefs, using RSA (Representational Similarity Analysis) on hidden-state geometry.

## Design

**96 narratives** = 3 topologies x 4 conditions x 8 instantiations

### Topologies (how information spreads)
- **Chain:** A -> B -> C -> D
- **Fork:** A -> B, A -> C, A -> D
- **Diamond:** A -> B, A -> C; B -> D, C -> D

### Conditions (per topology)
- 3 override conditions: object moves, different agents witness/learn about it
- 1 no-override control: no move, everyone shares the same belief

### Key dissociations
- `fork_override_D_tells_A`: Epistemic alignment (A,D agree) doesn't follow communication structure
- Cross-topology matches: Same belief partition, different communication network

### Confound controls
- 8 name sets rotated across instantiations (name confound)
- Fixed extraction sentence with all 4 names (position confound)
- 3 candidate RDMs: epistemic, communication, position (baseline controls)
- 8 object/location scenarios (content confound)

## Pipeline

```
Phase 0: Stimulus Generation          [CPU, ~5 min]
Phase 1: Behavioral Validation         [GPU, ~5 min]
Phase 2: Activation Extraction         [GPU, ~1 min]
Phase 3: RDM Construction             [CPU, ~20 sec]
Phase 4: RSA Analysis + Permutations  [CPU, ~1 hr]
Phase 5: Figures                       [CPU, ~5 min]
```

## File Structure

```
exp_6/
  code/
    config.py
    utils/
      utils.py
    slurm/
      phase1_behavioral.sh
      phase2_extract.sh
      phase3_rdms.sh
      phase4_rsa.sh
      phase5_figures.sh
    0_generate_stimuli.py
    1_behavioral_validation.py
    2_extract_activations.py
    3_construct_rdms.py
    4_rsa_analysis.py
    5_figures.py
  results/
    llama2_13b_chat/
      stimuli/                  # 96 generated narratives (JSON)
      behavioral/               # belief attribution accuracy
      activations/              # hidden states at agent name tokens (.pt)
      rdms/                     # pairwise cosine distance RDMs (.pkl)
      rsa/                      # RSA results + figures/
  logs/
    {behavioral,activations,rdms,rsa,figures}/
  EXPERIMENT_SPEC.md            # full design rationale and methodology
  README.md
```

## Model

LLaMA-2-13B-Chat (same as Experiments 2-4). Activations extracted from residual stream at last token of each agent name across all 40 layers.

## Key Methods

- **RDM distance metric:** Cosine distance between agent activation vectors
- **RSA correlation:** Spearman rank correlation between model RDM and candidate RDMs
- **Statistical tests:** Wilcoxon signed-rank (epistemic vs communication RSA), permutation tests (10k permutations), BH-FDR correction across layers
- **Checkpointing:** Phase 4 saves after each layer and resumes on restart
