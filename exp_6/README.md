# Experiment 6: Multi-Agent Belief Tracking via RSA

Tests whether LLaMA-2-13B-Chat internally represents the epistemic states (who-believes-what) of multiple agents, and whether this representational geometry tracks ground-truth belief structure rather than surface features like communication topology or textual position.

Extends classical 2-agent Theory of Mind probing (Zhu et al. 2024, Bortoletto et al. 2024) to 4-agent networks with divergent beliefs, using RSA (Representational Similarity Analysis) on hidden-state geometry.

## Sub-Experiments

### 1_deception — Lie Propagation (Planned)

**Design:** Fixed 4-agent topology (Ann→Ben, Ann→Cam, Cam→Dan) where each communication edge can be truth or lie. 2^3 = 8 conditions, all with identical surface structure (same names, same sentence count, same extraction sentence). Conditions are grouped by lie count (0/1/2/3 lies), and within each group, different lie placements produce different belief geometries despite identical surface statistics.

**Key advantages over 1_network:**
- **Position confound eliminated:** Same extraction sentence and agent order in every condition
- **Surface-stats-matched comparisons:** E1 vs E2 (both 1 lie, different belief geometry), E3 vs E6 (both 2 lies, but E6 has a double-flip where two lies cancel out)
- **Double flip test:** In E6, Ann lies to Cam and Cam lies to Dan, so Dan ends up with the *correct* belief. No surface heuristic (counting lies) can predict this — requires compositional belief tracking

### 2_network — Displacement Narratives (Complete)

**Design:** 96 narratives = 3 topologies (chain/fork/diamond) x 4 conditions x 8 instantiations. An object moves and only some agents learn about it, creating divergent beliefs. RSA compares the model's agent-pair distance matrix to three candidate RDMs: epistemic (who shares beliefs), communication (who talked to whom), and position (token proximity).

**Key results:**
- 84% behavioral accuracy on belief attribution probes
- Peak epistemic RSA: r = 0.45 (layer 9)
- Peak communication RSA: r = 0.18
- Peak position RSA: r = 0.61
- Cross-topology consistency: r = 0.95, 0.99 (same beliefs, different networks)
- Epistemic > communication at 23/40 layers (Wilcoxon, BH-FDR corrected)

**Limitation:** Position RDM dominates epistemic RDM at every layer — agents closer together in the extraction sentence have more similar representations regardless of belief state. The epistemic RDM is partially confounded with position (see `confound.md`). This motivates the deception experiment.

## File Structure

```
exp_6/
  2_network/
    code/
      config.py
      utils/
        utils.py
      slurm/
        phase{1-5}_*.sh
      0_generate_stimuli.py
      1_behavioral_validation.py
      2_extract_activations.py
      3_construct_rdms.py
      4_rsa_analysis.py           # checkpointed: saves after each layer
      5_figures.py
      5a_results_summary_generator.py
    results/
      llama2_13b_chat/
        stimuli/                   # 96 narratives (JSON)
        behavioral/                # belief attribution accuracy
        activations/               # hidden states at agent name tokens (.pt)
        rdms/                      # pairwise cosine distance RDMs (.pkl)
        rsa/                       # RSA results, figures/, results_summary.html
    logs/
      {behavioral,activations,rdms,rsa,figures}/
    network_idea.md                # full experiment specification
    confound.md                    # position confound analysis
  1_deception/
    deception_idea.md              # full design with all 8 conditions
  .gitignore
  README.md
```

## Model

LLaMA-2-13B-Chat (same as Experiments 2-4). Activations extracted from residual stream at last token of each agent name across all 40 layers.

## Methods

- **RDM distance metric:** Cosine distance between agent activation vectors
- **RSA correlation:** Spearman rank correlation between model RDM and candidate RDMs
- **Statistical tests:** Wilcoxon signed-rank (epistemic vs communication RSA), permutation tests (10k permutations), BH-FDR correction across layers
- **Checkpointing:** Phase 4 saves after each layer and resumes on restart

## References

- Gurnee & Tegmark (2023). Language Models Represent Space and Time. *arXiv:2310.02207*.
- Zhu, Zhang & Wang (2024). Language Models Represent Beliefs of Self and Others. *ICML 2024*.
- Bortoletto et al. (2024). Brittle Minds, Fixable Activations. *arXiv:2406.17513*.
- Kriegeskorte, Mur & Bandettini (2008). Representational similarity analysis. *Frontiers in Systems Neuroscience*.
