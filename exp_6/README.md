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

## File Structure

```
exp_6/
  1_deception/
    deception_idea.md              # full design with all 8 conditions
  .gitignore
  README.md
```

## Model

LLaMA-2-13B-Chat (same as Experiments 2-4). Activations extracted from residual stream at last token of each agent name across all 40 layers.
