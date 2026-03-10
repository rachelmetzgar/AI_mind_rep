# Exp 3: Pairwise Dimension Comparison — Raw Contrast Alignment (Partner Identity)

*Generated: 2026-03-08 14:56 | Turn: 5*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 1 | 6 | 6 | 7 |
| Emotions | Mental | 2 | 6 | 6 | 7 |
| Agency | Mental | 5 | 1 | 7 | 1 |
| Intentions | Mental | 6 | 0 | 9 | 0 |
| Prediction | Mental | 9 | 0 | 9 | 0 |
| Cognitive | Mental | 9 | 0 | 9 | 0 |
| Social | Mental | 9 | 0 | 10 | 0 |
| Attention | Mental | 12 | 0 | 11 | 0 |
| Embodiment | Physical | 0 | 12 | 5 | 9 |
| Roles | Physical | 6 | 2 | 7 | 2 |
| Animacy | Physical | 0 | 8 | 3 | 11 |
| Formality | Pragmatic | 0 | 9 | 0 | 15 |
| Expertise | Pragmatic | 2 | 6 | 3 | 12 |
| Helpfulness | Pragmatic | 0 | 9 | 0 | 15 |
| Baseline | Control | 9 | 1 | 9 | 0 |
| Biological | Control | 0 | 11 | 3 | 12 |
| Shapes | Control | 0 | 9 | 0 | 15 |
| SysPrompt (labeled) | SysPrompt | 10 | 0 | 9 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Shapes | Attention | -3.137 | 0.0000 | *** |
| Operational | Helpfulness | Attention | -3.101 | 0.0000 | *** |
| Operational | Formality | Attention | -3.064 | 0.0000 | *** |
| Operational | Social | Shapes | 2.981 | 0.0000 | *** |
| Operational | Social | Helpfulness | 2.945 | 0.0000 | *** |
| Operational | Social | Formality | 2.908 | 0.0000 | *** |
| Operational | Cognitive | Shapes | 2.766 | 0.0000 | *** |
| Operational | Cognitive | Helpfulness | 2.730 | 0.0000 | *** |
| Operational | Cognitive | Formality | 2.693 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (labeled) | -2.656 | 0.0000 | *** |
| Operational | Prediction | Shapes | 2.630 | 0.0000 | *** |
| Operational | Helpfulness | SysPrompt (labeled) | -2.621 | 0.0000 | *** |
| Operational | Prediction | Helpfulness | 2.595 | 0.0000 | *** |
| Operational | Formality | SysPrompt (labeled) | -2.584 | 0.0000 | *** |
| Operational | Prediction | Formality | 2.558 | 0.0000 | *** |
| Operational | Intentions | Shapes | 2.477 | 0.0000 | *** |
| Operational | Baseline | Shapes | 2.446 | 0.0000 | *** |
| Operational | Intentions | Helpfulness | 2.441 | 0.0000 | *** |
| Operational | Baseline | Helpfulness | 2.410 | 0.0000 | *** |
| Operational | Intentions | Formality | 2.404 | 0.0000 | *** |

## Methods

- **Analysis**: Raw Contrast Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
