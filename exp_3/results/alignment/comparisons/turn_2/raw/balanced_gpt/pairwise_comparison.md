# Exp 3: Pairwise Dimension Comparison — Raw Contrast Alignment (Partner Identity)

*Generated: 2026-03-04 11:36 | Turn: 2*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 7 | 5 | 9 |
| Emotions | Mental | 0 | 7 | 7 | 4 |
| Agency | Mental | 2 | 1 | 9 | 1 |
| Intentions | Mental | 2 | 1 | 8 | 1 |
| Prediction | Mental | 2 | 1 | 10 | 0 |
| Cognitive | Mental | 5 | 1 | 8 | 1 |
| Social | Mental | 4 | 1 | 14 | 0 |
| Attention | Mental | 6 | 1 | 10 | 0 |
| Embodiment | Physical | 0 | 12 | 5 | 7 |
| Roles | Physical | 0 | 3 | 8 | 1 |
| Animacy | Physical | 2 | 6 | 4 | 10 |
| Formality | Pragmatic | 2 | 1 | 0 | 15 |
| Expertise | Pragmatic | 6 | 1 | 3 | 12 |
| Helpfulness | Pragmatic | 0 | 1 | 0 | 15 |
| Baseline | Control | 5 | 1 | 6 | 3 |
| Biological | Control | 0 | 12 | 3 | 13 |
| Shapes | Control | 5 | 1 | 0 | 15 |
| SysPrompt (labeled) | SysPrompt | 17 | 0 | 7 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Social | Shapes | 8.250 | 0.0000 | *** |
| Operational | Social | Helpfulness | 8.073 | 0.0000 | *** |
| Operational | Shapes | Attention | -7.901 | 0.0000 | *** |
| Operational | Prediction | Shapes | 7.794 | 0.0000 | *** |
| Operational | Helpfulness | Attention | -7.724 | 0.0000 | *** |
| Operational | Prediction | Helpfulness | 7.617 | 0.0000 | *** |
| Operational | Social | Formality | 7.416 | 0.0000 | *** |
| Operational | Formality | Attention | -7.067 | 0.0000 | *** |
| Operational | Agency | Shapes | 7.061 | 0.0000 | *** |
| Operational | Prediction | Formality | 6.959 | 0.0000 | *** |
| Operational | Agency | Helpfulness | 6.884 | 0.0000 | *** |
| Operational | Roles | Shapes | 6.747 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (labeled) | -6.731 | 0.0000 | *** |
| Operational | Cognitive | Shapes | 6.694 | 0.0000 | *** |
| Operational | Intentions | Shapes | 6.656 | 0.0000 | *** |
| Operational | Roles | Helpfulness | 6.570 | 0.0000 | *** |
| Operational | Helpfulness | SysPrompt (labeled) | -6.554 | 0.0000 | *** |
| Operational | Cognitive | Helpfulness | 6.517 | 0.0000 | *** |
| Operational | Intentions | Helpfulness | 6.479 | 0.0000 | *** |
| Operational | Agency | Formality | 6.226 | 0.0000 | *** |

## Methods

- **Analysis**: Raw Contrast Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
