# Exp 3: Pairwise Dimension Comparison — Residual Alignment (Partner Identity)

*Generated: 2026-03-04 11:38 | Turn: 3*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 1 | 0 | 3 | 7 |
| Emotions | Mental | 2 | 0 | 8 | 2 |
| Agency | Mental | 1 | 0 | 8 | 0 |
| Intentions | Mental | 1 | 0 | 8 | 0 |
| Prediction | Mental | 1 | 0 | 11 | 0 |
| Cognitive | Mental | 1 | 0 | 8 | 0 |
| Social | Mental | 2 | 0 | 10 | 0 |
| Attention | Mental | 2 | 0 | 11 | 0 |
| Embodiment | Physical | 0 | 4 | 3 | 8 |
| Roles | Physical | 0 | 0 | 5 | 3 |
| Animacy | Physical | 0 | 0 | 0 | 12 |
| Formality | Pragmatic | 1 | 0 | 1 | 9 |
| Expertise | Pragmatic | 2 | 0 | 6 | 3 |
| Helpfulness | Pragmatic | 0 | 0 | 0 | 10 |
| Baseline | Control | 0 | 0 | 0 | 9 |
| Biological | Control | 0 | 12 | 0 | 12 |
| Shapes | Control | 1 | 0 | 0 | 13 |
| SysPrompt (labeled) | SysPrompt | 1 | 0 | 6 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Shapes | Attention | -2.124 | 0.0000 | *** |
| Operational | Helpfulness | Attention | -2.041 | 0.0000 | *** |
| Operational | Biological | Attention | -2.023 | 0.0000 | *** |
| Operational | Animacy | Attention | -1.998 | 0.0000 | *** |
| Operational | Social | Shapes | 1.984 | 0.0000 | *** |
| Operational | Baseline | Attention | -1.973 | 0.0000 | *** |
| Operational | Prediction | Shapes | 1.921 | 0.0000 | *** |
| Operational | Social | Helpfulness | 1.900 | 0.0000 | *** |
| Operational | Social | Biological | 1.882 | 0.0000 | *** |
| Operational | Social | Animacy | 1.857 | 0.0000 | *** |
| Operational | Prediction | Helpfulness | 1.837 | 0.0000 | *** |
| Operational | Baseline | Social | -1.833 | 0.0000 | *** |
| Operational | Prediction | Biological | 1.819 | 0.0000 | *** |
| Operational | Prediction | Animacy | 1.794 | 0.0000 | *** |
| Operational | Baseline | Prediction | -1.769 | 0.0000 | *** |
| Operational | Formality | Attention | -1.742 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (labeled) | -1.651 | 0.0000 | *** |
| Operational | Cognitive | Shapes | 1.624 | 0.0000 | *** |
| Operational | Social | Formality | 1.601 | 0.0000 | *** |
| Operational | Embodiment | Attention | -1.591 | 0.0000 | *** |

## Methods

- **Analysis**: Residual Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
