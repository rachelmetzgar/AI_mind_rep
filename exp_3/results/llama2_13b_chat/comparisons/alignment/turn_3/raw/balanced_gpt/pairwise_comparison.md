# Exp 3: Pairwise Dimension Comparison — Raw Contrast Alignment (Partner Identity)

*Generated: 2026-03-04 11:38 | Turn: 3*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 5 | 5 | 7 |
| Emotions | Mental | 0 | 2 | 6 | 3 |
| Agency | Mental | 0 | 1 | 8 | 2 |
| Intentions | Mental | 0 | 0 | 8 | 2 |
| Prediction | Mental | 0 | 0 | 11 | 0 |
| Cognitive | Mental | 0 | 0 | 8 | 1 |
| Social | Mental | 2 | 0 | 14 | 0 |
| Attention | Mental | 1 | 0 | 13 | 0 |
| Embodiment | Physical | 0 | 2 | 5 | 7 |
| Roles | Physical | 0 | 1 | 8 | 3 |
| Animacy | Physical | 0 | 0 | 5 | 10 |
| Formality | Pragmatic | 0 | 0 | 1 | 15 |
| Expertise | Pragmatic | 2 | 0 | 3 | 13 |
| Helpfulness | Pragmatic | 0 | 0 | 0 | 15 |
| Baseline | Control | 4 | 0 | 6 | 3 |
| Biological | Control | 0 | 4 | 3 | 13 |
| Shapes | Control | 0 | 0 | 0 | 16 |
| SysPrompt (labeled) | SysPrompt | 6 | 0 | 6 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Social | Shapes | 5.844 | 0.0000 | *** |
| Operational | Social | Helpfulness | 5.789 | 0.0000 | *** |
| Operational | Shapes | Attention | -5.674 | 0.0000 | *** |
| Operational | Helpfulness | Attention | -5.619 | 0.0000 | *** |
| Operational | Prediction | Shapes | 5.424 | 0.0000 | *** |
| Operational | Prediction | Helpfulness | 5.369 | 0.0000 | *** |
| Operational | Social | Formality | 5.213 | 0.0000 | *** |
| Operational | Formality | Attention | -5.043 | 0.0000 | *** |
| Operational | Cognitive | Shapes | 4.861 | 0.0000 | *** |
| Operational | Cognitive | Helpfulness | 4.806 | 0.0000 | *** |
| Operational | Prediction | Formality | 4.793 | 0.0000 | *** |
| Operational | Agency | Shapes | 4.778 | 0.0000 | *** |
| Operational | Agency | Helpfulness | 4.723 | 0.0000 | *** |
| Operational | Intentions | Shapes | 4.517 | 0.0000 | *** |
| Operational | Roles | Shapes | 4.476 | 0.0000 | *** |
| Operational | Intentions | Helpfulness | 4.462 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (labeled) | -4.461 | 0.0000 | *** |
| Operational | Roles | Helpfulness | 4.421 | 0.0000 | *** |
| Operational | Helpfulness | SysPrompt (labeled) | -4.406 | 0.0000 | *** |
| Operational | Baseline | Shapes | 4.316 | 0.0000 | *** |

## Methods

- **Analysis**: Raw Contrast Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
