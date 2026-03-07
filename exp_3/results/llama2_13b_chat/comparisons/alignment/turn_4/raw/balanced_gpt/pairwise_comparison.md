# Exp 3: Pairwise Dimension Comparison — Raw Contrast Alignment (Partner Identity)

*Generated: 2026-03-04 11:39 | Turn: 4*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 10 | 6 | 6 |
| Emotions | Mental | 0 | 7 | 7 | 4 |
| Agency | Mental | 0 | 1 | 8 | 1 |
| Intentions | Mental | 2 | 0 | 8 | 1 |
| Prediction | Mental | 3 | 1 | 9 | 0 |
| Cognitive | Mental | 6 | 0 | 9 | 0 |
| Social | Mental | 3 | 0 | 9 | 0 |
| Attention | Mental | 5 | 0 | 13 | 0 |
| Embodiment | Physical | 0 | 9 | 5 | 10 |
| Roles | Physical | 0 | 6 | 7 | 1 |
| Animacy | Physical | 5 | 1 | 5 | 11 |
| Formality | Pragmatic | 0 | 2 | 0 | 15 |
| Expertise | Pragmatic | 5 | 1 | 3 | 13 |
| Helpfulness | Pragmatic | 1 | 0 | 0 | 15 |
| Baseline | Control | 5 | 1 | 7 | 1 |
| Biological | Control | 0 | 11 | 3 | 13 |
| Shapes | Control | 4 | 0 | 0 | 15 |
| SysPrompt (labeled) | SysPrompt | 11 | 0 | 7 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Shapes | Attention | -5.414 | 0.0000 | *** |
| Operational | Helpfulness | Attention | -5.377 | 0.0000 | *** |
| Operational | Formality | Attention | -5.086 | 0.0000 | *** |
| Operational | Social | Shapes | 4.873 | 0.0000 | *** |
| Operational | Social | Helpfulness | 4.836 | 0.0000 | *** |
| Operational | Prediction | Shapes | 4.761 | 0.0000 | *** |
| Operational | Prediction | Helpfulness | 4.723 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (labeled) | -4.609 | 0.0000 | *** |
| Operational | Helpfulness | SysPrompt (labeled) | -4.572 | 0.0000 | *** |
| Operational | Cognitive | Shapes | 4.550 | 0.0000 | *** |
| Operational | Social | Formality | 4.545 | 0.0000 | *** |
| Operational | Cognitive | Helpfulness | 4.513 | 0.0000 | *** |
| Operational | Prediction | Formality | 4.432 | 0.0000 | *** |
| Operational | Formality | SysPrompt (labeled) | -4.280 | 0.0000 | *** |
| Operational | Cognitive | Formality | 4.222 | 0.0000 | *** |
| Operational | Expertise | Attention | -4.182 | 0.0000 | *** |
| Operational | Agency | Shapes | 4.105 | 0.0000 | *** |
| Operational | Agency | Helpfulness | 4.068 | 0.0000 | *** |
| Operational | Intentions | Shapes | 4.049 | 0.0000 | *** |
| Operational | Intentions | Helpfulness | 4.012 | 0.0000 | *** |

## Methods

- **Analysis**: Raw Contrast Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
