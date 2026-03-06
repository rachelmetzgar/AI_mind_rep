# Exp 3: Pairwise Dimension Comparison — Raw Contrast Alignment (Control)

*Generated: 2026-03-04 11:35 | Turn: 1*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 1 | 5 | 4 | 5 |
| Emotions | Mental | 1 | 6 | 4 | 8 |
| Agency | Mental | 2 | 2 | 8 | 1 |
| Intentions | Mental | 5 | 0 | 4 | 3 |
| Prediction | Mental | 3 | 1 | 9 | 1 |
| Cognitive | Mental | 5 | 1 | 4 | 6 |
| Social | Mental | 5 | 0 | 9 | 1 |
| Attention | Mental | 4 | 1 | 7 | 1 |
| Embodiment | Physical | 1 | 8 | 6 | 1 |
| Roles | Physical | 3 | 1 | 10 | 1 |
| Animacy | Physical | 2 | 2 | 10 | 1 |
| Formality | Pragmatic | 0 | 11 | 0 | 15 |
| Expertise | Pragmatic | 2 | 2 | 1 | 14 |
| Helpfulness | Pragmatic | 0 | 2 | 0 | 14 |
| Baseline | Control | 14 | 0 | 17 | 0 |
| Biological | Control | 0 | 14 | 4 | 7 |
| Shapes | Control | 0 | 1 | 0 | 14 |
| SysPrompt (labeled) | SysPrompt | 9 | 0 | 4 | 8 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Baseline | Formality | 8.614 | 0.0000 | *** |
| Operational | Baseline | Helpfulness | 8.552 | 0.0000 | *** |
| Operational | Baseline | Expertise | 8.352 | 0.0000 | *** |
| Operational | Baseline | Shapes | 8.294 | 0.0000 | *** |
| Operational | Roles | Formality | 5.722 | 0.0000 | *** |
| Operational | Roles | Helpfulness | 5.660 | 0.0000 | *** |
| Operational | Animacy | Formality | 5.514 | 0.0000 | *** |
| Operational | Roles | Expertise | 5.461 | 0.0000 | *** |
| Operational | Animacy | Helpfulness | 5.452 | 0.0000 | *** |
| Operational | Roles | Shapes | 5.402 | 0.0000 | *** |
| Operational | Social | Formality | 5.257 | 0.0000 | *** |
| Operational | Animacy | Expertise | 5.253 | 0.0000 | *** |
| Operational | Social | Helpfulness | 5.195 | 0.0000 | *** |
| Operational | Animacy | Shapes | 5.194 | 0.0000 | *** |
| Operational | Baseline | SysPrompt (labeled) | 5.185 | 0.0000 | *** |
| Operational | Baseline | Emotions | 5.106 | 0.0000 | *** |
| Operational | Prediction | Formality | 5.077 | 0.0000 | *** |
| Operational | Prediction | Helpfulness | 5.015 | 0.0000 | *** |
| Operational | Social | Expertise | 4.995 | 0.0000 | *** |
| Operational | Baseline | Biological | 4.938 | 0.0000 | *** |

## Methods

- **Analysis**: Raw Contrast Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
