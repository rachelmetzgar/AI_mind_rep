# Exp 3: Pairwise Dimension Comparison — Raw Contrast Alignment (Control)

*Generated: 2026-03-04 11:36 | Turn: 2*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 5 | 0 | 4 | 0 |
| Emotions | Mental | 4 | 1 | 4 | 0 |
| Agency | Mental | 5 | 0 | 5 | 0 |
| Intentions | Mental | 10 | 0 | 5 | 0 |
| Prediction | Mental | 6 | 0 | 4 | 0 |
| Cognitive | Mental | 6 | 0 | 3 | 0 |
| Social | Mental | 6 | 0 | 5 | 0 |
| Attention | Mental | 8 | 0 | 3 | 0 |
| Embodiment | Physical | 4 | 2 | 5 | 0 |
| Roles | Physical | 4 | 0 | 4 | 0 |
| Animacy | Physical | 0 | 12 | 4 | 0 |
| Formality | Pragmatic | 1 | 1 | 1 | 11 |
| Expertise | Pragmatic | 0 | 14 | 0 | 17 |
| Helpfulness | Pragmatic | 0 | 11 | 1 | 5 |
| Baseline | Control | 4 | 2 | 4 | 0 |
| Biological | Control | 2 | 5 | 5 | 0 |
| Shapes | Control | 0 | 7 | 1 | 13 |
| SysPrompt (labeled) | SysPrompt | 1 | 11 | 1 | 13 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Social | Expertise | 1.390 | 0.0000 | *** |
| Operational | Expertise | Biological | -1.286 | 0.0000 | *** |
| Operational | Embodiment | Expertise | 1.249 | 0.0000 | *** |
| Operational | Intentions | Expertise | 1.205 | 0.0000 | *** |
| Operational | Social | Shapes | 1.196 | 0.0000 | *** |
| Operational | Baseline | Expertise | 1.172 | 0.0000 | *** |
| Operational | Agency | Expertise | 1.153 | 0.0000 | *** |
| Operational | Social | SysPrompt (labeled) | 1.121 | 0.0000 | *** |
| Operational | Phenomenology | Expertise | 1.102 | 0.0000 | *** |
| Operational | Biological | Shapes | 1.092 | 0.0000 | *** |
| Operational | Roles | Expertise | 1.068 | 0.0000 | *** |
| Operational | Prediction | Expertise | 1.066 | 0.0000 | *** |
| Operational | Embodiment | Shapes | 1.054 | 0.0000 | *** |
| Operational | Biological | SysPrompt (labeled) | 1.016 | 0.0000 | *** |
| Operational | Intentions | Shapes | 1.010 | 0.0000 | *** |
| Operational | Emotions | Expertise | 1.003 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (labeled) | 0.979 | 0.0000 | *** |
| Operational | Expertise | Attention | -0.978 | 0.0000 | *** |
| Operational | Baseline | Shapes | 0.977 | 0.0000 | *** |
| Operational | Social | Formality | 0.963 | 0.0000 | *** |

## Methods

- **Analysis**: Raw Contrast Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
