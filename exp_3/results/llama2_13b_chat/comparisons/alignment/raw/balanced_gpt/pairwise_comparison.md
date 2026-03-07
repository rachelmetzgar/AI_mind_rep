# Exp 3: Pairwise Dimension Comparison — Raw Contrast Alignment (Partner Identity)

*Generated: 2026-03-04 11:35 | Turn: 1*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 4 | 5 | 8 |
| Emotions | Mental | 0 | 5 | 5 | 7 |
| Agency | Mental | 0 | 4 | 8 | 1 |
| Intentions | Mental | 0 | 2 | 5 | 4 |
| Prediction | Mental | 0 | 1 | 10 | 1 |
| Cognitive | Mental | 7 | 1 | 7 | 1 |
| Social | Mental | 11 | 0 | 10 | 1 |
| Attention | Mental | 6 | 1 | 10 | 1 |
| Embodiment | Physical | 0 | 5 | 4 | 8 |
| Roles | Physical | 0 | 4 | 9 | 1 |
| Animacy | Physical | 0 | 3 | 5 | 6 |
| Formality | Pragmatic | 0 | 2 | 0 | 15 |
| Expertise | Pragmatic | 2 | 2 | 0 | 14 |
| Helpfulness | Pragmatic | 0 | 1 | 0 | 14 |
| Baseline | Control | 0 | 1 | 9 | 1 |
| Biological | Control | 0 | 4 | 4 | 12 |
| Shapes | Control | 0 | 2 | 1 | 14 |
| SysPrompt (labeled) | SysPrompt | 16 | 0 | 17 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Formality | SysPrompt (labeled) | -4.866 | 0.0000 | *** |
| Operational | Expertise | SysPrompt (labeled) | -4.752 | 0.0000 | *** |
| Operational | Helpfulness | SysPrompt (labeled) | -4.666 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (labeled) | -4.609 | 0.0000 | *** |
| Operational | Baseline | Formality | 3.463 | 0.0000 | *** |
| Operational | Baseline | Expertise | 3.349 | 0.0000 | *** |
| Operational | Formality | Attention | -3.316 | 0.0000 | *** |
| Operational | Prediction | Formality | 3.315 | 0.0000 | *** |
| Operational | Social | Formality | 3.266 | 0.0000 | *** |
| Operational | Baseline | Helpfulness | 3.263 | 0.0000 | *** |
| Operational | Baseline | Shapes | 3.206 | 0.0000 | *** |
| Operational | Expertise | Attention | -3.202 | 0.0000 | *** |
| Operational | Prediction | Expertise | 3.200 | 0.0000 | *** |
| Operational | Social | Expertise | 3.151 | 0.0000 | *** |
| Operational | Helpfulness | Attention | -3.116 | 0.0000 | *** |
| Operational | Prediction | Helpfulness | 3.115 | 0.0000 | *** |
| Operational | Biological | SysPrompt (labeled) | -3.072 | 0.0000 | *** |
| Operational | Roles | Formality | 3.070 | 0.0000 | *** |
| Operational | Social | Helpfulness | 3.066 | 0.0000 | *** |
| Operational | Shapes | Attention | -3.059 | 0.0000 | *** |

## Methods

- **Analysis**: Raw Contrast Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
