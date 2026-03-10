# Exp 3: Pairwise Dimension Comparison — Residual Alignment (Partner Identity)

*Generated: 2026-03-08 14:56 | Turn: 5*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 2 | 0 | 5 |
| Emotions | Mental | 0 | 2 | 0 | 5 |
| Agency | Mental | 0 | 0 | 5 | 0 |
| Intentions | Mental | 0 | 0 | 8 | 0 |
| Prediction | Mental | 0 | 0 | 5 | 0 |
| Cognitive | Mental | 1 | 0 | 10 | 0 |
| Social | Mental | 0 | 0 | 8 | 0 |
| Attention | Mental | 5 | 0 | 11 | 0 |
| Embodiment | Physical | 0 | 3 | 0 | 8 |
| Roles | Physical | 0 | 0 | 0 | 1 |
| Animacy | Physical | 0 | 0 | 0 | 5 |
| Formality | Pragmatic | 0 | 1 | 0 | 8 |
| Expertise | Pragmatic | 0 | 1 | 4 | 2 |
| Helpfulness | Pragmatic | 0 | 1 | 0 | 7 |
| Baseline | Control | 0 | 0 | 0 | 2 |
| Biological | Control | 0 | 2 | 0 | 8 |
| Shapes | Control | 0 | 2 | 0 | 8 |
| SysPrompt (labeled) | SysPrompt | 8 | 0 | 8 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Shapes | Attention | -1.163 | 0.0000 | *** |
| Operational | Helpfulness | Attention | -1.134 | 0.0000 | *** |
| Operational | Formality | Attention | -1.126 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (labeled) | -1.120 | 0.0000 | *** |
| Operational | Embodiment | Attention | -1.105 | 0.0000 | *** |
| Operational | Helpfulness | SysPrompt (labeled) | -1.090 | 0.0078 | ** |
| Operational | Formality | SysPrompt (labeled) | -1.082 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (labeled) | -1.062 | 0.0000 | *** |
| Operational | Social | Shapes | 1.023 | 0.0000 | *** |
| Operational | Animacy | Attention | -1.017 | 0.0000 | *** |
| Operational | Biological | Attention | -1.015 | 0.0000 | *** |
| Operational | Social | Helpfulness | 0.993 | 0.0000 | *** |
| Operational | Cognitive | Shapes | 0.990 | 0.0000 | *** |
| Operational | Social | Formality | 0.985 | 0.0000 | *** |
| Operational | Baseline | Attention | -0.979 | 0.0136 | * |
| Operational | Animacy | SysPrompt (labeled) | -0.973 | 0.0000 | *** |
| Operational | Biological | SysPrompt (labeled) | -0.971 | 0.0000 | *** |
| Operational | Social | Embodiment | 0.965 | 0.0000 | *** |
| Operational | Cognitive | Helpfulness | 0.960 | 0.0000 | *** |
| Operational | Cognitive | Formality | 0.952 | 0.0000 | *** |

## Methods

- **Analysis**: Residual Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
