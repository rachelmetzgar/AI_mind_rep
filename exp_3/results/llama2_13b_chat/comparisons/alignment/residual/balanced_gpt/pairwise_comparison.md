# Exp 3: Pairwise Dimension Comparison — Residual Alignment (Partner Identity)

*Generated: 2026-03-04 11:35 | Turn: 1*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 5 | 1 | 7 |
| Emotions | Mental | 0 | 2 | 3 | 4 |
| Agency | Mental | 0 | 2 | 8 | 1 |
| Intentions | Mental | 0 | 2 | 6 | 1 |
| Prediction | Mental | 1 | 0 | 9 | 1 |
| Cognitive | Mental | 3 | 2 | 7 | 1 |
| Social | Mental | 12 | 0 | 9 | 1 |
| Attention | Mental | 3 | 1 | 9 | 1 |
| Embodiment | Physical | 0 | 6 | 1 | 7 |
| Roles | Physical | 0 | 2 | 1 | 1 |
| Animacy | Physical | 0 | 2 | 1 | 7 |
| Formality | Pragmatic | 0 | 2 | 0 | 15 |
| Expertise | Pragmatic | 3 | 2 | 0 | 8 |
| Helpfulness | Pragmatic | 0 | 1 | 1 | 5 |
| Baseline | Control | 0 | 1 | 0 | 1 |
| Biological | Control | 0 | 5 | 1 | 8 |
| Shapes | Control | 0 | 2 | 1 | 6 |
| SysPrompt (labeled) | SysPrompt | 15 | 0 | 17 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Formality | SysPrompt (labeled) | -2.328 | 0.0000 | *** |
| Operational | Expertise | SysPrompt (labeled) | -2.204 | 0.0000 | *** |
| Operational | Biological | SysPrompt (labeled) | -2.046 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (labeled) | -2.028 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (labeled) | -2.027 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (labeled) | -2.025 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (labeled) | -2.010 | 0.0000 | *** |
| Operational | Baseline | SysPrompt (labeled) | -2.003 | 0.0107 | * |
| Operational | Helpfulness | SysPrompt (labeled) | -2.003 | 0.0000 | *** |
| Operational | Emotions | SysPrompt (labeled) | -1.799 | 0.0000 | *** |
| Operational | Roles | SysPrompt (labeled) | -1.732 | 0.0000 | *** |
| Operational | Intentions | SysPrompt (labeled) | -1.678 | 0.0000 | *** |
| Operational | Cognitive | SysPrompt (labeled) | -1.585 | 0.0000 | *** |
| Operational | Agency | SysPrompt (labeled) | -1.522 | 0.0000 | *** |
| Operational | Social | SysPrompt (labeled) | -1.467 | 0.0000 | *** |
| Operational | Prediction | SysPrompt (labeled) | -1.390 | 0.0000 | *** |
| Operational | Attention | SysPrompt (labeled) | -1.360 | 0.0059 | ** |
| Operational | Formality | Attention | -0.968 | 0.0000 | *** |
| Operational | Prediction | Formality | 0.938 | 0.0000 | *** |
| Operational | Social | Formality | 0.861 | 0.0000 | *** |

## Methods

- **Analysis**: Residual Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
