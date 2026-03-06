# Exp 3: Pairwise Dimension Comparison — Residual Alignment (Partner Identity)

*Generated: 2026-03-04 11:39 | Turn: 4*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 2 | 2 | 7 |
| Emotions | Mental | 1 | 0 | 6 | 1 |
| Agency | Mental | 1 | 0 | 8 | 0 |
| Intentions | Mental | 1 | 0 | 8 | 0 |
| Prediction | Mental | 0 | 0 | 8 | 0 |
| Cognitive | Mental | 2 | 0 | 10 | 0 |
| Social | Mental | 1 | 0 | 8 | 0 |
| Attention | Mental | 3 | 0 | 11 | 0 |
| Embodiment | Physical | 0 | 3 | 0 | 11 |
| Roles | Physical | 0 | 0 | 2 | 2 |
| Animacy | Physical | 0 | 0 | 0 | 9 |
| Formality | Pragmatic | 0 | 1 | 0 | 9 |
| Expertise | Pragmatic | 1 | 0 | 5 | 2 |
| Helpfulness | Pragmatic | 0 | 0 | 0 | 8 |
| Baseline | Control | 0 | 0 | 0 | 7 |
| Biological | Control | 0 | 8 | 0 | 11 |
| Shapes | Control | 0 | 0 | 0 | 9 |
| SysPrompt (labeled) | SysPrompt | 4 | 0 | 8 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Biological | Attention | -2.203 | 0.0000 | *** |
| Operational | Biological | SysPrompt (labeled) | -2.152 | 0.0000 | *** |
| Operational | Shapes | Attention | -2.114 | 0.0000 | *** |
| Operational | Animacy | Attention | -2.090 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (labeled) | -2.063 | 0.0000 | *** |
| Operational | Embodiment | Attention | -2.056 | 0.0000 | *** |
| Operational | Helpfulness | Attention | -2.043 | 0.0000 | *** |
| Operational | Baseline | Attention | -2.040 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (labeled) | -2.038 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (labeled) | -2.005 | 0.0000 | *** |
| Operational | Helpfulness | SysPrompt (labeled) | -1.991 | 0.0056 | ** |
| Operational | Baseline | SysPrompt (labeled) | -1.988 | 0.0194 | * |
| Operational | Formality | Attention | -1.952 | 0.0000 | *** |
| Operational | Formality | SysPrompt (labeled) | -1.901 | 0.0000 | *** |
| Operational | Cognitive | Biological | 1.762 | 0.0000 | *** |
| Operational | Cognitive | Shapes | 1.673 | 0.0000 | *** |
| Operational | Phenomenology | Attention | -1.673 | 0.0000 | *** |
| Operational | Cognitive | Animacy | 1.648 | 0.0000 | *** |
| Operational | Prediction | Biological | 1.634 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (labeled) | -1.622 | 0.0000 | *** |

## Methods

- **Analysis**: Residual Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
