# Exp 3: Pairwise Dimension Comparison — Residual Alignment (Partner Identity)

*Generated: 2026-03-04 11:37 | Turn: 2*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 3 | 1 | 10 |
| Emotions | Mental | 1 | 1 | 8 | 3 |
| Agency | Mental | 1 | 1 | 8 | 0 |
| Intentions | Mental | 1 | 1 | 8 | 0 |
| Prediction | Mental | 0 | 0 | 9 | 0 |
| Cognitive | Mental | 1 | 1 | 8 | 0 |
| Social | Mental | 1 | 0 | 9 | 0 |
| Attention | Mental | 4 | 0 | 10 | 0 |
| Embodiment | Physical | 0 | 3 | 2 | 9 |
| Roles | Physical | 0 | 0 | 6 | 1 |
| Animacy | Physical | 0 | 1 | 0 | 10 |
| Formality | Pragmatic | 1 | 1 | 0 | 10 |
| Expertise | Pragmatic | 4 | 0 | 8 | 0 |
| Helpfulness | Pragmatic | 0 | 3 | 0 | 10 |
| Baseline | Control | 0 | 1 | 0 | 9 |
| Biological | Control | 0 | 10 | 0 | 12 |
| Shapes | Control | 1 | 0 | 0 | 11 |
| SysPrompt (labeled) | SysPrompt | 11 | 0 | 8 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Shapes | Attention | -2.726 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (labeled) | -2.710 | 0.0000 | *** |
| Operational | Social | Shapes | 2.684 | 0.0000 | *** |
| Operational | Biological | Attention | -2.632 | 0.0000 | *** |
| Operational | Biological | SysPrompt (labeled) | -2.616 | 0.0000 | *** |
| Operational | Social | Biological | 2.591 | 0.0000 | *** |
| Operational | Prediction | Shapes | 2.556 | 0.0000 | *** |
| Operational | Expertise | Shapes | 2.556 | 0.0000 | *** |
| Operational | Helpfulness | Attention | -2.508 | 0.0000 | *** |
| Operational | Animacy | Attention | -2.496 | 0.0000 | *** |
| Operational | Helpfulness | SysPrompt (labeled) | -2.492 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (labeled) | -2.480 | 0.0000 | *** |
| Operational | Social | Helpfulness | 2.467 | 0.0000 | *** |
| Operational | Prediction | Biological | 2.462 | 0.0000 | *** |
| Operational | Expertise | Biological | 2.462 | 0.0000 | *** |
| Operational | Social | Animacy | 2.455 | 0.0000 | *** |
| Operational | Baseline | Attention | -2.398 | 0.0000 | *** |
| Operational | Baseline | SysPrompt (labeled) | -2.382 | 0.0157 | * |
| Operational | Formality | Attention | -2.362 | 0.0000 | *** |
| Operational | Baseline | Social | -2.357 | 0.0000 | *** |

## Methods

- **Analysis**: Residual Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
