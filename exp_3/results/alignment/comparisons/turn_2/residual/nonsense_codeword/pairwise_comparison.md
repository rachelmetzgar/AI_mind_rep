# Exp 3: Pairwise Dimension Comparison — Residual Alignment (Control)

*Generated: 2026-03-04 11:37 | Turn: 2*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 0 | 1 | 0 |
| Emotions | Mental | 0 | 0 | 1 | 0 |
| Agency | Mental | 0 | 0 | 2 | 0 |
| Intentions | Mental | 2 | 0 | 2 | 0 |
| Prediction | Mental | 1 | 0 | 0 | 0 |
| Cognitive | Mental | 2 | 0 | 2 | 0 |
| Social | Mental | 2 | 0 | 2 | 0 |
| Attention | Mental | 2 | 0 | 2 | 0 |
| Embodiment | Physical | 0 | 0 | 1 | 0 |
| Roles | Physical | 0 | 0 | 0 | 0 |
| Animacy | Physical | 0 | 0 | 1 | 0 |
| Formality | Pragmatic | 1 | 0 | 2 | 0 |
| Expertise | Pragmatic | 0 | 6 | 0 | 12 |
| Helpfulness | Pragmatic | 0 | 0 | 1 | 0 |
| Baseline | Control | 0 | 0 | 0 | 0 |
| Biological | Control | 0 | 0 | 2 | 0 |
| Shapes | Control | 0 | 0 | 0 | 0 |
| SysPrompt (labeled) | SysPrompt | 0 | 4 | 0 | 7 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Expertise | Biological | -0.735 | 0.0000 | *** |
| Operational | Social | Expertise | 0.705 | 0.0000 | *** |
| Operational | Formality | Expertise | 0.672 | 0.0360 | * |
| Operational | Intentions | Expertise | 0.624 | 0.0000 | *** |
| Operational | Biological | SysPrompt (labeled) | 0.623 | 0.0000 | *** |
| Operational | Social | SysPrompt (labeled) | 0.593 | 0.0255 | * |
| Operational | Expertise | Helpfulness | -0.567 | 0.0000 | *** |
| Operational | Formality | SysPrompt (labeled) | 0.560 | 0.0360 | * |
| Operational | Cognitive | Expertise | 0.538 | 0.0255 | * |
| Operational | Embodiment | Expertise | 0.515 | 0.0255 | * |
| Operational | Intentions | SysPrompt (labeled) | 0.512 | 0.0000 | *** |
| Operational | Agency | Expertise | 0.507 | 0.0255 | * |
| Operational | Expertise | Attention | -0.452 | 0.0000 | *** |
| Operational | Emotions | Expertise | 0.439 | 0.0483 | * |
| Operational | Cognitive | SysPrompt (labeled) | 0.426 | 0.0255 | * |
| Operational | Phenomenology | Expertise | 0.419 | 0.0483 | * |
| Operational | Agency | SysPrompt (labeled) | 0.395 | 0.0360 | * |
| Operational | Animacy | Expertise | 0.364 | 0.0360 | * |
| Operational | Attention | SysPrompt (labeled) | 0.340 | 0.0360 | * |
| Metacognitive | Intentions | Expertise | 0.311 | 0.0000 | *** |

## Methods

- **Analysis**: Residual Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
