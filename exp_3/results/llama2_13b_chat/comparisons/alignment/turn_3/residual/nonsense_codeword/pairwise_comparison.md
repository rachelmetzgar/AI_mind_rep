# Exp 3: Pairwise Dimension Comparison — Residual Alignment (Control)

*Generated: 2026-03-04 11:38 | Turn: 3*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 0 | 0 | 0 |
| Emotions | Mental | 0 | 1 | 0 | 2 |
| Agency | Mental | 0 | 0 | 0 | 0 |
| Intentions | Mental | 0 | 0 | 3 | 0 |
| Prediction | Mental | 0 | 0 | 0 | 0 |
| Cognitive | Mental | 0 | 0 | 5 | 0 |
| Social | Mental | 1 | 0 | 0 | 1 |
| Attention | Mental | 0 | 0 | 0 | 0 |
| Embodiment | Physical | 0 | 0 | 0 | 0 |
| Roles | Physical | 0 | 0 | 0 | 0 |
| Animacy | Physical | 0 | 0 | 0 | 3 |
| Formality | Pragmatic | 0 | 0 | 0 | 0 |
| Expertise | Pragmatic | 0 | 0 | 2 | 0 |
| Helpfulness | Pragmatic | 0 | 0 | 0 | 0 |
| Baseline | Control | 0 | 0 | 0 | 0 |
| Biological | Control | 0 | 0 | 0 | 1 |
| Shapes | Control | 0 | 0 | 0 | 0 |
| SysPrompt (labeled) | SysPrompt | 0 | 0 | 0 | 3 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Cognitive | Animacy | 0.945 | 0.0000 | *** |
| Operational | Cognitive | SysPrompt (labeled) | 0.908 | 0.0000 | *** |
| Operational | Emotions | Cognitive | -0.840 | 0.0306 | * |
| Operational | Cognitive | Social | 0.759 | 0.0306 | * |
| Operational | Cognitive | Biological | 0.746 | 0.0000 | *** |
| Operational | Intentions | Animacy | 0.745 | 0.0306 | * |
| Operational | Intentions | SysPrompt (labeled) | 0.708 | 0.0000 | *** |
| Operational | Animacy | Expertise | -0.692 | 0.0000 | *** |
| Operational | Expertise | SysPrompt (labeled) | 0.655 | 0.0000 | *** |
| Operational | Emotions | Intentions | -0.640 | 0.0000 | *** |
| Metacognitive | Emotions | Social | -0.087 | 0.0000 | *** |

## Methods

- **Analysis**: Residual Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
