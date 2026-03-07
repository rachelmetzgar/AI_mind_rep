# Exp 3: Pairwise Dimension Comparison — Residual Alignment (Control)

*Generated: 2026-03-04 11:36 | Turn: 1*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 1 | 4 | 0 | 0 |
| Emotions | Mental | 3 | 0 | 0 | 0 |
| Agency | Mental | 2 | 0 | 0 | 0 |
| Intentions | Mental | 3 | 0 | 0 | 0 |
| Prediction | Mental | 2 | 0 | 0 | 0 |
| Cognitive | Mental | 2 | 1 | 0 | 0 |
| Social | Mental | 6 | 0 | 0 | 0 |
| Attention | Mental | 2 | 0 | 0 | 0 |
| Embodiment | Physical | 0 | 9 | 0 | 0 |
| Roles | Physical | 1 | 0 | 0 | 0 |
| Animacy | Physical | 1 | 1 | 0 | 0 |
| Formality | Pragmatic | 0 | 2 | 0 | 0 |
| Expertise | Pragmatic | 4 | 0 | 0 | 0 |
| Helpfulness | Pragmatic | 1 | 0 | 0 | 0 |
| Baseline | Control | 0 | 0 | 0 | 0 |
| Biological | Control | 0 | 14 | 0 | 0 |
| Shapes | Control | 1 | 0 | 0 | 0 |
| SysPrompt (labeled) | SysPrompt | 2 | 0 | 0 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Metacognitive | Social | Biological | 0.416 | 0.0000 | *** |
| Metacognitive | Social | Embodiment | 0.344 | 0.0000 | *** |
| Metacognitive | Intentions | Biological | 0.337 | 0.0000 | *** |
| Metacognitive | Social | Formality | 0.310 | 0.0000 | *** |
| Metacognitive | Expertise | Biological | 0.310 | 0.0000 | *** |
| Metacognitive | Emotions | Biological | 0.310 | 0.0000 | *** |
| Metacognitive | Phenomenology | Social | -0.282 | 0.0000 | *** |
| Metacognitive | Social | Animacy | 0.274 | 0.0408 | * |
| Metacognitive | Intentions | Embodiment | 0.265 | 0.0000 | *** |
| Metacognitive | Biological | Shapes | -0.258 | 0.0000 | *** |
| Metacognitive | Biological | Attention | -0.251 | 0.0000 | *** |
| Metacognitive | Prediction | Biological | 0.245 | 0.0000 | *** |
| Metacognitive | Agency | Biological | 0.242 | 0.0000 | *** |
| Metacognitive | Embodiment | Expertise | -0.239 | 0.0000 | *** |
| Metacognitive | Emotions | Embodiment | 0.239 | 0.0000 | *** |
| Metacognitive | Biological | SysPrompt (labeled) | -0.223 | 0.0000 | *** |
| Metacognitive | Cognitive | Biological | 0.210 | 0.0000 | *** |
| Metacognitive | Helpfulness | Biological | 0.206 | 0.0235 | * |
| Metacognitive | Cognitive | Social | -0.206 | 0.0494 | * |
| Metacognitive | Formality | Expertise | -0.205 | 0.0328 | * |

## Methods

- **Analysis**: Residual Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
