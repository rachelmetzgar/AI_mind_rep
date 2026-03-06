# Exp 3: Pairwise Dimension Comparison — Residual Alignment (Control)

*Generated: 2026-03-04 11:39 | Turn: 4*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 0 | 0 | 1 |
| Emotions | Mental | 0 | 4 | 1 | 2 |
| Agency | Mental | 0 | 0 | 4 | 0 |
| Intentions | Mental | 0 | 0 | 9 | 0 |
| Prediction | Mental | 1 | 0 | 1 | 0 |
| Cognitive | Mental | 0 | 0 | 1 | 0 |
| Social | Mental | 1 | 0 | 1 | 0 |
| Attention | Mental | 1 | 0 | 1 | 1 |
| Embodiment | Physical | 0 | 0 | 1 | 0 |
| Roles | Physical | 0 | 0 | 0 | 0 |
| Animacy | Physical | 0 | 0 | 0 | 2 |
| Formality | Pragmatic | 1 | 0 | 0 | 1 |
| Expertise | Pragmatic | 0 | 0 | 1 | 1 |
| Helpfulness | Pragmatic | 0 | 0 | 0 | 1 |
| Baseline | Control | 0 | 0 | 0 | 0 |
| Biological | Control | 0 | 0 | 0 | 2 |
| Shapes | Control | 0 | 0 | 0 | 0 |
| SysPrompt (labeled) | SysPrompt | 0 | 0 | 0 | 9 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Agency | SysPrompt (labeled) | 0.275 | 0.0000 | *** |
| Operational | Intentions | SysPrompt (labeled) | 0.261 | 0.0000 | *** |
| Operational | Agency | Animacy | 0.217 | 0.0000 | *** |
| Operational | Agency | Biological | 0.213 | 0.0000 | *** |
| Operational | Intentions | Animacy | 0.204 | 0.0000 | *** |
| Operational | Intentions | Biological | 0.200 | 0.0000 | *** |
| Operational | Intentions | Helpfulness | 0.191 | 0.0459 | * |
| Operational | Social | SysPrompt (labeled) | 0.183 | 0.0000 | *** |
| Operational | Intentions | Formality | 0.183 | 0.0459 | * |
| Operational | Cognitive | SysPrompt (labeled) | 0.181 | 0.0000 | *** |
| Operational | Emotions | Agency | -0.178 | 0.0000 | *** |
| Operational | Prediction | SysPrompt (labeled) | 0.175 | 0.0000 | *** |
| Operational | Emotions | Intentions | -0.164 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (labeled) | 0.154 | 0.0000 | *** |
| Operational | Phenomenology | Intentions | -0.143 | 0.0000 | *** |
| Operational | Attention | SysPrompt (labeled) | 0.134 | 0.0000 | *** |
| Operational | Intentions | Expertise | 0.131 | 0.0360 | * |
| Operational | Expertise | SysPrompt (labeled) | 0.130 | 0.0000 | *** |
| Operational | Intentions | Attention | 0.127 | 0.0459 | * |
| Metacognitive | Emotions | Formality | -0.110 | 0.0000 | *** |

## Methods

- **Analysis**: Residual Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
