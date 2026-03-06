# Exp 3: Pairwise Dimension Comparison — Residual Alignment (Control)

*Generated: 2026-03-04 11:27 | Turn: 5*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 0 | 1 | 2 |
| Emotions | Mental | 0 | 2 | 6 | 0 |
| Agency | Mental | 0 | 2 | 12 | 0 |
| Intentions | Mental | 0 | 2 | 8 | 0 |
| Prediction | Mental | 0 | 0 | 2 | 0 |
| Cognitive | Mental | 0 | 0 | 1 | 1 |
| Social | Mental | 0 | 0 | 1 | 0 |
| Attention | Mental | 4 | 0 | 1 | 3 |
| Embodiment | Physical | 0 | 0 | 1 | 1 |
| Roles | Physical | 0 | 0 | 1 | 1 |
| Animacy | Physical | 0 | 0 | 1 | 2 |
| Formality | Pragmatic | 0 | 0 | 1 | 3 |
| Expertise | Pragmatic | 5 | 0 | 1 | 2 |
| Helpfulness | Pragmatic | 0 | 0 | 1 | 2 |
| Baseline | Control | 0 | 0 | 1 | 0 |
| Biological | Control | 0 | 2 | 1 | 4 |
| Shapes | Control | 0 | 0 | 1 | 3 |
| SysPrompt (labeled) | SysPrompt | 0 | 1 | 0 | 17 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Agency | SysPrompt (labeled) | 0.309 | 0.0000 | *** |
| Operational | Intentions | SysPrompt (labeled) | 0.272 | 0.0000 | *** |
| Operational | Emotions | SysPrompt (labeled) | 0.249 | 0.0000 | *** |
| Operational | Prediction | SysPrompt (labeled) | 0.231 | 0.0000 | *** |
| Operational | Roles | SysPrompt (labeled) | 0.207 | 0.0000 | *** |
| Operational | Social | SysPrompt (labeled) | 0.199 | 0.0000 | *** |
| Operational | Agency | Shapes | 0.194 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (labeled) | 0.187 | 0.0000 | *** |
| Operational | Cognitive | SysPrompt (labeled) | 0.178 | 0.0000 | *** |
| Operational | Attention | SysPrompt (labeled) | 0.173 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (labeled) | 0.171 | 0.0000 | *** |
| Operational | Agency | Formality | 0.171 | 0.0000 | *** |
| Operational | Agency | Helpfulness | 0.164 | 0.0185 | * |
| Operational | Agency | Expertise | 0.164 | 0.0000 | *** |
| Operational | Agency | Biological | 0.160 | 0.0000 | *** |
| Operational | Intentions | Shapes | 0.157 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (labeled) | 0.156 | 0.0000 | *** |
| Operational | Agency | Animacy | 0.152 | 0.0000 | *** |
| Operational | Biological | SysPrompt (labeled) | 0.149 | 0.0000 | *** |
| Operational | Expertise | SysPrompt (labeled) | 0.145 | 0.0000 | *** |

## Methods

- **Analysis**: Residual Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
