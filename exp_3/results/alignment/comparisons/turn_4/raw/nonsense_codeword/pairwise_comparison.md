# Exp 3: Pairwise Dimension Comparison — Raw Contrast Alignment (Control)

*Generated: 2026-03-04 11:39 | Turn: 4*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 1 | 0 | 3 | 1 |
| Emotions | Mental | 0 | 12 | 3 | 2 |
| Agency | Mental | 0 | 1 | 9 | 0 |
| Intentions | Mental | 0 | 1 | 12 | 0 |
| Prediction | Mental | 1 | 0 | 3 | 0 |
| Cognitive | Mental | 1 | 0 | 3 | 0 |
| Social | Mental | 1 | 0 | 3 | 0 |
| Attention | Mental | 1 | 0 | 3 | 1 |
| Embodiment | Physical | 1 | 0 | 3 | 0 |
| Roles | Physical | 0 | 1 | 1 | 1 |
| Animacy | Physical | 2 | 0 | 0 | 10 |
| Formality | Pragmatic | 1 | 0 | 0 | 2 |
| Expertise | Pragmatic | 1 | 0 | 2 | 2 |
| Helpfulness | Pragmatic | 1 | 0 | 0 | 2 |
| Baseline | Control | 1 | 0 | 0 | 9 |
| Biological | Control | 5 | 0 | 1 | 2 |
| Shapes | Control | 0 | 0 | 0 | 2 |
| SysPrompt (labeled) | SysPrompt | 0 | 2 | 0 | 12 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Intentions | SysPrompt (labeled) | 0.268 | 0.0000 | *** |
| Operational | Intentions | Animacy | 0.262 | 0.0000 | *** |
| Operational | Baseline | Intentions | -0.253 | 0.0000 | *** |
| Operational | Agency | SysPrompt (labeled) | 0.239 | 0.0000 | *** |
| Operational | Intentions | Helpfulness | 0.235 | 0.0087 | ** |
| Operational | Agency | Animacy | 0.233 | 0.0000 | *** |
| Operational | Intentions | Formality | 0.232 | 0.0000 | *** |
| Operational | Baseline | Agency | -0.224 | 0.0000 | *** |
| Operational | Intentions | Shapes | 0.209 | 0.0000 | *** |
| Operational | Agency | Helpfulness | 0.206 | 0.0000 | *** |
| Operational | Agency | Formality | 0.203 | 0.0087 | ** |
| Operational | Intentions | Biological | 0.198 | 0.0000 | *** |
| Operational | Intentions | Expertise | 0.183 | 0.0000 | *** |
| Operational | Agency | Shapes | 0.180 | 0.0224 | * |
| Operational | Prediction | SysPrompt (labeled) | 0.175 | 0.0000 | *** |
| Operational | Intentions | Roles | 0.172 | 0.0278 | * |
| Operational | Social | SysPrompt (labeled) | 0.171 | 0.0000 | *** |
| Operational | Agency | Biological | 0.169 | 0.0087 | ** |
| Operational | Prediction | Animacy | 0.169 | 0.0000 | *** |
| Operational | Social | Animacy | 0.165 | 0.0000 | *** |

## Methods

- **Analysis**: Raw Contrast Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
