# Exp 3: Pairwise Dimension Comparison — Raw Contrast Alignment (Control)

*Generated: 2026-03-04 11:26 | Turn: 5*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 1 | 3 | 3 |
| Emotions | Mental | 0 | 3 | 10 | 0 |
| Agency | Mental | 0 | 5 | 14 | 0 |
| Intentions | Mental | 0 | 3 | 13 | 0 |
| Prediction | Mental | 0 | 1 | 6 | 0 |
| Cognitive | Mental | 2 | 0 | 3 | 1 |
| Social | Mental | 0 | 1 | 4 | 2 |
| Attention | Mental | 1 | 0 | 3 | 3 |
| Embodiment | Physical | 0 | 1 | 3 | 2 |
| Roles | Physical | 0 | 2 | 3 | 2 |
| Animacy | Physical | 5 | 0 | 1 | 10 |
| Formality | Pragmatic | 2 | 0 | 1 | 3 |
| Expertise | Pragmatic | 11 | 0 | 2 | 4 |
| Helpfulness | Pragmatic | 0 | 0 | 1 | 3 |
| Baseline | Control | 0 | 1 | 1 | 12 |
| Biological | Control | 0 | 1 | 2 | 4 |
| Shapes | Control | 0 | 0 | 1 | 5 |
| SysPrompt (labeled) | SysPrompt | 0 | 2 | 0 | 17 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Intentions | SysPrompt (labeled) | 0.289 | 0.0000 | *** |
| Operational | Agency | SysPrompt (labeled) | 0.285 | 0.0000 | *** |
| Operational | Emotions | SysPrompt (labeled) | 0.252 | 0.0000 | *** |
| Operational | Prediction | SysPrompt (labeled) | 0.225 | 0.0000 | *** |
| Operational | Baseline | Intentions | -0.215 | 0.0000 | *** |
| Operational | Baseline | Agency | -0.210 | 0.0000 | *** |
| Operational | Social | SysPrompt (labeled) | 0.202 | 0.0000 | *** |
| Operational | Cognitive | SysPrompt (labeled) | 0.196 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (labeled) | 0.192 | 0.0000 | *** |
| Operational | Attention | SysPrompt (labeled) | 0.188 | 0.0000 | *** |
| Operational | Intentions | Animacy | 0.185 | 0.0000 | *** |
| Operational | Intentions | Shapes | 0.182 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (labeled) | 0.182 | 0.0000 | *** |
| Operational | Agency | Animacy | 0.180 | 0.0000 | *** |
| Operational | Agency | Shapes | 0.178 | 0.0000 | *** |
| Operational | Baseline | Emotions | -0.177 | 0.0000 | *** |
| Operational | Roles | SysPrompt (labeled) | 0.177 | 0.0000 | *** |
| Operational | Intentions | Helpfulness | 0.160 | 0.0061 | ** |
| Operational | Agency | Helpfulness | 0.156 | 0.0000 | *** |
| Operational | Intentions | Expertise | 0.153 | 0.0000 | *** |

## Methods

- **Analysis**: Raw Contrast Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
