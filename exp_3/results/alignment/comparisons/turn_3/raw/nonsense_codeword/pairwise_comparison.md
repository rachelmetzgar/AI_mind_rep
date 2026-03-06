# Exp 3: Pairwise Dimension Comparison — Raw Contrast Alignment (Control)

*Generated: 2026-03-04 11:38 | Turn: 3*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 0 | 0 | 2 | 0 |
| Emotions | Mental | 0 | 5 | 2 | 4 |
| Agency | Mental | 0 | 0 | 4 | 0 |
| Intentions | Mental | 6 | 0 | 11 | 0 |
| Prediction | Mental | 5 | 0 | 2 | 0 |
| Cognitive | Mental | 5 | 0 | 11 | 0 |
| Social | Mental | 0 | 0 | 2 | 2 |
| Attention | Mental | 5 | 0 | 2 | 0 |
| Embodiment | Physical | 1 | 0 | 3 | 0 |
| Roles | Physical | 0 | 0 | 2 | 2 |
| Animacy | Physical | 0 | 6 | 0 | 12 |
| Formality | Pragmatic | 0 | 0 | 0 | 2 |
| Expertise | Pragmatic | 0 | 5 | 2 | 2 |
| Helpfulness | Pragmatic | 0 | 1 | 0 | 2 |
| Baseline | Control | 3 | 0 | 0 | 2 |
| Biological | Control | 0 | 0 | 2 | 2 |
| Shapes | Control | 0 | 4 | 0 | 3 |
| SysPrompt (labeled) | SysPrompt | 0 | 4 | 0 | 12 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Cognitive | Shapes | 1.039 | 0.0000 | *** |
| Operational | Intentions | Shapes | 1.011 | 0.0000 | *** |
| Operational | Cognitive | SysPrompt (labeled) | 0.978 | 0.0000 | *** |
| Operational | Cognitive | Animacy | 0.978 | 0.0000 | *** |
| Operational | Cognitive | Formality | 0.952 | 0.0000 | *** |
| Operational | Intentions | SysPrompt (labeled) | 0.950 | 0.0000 | *** |
| Operational | Intentions | Animacy | 0.949 | 0.0000 | *** |
| Operational | Intentions | Formality | 0.924 | 0.0000 | *** |
| Operational | Cognitive | Helpfulness | 0.854 | 0.0000 | *** |
| Operational | Intentions | Helpfulness | 0.826 | 0.0106 | * |
| Operational | Agency | Shapes | 0.774 | 0.0106 | * |
| Operational | Emotions | Cognitive | -0.761 | 0.0000 | *** |
| Operational | Baseline | Cognitive | -0.751 | 0.0191 | * |
| Operational | Emotions | Intentions | -0.733 | 0.0000 | *** |
| Operational | Baseline | Intentions | -0.723 | 0.0106 | * |
| Operational | Agency | SysPrompt (labeled) | 0.713 | 0.0000 | *** |
| Operational | Agency | Animacy | 0.713 | 0.0000 | *** |
| Operational | Cognitive | Social | 0.687 | 0.0000 | *** |
| Operational | Cognitive | Roles | 0.684 | 0.0262 | * |
| Operational | Intentions | Social | 0.659 | 0.0408 | * |

## Methods

- **Analysis**: Raw Contrast Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
