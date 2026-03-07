# Exp 3: Pairwise Dimension Comparison — Standalone Concept Alignment (Control)

*Generated: 2026-03-04 11:39 | Turn: 4*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 5 | 8 | 0 | 13 |
| Emotions | Mental | 12 | 2 | 0 | 13 |
| Agency | Mental | 8 | 2 | 9 | 7 |
| Intentions | Mental | 8 | 3 | 9 | 7 |
| Prediction | Mental | 5 | 6 | 0 | 13 |
| Cognitive | Mental | 5 | 10 | 0 | 13 |
| Social | Mental | 4 | 11 | 0 | 13 |
| Attention | Mental | 5 | 5 | 1 | 12 |
| Embodiment | Physical | 13 | 2 | 0 | 14 |
| Roles | Physical | 5 | 10 | 9 | 7 |
| Animacy | Physical | 9 | 2 | 8 | 7 |
| Formality | Pragmatic | 0 | 17 | 0 | 13 |
| Expertise | Pragmatic | 0 | 17 | 9 | 7 |
| Helpfulness | Pragmatic | 0 | 17 | 9 | 7 |
| Biological | Control | 9 | 2 | 15 | 5 |
| Shapes | Control | 6 | 5 | 0 | 13 |
| Human (concept) | Entity | 1 | 16 | 17 | 2 |
| AI (concept) | Entity | 0 | 18 | 15 | 5 |
| SysPrompt (talkto human) | SysPrompt | 21 | 0 | 18 | 2 |
| SysPrompt (talkto AI) | SysPrompt | 13 | 1 | 17 | 3 |
| SysPrompt (bare human) | SysPrompt | 19 | 1 | 21 | 0 |
| SysPrompt (bare AI) | SysPrompt | 10 | 3 | 20 | 1 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Embodiment | SysPrompt (bare human) | -0.081 | 0.0000 | *** |
| Operational | Prediction | SysPrompt (bare human) | -0.079 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (bare human) | -0.078 | 0.0000 | *** |
| Operational | Cognitive | SysPrompt (bare human) | -0.078 | 0.0000 | *** |
| Operational | Social | SysPrompt (bare human) | -0.077 | 0.0000 | *** |
| Operational | Formality | SysPrompt (bare human) | -0.077 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (bare human) | -0.077 | 0.0000 | *** |
| Operational | Emotions | SysPrompt (bare human) | -0.076 | 0.0000 | *** |
| Operational | Attention | SysPrompt (bare human) | -0.073 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (bare AI) | -0.071 | 0.0000 | *** |
| Operational | Prediction | SysPrompt (bare AI) | -0.069 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (bare human) | -0.069 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (bare AI) | -0.069 | 0.0000 | *** |
| Operational | Cognitive | SysPrompt (bare AI) | -0.069 | 0.0000 | *** |
| Operational | Social | SysPrompt (bare AI) | -0.068 | 0.0000 | *** |
| Operational | Formality | SysPrompt (bare AI) | -0.067 | 0.0000 | *** |
| Operational | Intentions | SysPrompt (bare human) | -0.067 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (bare AI) | -0.067 | 0.0000 | *** |
| Operational | Roles | SysPrompt (bare human) | -0.067 | 0.0000 | *** |
| Operational | Emotions | SysPrompt (bare AI) | -0.066 | 0.0000 | *** |

## Methods

- **Analysis**: Standalone Concept Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
