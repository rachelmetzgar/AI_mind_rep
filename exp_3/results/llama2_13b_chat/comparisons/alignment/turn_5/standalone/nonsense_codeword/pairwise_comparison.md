# Exp 3: Pairwise Dimension Comparison — Standalone Concept Alignment (Control)

*Generated: 2026-03-08 14:56 | Turn: 5*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 17 | 0 | 2 | 6 |
| Emotions | Mental | 19 | 0 | 2 | 11 |
| Agency | Mental | 4 | 6 | 4 | 5 |
| Intentions | Mental | 5 | 5 | 8 | 4 |
| Prediction | Mental | 5 | 5 | 1 | 11 |
| Cognitive | Mental | 0 | 15 | 2 | 6 |
| Social | Mental | 1 | 16 | 2 | 6 |
| Attention | Mental | 8 | 5 | 4 | 4 |
| Embodiment | Physical | 5 | 5 | 17 | 2 |
| Roles | Physical | 1 | 16 | 5 | 4 |
| Animacy | Physical | 5 | 5 | 7 | 4 |
| Formality | Pragmatic | 16 | 1 | 2 | 8 |
| Expertise | Pragmatic | 2 | 6 | 2 | 8 |
| Helpfulness | Pragmatic | 5 | 5 | 2 | 7 |
| Biological | Control | 5 | 5 | 19 | 1 |
| Shapes | Control | 17 | 0 | 11 | 4 |
| Human (concept) | Entity | 5 | 6 | 0 | 19 |
| AI (concept) | Entity | 5 | 5 | 0 | 20 |
| SysPrompt (talkto human) | SysPrompt | 17 | 1 | 12 | 3 |
| SysPrompt (talkto AI) | SysPrompt | 5 | 4 | 2 | 9 |
| SysPrompt (bare human) | SysPrompt | 1 | 17 | 21 | 0 |
| SysPrompt (bare AI) | SysPrompt | 0 | 20 | 18 | 1 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Human (concept) | SysPrompt (bare human) | -0.047 | 0.0000 | *** |
| Operational | AI (concept) | SysPrompt (bare human) | -0.047 | 0.0000 | *** |
| Operational | Prediction | SysPrompt (bare human) | -0.042 | 0.0000 | *** |
| Operational | Emotions | SysPrompt (bare human) | -0.041 | 0.0000 | *** |
| Operational | SysPrompt (talkto AI) | SysPrompt (bare human) | -0.040 | 0.0000 | *** |
| Operational | Formality | SysPrompt (bare human) | -0.040 | 0.0000 | *** |
| Operational | Expertise | SysPrompt (bare human) | -0.040 | 0.0000 | *** |
| Operational | Helpfulness | SysPrompt (bare human) | -0.039 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (bare human) | -0.037 | 0.0000 | *** |
| Operational | Social | SysPrompt (bare human) | -0.037 | 0.0000 | *** |
| Operational | Cognitive | SysPrompt (bare human) | -0.037 | 0.0000 | *** |
| Operational | Agency | SysPrompt (bare human) | -0.036 | 0.0000 | *** |
| Operational | Roles | SysPrompt (bare human) | -0.035 | 0.0000 | *** |
| Operational | Attention | SysPrompt (bare human) | -0.035 | 0.0000 | *** |
| Metacognitive | Emotions | SysPrompt (bare AI) | 0.035 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (bare human) | -0.033 | 0.0000 | *** |
| Operational | Intentions | SysPrompt (bare human) | -0.033 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (bare human) | -0.032 | 0.0000 | *** |
| Operational | SysPrompt (talkto human) | SysPrompt (bare human) | -0.031 | 0.0000 | *** |
| Metacognitive | Shapes | SysPrompt (bare AI) | 0.031 | 0.0000 | *** |

## Methods

- **Analysis**: Standalone Concept Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
