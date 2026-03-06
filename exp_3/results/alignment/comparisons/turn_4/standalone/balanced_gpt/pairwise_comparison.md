# Exp 3: Pairwise Dimension Comparison — Standalone Concept Alignment (Partner Identity)

*Generated: 2026-03-04 11:39 | Turn: 4*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 16 | 0 | 12 | 5 |
| Emotions | Mental | 15 | 0 | 8 | 8 |
| Agency | Mental | 9 | 1 | 6 | 8 |
| Intentions | Mental | 9 | 6 | 6 | 10 |
| Prediction | Mental | 9 | 5 | 5 | 11 |
| Cognitive | Mental | 3 | 12 | 6 | 9 |
| Social | Mental | 2 | 16 | 2 | 17 |
| Attention | Mental | 9 | 6 | 9 | 6 |
| Embodiment | Physical | 15 | 0 | 14 | 1 |
| Roles | Physical | 7 | 6 | 7 | 8 |
| Animacy | Physical | 15 | 0 | 15 | 1 |
| Formality | Pragmatic | 4 | 12 | 2 | 17 |
| Expertise | Pragmatic | 2 | 13 | 5 | 15 |
| Helpfulness | Pragmatic | 3 | 13 | 2 | 17 |
| Biological | Control | 9 | 5 | 15 | 1 |
| Shapes | Control | 9 | 6 | 15 | 1 |
| Human (concept) | Entity | 15 | 0 | 13 | 4 |
| AI (concept) | Entity | 13 | 0 | 20 | 0 |
| SysPrompt (talkto human) | SysPrompt | 0 | 21 | 0 | 21 |
| SysPrompt (talkto AI) | SysPrompt | 2 | 15 | 6 | 2 |
| SysPrompt (bare human) | SysPrompt | 1 | 20 | 1 | 20 |
| SysPrompt (bare AI) | SysPrompt | 3 | 13 | 13 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | AI (concept) | SysPrompt (talkto human) | 1.588 | 0.0000 | *** |
| Operational | AI (concept) | SysPrompt (bare human) | 1.358 | 0.0000 | *** |
| Operational | SysPrompt (talkto human) | SysPrompt (bare AI) | -1.226 | 0.0000 | *** |
| Operational | Social | AI (concept) | -1.177 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (talkto human) | 1.159 | 0.0000 | *** |
| Operational | Biological | SysPrompt (talkto human) | 1.152 | 0.0000 | *** |
| Operational | Formality | AI (concept) | -1.138 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (talkto human) | 1.118 | 0.0000 | *** |
| Operational | Helpfulness | AI (concept) | -1.117 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (talkto human) | 1.074 | 0.0000 | *** |
| Operational | SysPrompt (bare human) | SysPrompt (bare AI) | -0.996 | 0.0000 | *** |
| Operational | SysPrompt (talkto human) | SysPrompt (talkto AI) | -0.988 | 0.0000 | *** |
| Operational | Human (concept) | SysPrompt (talkto human) | 0.979 | 0.0000 | *** |
| Operational | Expertise | AI (concept) | -0.976 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (talkto human) | 0.954 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (bare human) | 0.930 | 0.0000 | *** |
| Operational | Biological | SysPrompt (bare human) | 0.922 | 0.0000 | *** |
| Operational | Prediction | AI (concept) | -0.893 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (bare human) | 0.888 | 0.0000 | *** |
| Operational | Attention | SysPrompt (talkto human) | 0.888 | 0.0000 | *** |

## Methods

- **Analysis**: Standalone Concept Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
