# Exp 3: Pairwise Dimension Comparison — Standalone Concept Alignment (Partner Identity)

*Generated: 2026-03-04 11:36 | Turn: 1*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 3 | 8 | 5 | 3 |
| Emotions | Mental | 7 | 6 | 4 | 6 |
| Agency | Mental | 8 | 4 | 3 | 6 |
| Intentions | Mental | 7 | 5 | 4 | 5 |
| Prediction | Mental | 14 | 2 | 4 | 4 |
| Cognitive | Mental | 3 | 6 | 11 | 3 |
| Social | Mental | 13 | 2 | 4 | 6 |
| Attention | Mental | 0 | 10 | 4 | 6 |
| Embodiment | Physical | 2 | 6 | 5 | 3 |
| Roles | Physical | 0 | 11 | 4 | 6 |
| Animacy | Physical | 0 | 14 | 4 | 6 |
| Formality | Pragmatic | 0 | 10 | 12 | 3 |
| Expertise | Pragmatic | 3 | 6 | 17 | 0 |
| Helpfulness | Pragmatic | 4 | 6 | 17 | 0 |
| Biological | Control | 0 | 15 | 3 | 16 |
| Shapes | Control | 0 | 15 | 3 | 8 |
| Human (concept) | Entity | 8 | 3 | 1 | 19 |
| AI (concept) | Entity | 0 | 10 | 17 | 0 |
| SysPrompt (talkto human) | SysPrompt | 20 | 1 | 0 | 21 |
| SysPrompt (talkto AI) | SysPrompt | 21 | 0 | 4 | 1 |
| SysPrompt (bare human) | SysPrompt | 15 | 2 | 1 | 19 |
| SysPrompt (bare AI) | SysPrompt | 16 | 2 | 14 | 0 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | AI (concept) | SysPrompt (talkto human) | 0.169 | 0.0000 | *** |
| Operational | SysPrompt (talkto human) | SysPrompt (bare AI) | -0.164 | 0.0000 | *** |
| Operational | Expertise | SysPrompt (talkto human) | 0.164 | 0.0000 | *** |
| Operational | Helpfulness | SysPrompt (talkto human) | 0.158 | 0.0000 | *** |
| Operational | SysPrompt (talkto human) | SysPrompt (talkto AI) | -0.149 | 0.0000 | *** |
| Operational | Formality | SysPrompt (talkto human) | 0.134 | 0.0000 | *** |
| Operational | Cognitive | SysPrompt (talkto human) | 0.133 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (talkto human) | 0.122 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (talkto human) | 0.121 | 0.0000 | *** |
| Operational | AI (concept) | SysPrompt (bare human) | 0.120 | 0.0000 | *** |
| Operational | SysPrompt (bare human) | SysPrompt (bare AI) | -0.115 | 0.0000 | *** |
| Operational | Expertise | SysPrompt (bare human) | 0.115 | 0.0000 | *** |
| Operational | Attention | SysPrompt (talkto human) | 0.114 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (talkto human) | 0.114 | 0.0000 | *** |
| Operational | Intentions | SysPrompt (talkto human) | 0.113 | 0.0000 | *** |
| Operational | Prediction | SysPrompt (talkto human) | 0.113 | 0.0000 | *** |
| Operational | Social | SysPrompt (talkto human) | 0.113 | 0.0000 | *** |
| Operational | Roles | SysPrompt (talkto human) | 0.112 | 0.0000 | *** |
| Operational | Emotions | SysPrompt (talkto human) | 0.111 | 0.0000 | *** |
| Operational | Agency | SysPrompt (talkto human) | 0.109 | 0.0000 | *** |

## Methods

- **Analysis**: Standalone Concept Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
