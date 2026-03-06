# Exp 3: Pairwise Dimension Comparison — Standalone Concept Alignment (Control)

*Generated: 2026-03-04 11:38 | Turn: 3*

## Summary

| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |
|---|---|---|---|---|---|
| Phenomenology | Mental | 7 | 3 | 2 | 13 |
| Emotions | Mental | 15 | 1 | 6 | 9 |
| Agency | Mental | 14 | 2 | 4 | 10 |
| Intentions | Mental | 7 | 2 | 1 | 12 |
| Prediction | Mental | 12 | 2 | 1 | 11 |
| Cognitive | Mental | 5 | 10 | 7 | 8 |
| Social | Mental | 5 | 6 | 13 | 4 |
| Attention | Mental | 7 | 4 | 0 | 16 |
| Embodiment | Physical | 12 | 2 | 0 | 15 |
| Roles | Physical | 5 | 6 | 5 | 9 |
| Animacy | Physical | 6 | 4 | 1 | 15 |
| Formality | Pragmatic | 0 | 17 | 11 | 4 |
| Expertise | Pragmatic | 0 | 18 | 13 | 4 |
| Helpfulness | Pragmatic | 0 | 18 | 13 | 4 |
| Biological | Control | 4 | 13 | 4 | 10 |
| Shapes | Control | 7 | 6 | 0 | 19 |
| Human (concept) | Entity | 0 | 18 | 13 | 4 |
| AI (concept) | Entity | 3 | 16 | 9 | 8 |
| SysPrompt (talkto human) | SysPrompt | 6 | 6 | 21 | 0 |
| SysPrompt (talkto AI) | SysPrompt | 6 | 6 | 20 | 1 |
| SysPrompt (bare human) | SysPrompt | 21 | 0 | 18 | 2 |
| SysPrompt (bare AI) | SysPrompt | 19 | 1 | 18 | 2 |

## Top Significant Pairs

| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |
|---|---|---|---|---|---|
| Operational | Shapes | SysPrompt (talkto human) | -0.240 | 0.0000 | *** |
| Operational | Attention | SysPrompt (talkto human) | -0.236 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (talkto human) | -0.235 | 0.0000 | *** |
| Operational | Animacy | SysPrompt (talkto human) | -0.231 | 0.0000 | *** |
| Operational | Phenomenology | SysPrompt (talkto human) | -0.229 | 0.0000 | *** |
| Operational | Prediction | SysPrompt (talkto human) | -0.228 | 0.0000 | *** |
| Operational | Intentions | SysPrompt (talkto human) | -0.227 | 0.0000 | *** |
| Operational | Biological | SysPrompt (talkto human) | -0.224 | 0.0000 | *** |
| Operational | Agency | SysPrompt (talkto human) | -0.218 | 0.0000 | *** |
| Operational | Roles | SysPrompt (talkto human) | -0.216 | 0.0000 | *** |
| Operational | Emotions | SysPrompt (talkto human) | -0.215 | 0.0000 | *** |
| Operational | Cognitive | SysPrompt (talkto human) | -0.206 | 0.0000 | *** |
| Operational | AI (concept) | SysPrompt (talkto human) | -0.202 | 0.0000 | *** |
| Operational | Formality | SysPrompt (talkto human) | -0.188 | 0.0000 | *** |
| Operational | Shapes | SysPrompt (talkto AI) | -0.172 | 0.0000 | *** |
| Operational | Human (concept) | SysPrompt (talkto human) | -0.169 | 0.0000 | *** |
| Operational | Helpfulness | SysPrompt (talkto human) | -0.169 | 0.0000 | *** |
| Operational | Attention | SysPrompt (talkto AI) | -0.168 | 0.0000 | *** |
| Operational | Embodiment | SysPrompt (talkto AI) | -0.166 | 0.0000 | *** |
| Operational | Expertise | SysPrompt (talkto human) | -0.166 | 0.0000 | *** |

## Methods

- **Analysis**: Standalone Concept Alignment, pairwise bootstrap comparison
- **FDR**: Benjamini-Hochberg, q = 0.05
- **Bootstrap**: 1,000 paired iterations
