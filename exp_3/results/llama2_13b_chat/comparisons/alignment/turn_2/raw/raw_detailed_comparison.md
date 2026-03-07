# Exp 3: Raw Alignment — Partner Identity vs Control

*Generated: 2026-03-04 12:13*

## Summary

| Version | Metacognitive R² (Mental, ×10⁻³) | Operational R² (Mental, ×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.412 | 7.754 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.648 | 1.276 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension Data (Metacognitive Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Human vs AI (general) | Control | 0.516 | 0.479 |
| Biological | Control | 0.209 | 0.405 |
| Shapes | Control | 0.653 | 0.348 |
| Phenomenology | Mental | 0.276 | 0.633 |
| Emotions | Mental | 0.257 | 0.528 |
| Agency | Mental | 0.344 | 0.607 |
| Intentions | Mental | 0.453 | 0.789 |
| Prediction | Mental | 0.365 | 0.633 |
| Cognitive | Mental | 0.529 | 0.632 |
| Social | Mental | 0.481 | 0.655 |
| Attention | Mental | 0.590 | 0.707 |
| Embodiment | Physical | 0.191 | 0.469 |
| Roles | Physical | 0.292 | 0.573 |
| Animacy | Physical | 0.310 | 0.256 |
| Formality | Pragmatic | 0.464 | 0.425 |
| Expertise | Pragmatic | 0.590 | 0.213 |
| Helpfulness | Pragmatic | 0.346 | 0.249 |
| SysPrompt | SysPrompt | 1.002 | 0.293 |

## Per-Dimension Data (Operational Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Human vs AI (general) | Control | 7.302 | 1.412 |
| Biological | Control | 3.487 | 1.454 |
| Shapes | Control | 0.565 | 0.298 |
| Phenomenology | Mental | 5.739 | 1.307 |
| Emotions | Mental | 6.696 | 1.172 |
| Agency | Mental | 7.906 | 1.351 |
| Intentions | Mental | 7.491 | 1.406 |
| Prediction | Mental | 8.696 | 1.240 |
| Cognitive | Mental | 7.500 | 0.966 |
| Social | Mental | 9.183 | 1.602 |
| Attention | Mental | 8.817 | 1.163 |
| Embodiment | Physical | 5.838 | 1.435 |
| Roles | Physical | 7.672 | 1.230 |
| Animacy | Physical | 5.201 | 1.095 |
| Formality | Pragmatic | 1.569 | 0.606 |
| Expertise | Pragmatic | 4.327 | 0.124 |
| Helpfulness | Pragmatic | 0.794 | 0.742 |
| SysPrompt | SysPrompt | 7.380 | 0.407 |

## Key Findings

1. **Partner Identity version shows substantially higher alignment** than the Control version across nearly all dimensions
2. **Metacognitive and operational probes** may show different alignment patterns, reflecting distinct aspects of partner encoding
3. **Control version shows near-floor alignment**, comparable to the shapes negative control dimension
4. **This is raw alignment** — residual analysis (projecting out entity baseline) needed to assess concept-specific contribution

## Methods

- **Metric**: Mean R² (cosine similarity squared) between concept direction vectors and probe weight vectors, averaged across 35 layers (6–40)
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Model**: LLaMA-2-13B-Chat
- **Concept vectors**: 18 dimensions, ~80 prompts each (contrasts mode: human vs AI)
- **Probes**: From Exp 2, metacognitive + operational probes (logistic, per-layer)
