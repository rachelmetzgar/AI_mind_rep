# Exp 3: Raw Alignment (No Baseline) — Partner Identity vs Control

*Generated: 2026-03-05 00:23*

## Summary

| Version | Metacognitive R² (Mental, ×10⁻³) | Operational R² (Mental, ×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.378 | 7.554 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.622 | 1.287 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension Data (Metacognitive Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Biological | Control | 0.209 | 0.405 |
| Shapes | Control | 0.653 | 0.348 |
| Social | Mental | 0.481 | 0.655 |
| Attention | Mental | 0.590 | 0.707 |
| Prediction | Mental | 0.365 | 0.633 |
| Agency | Mental | 0.344 | 0.607 |
| Cognitive | Mental | 0.529 | 0.632 |
| Roles | Mental | 0.292 | 0.573 |
| Intentions | Mental | 0.453 | 0.789 |
| Emotions | Mental | 0.257 | 0.528 |
| Embodiment | Mental | 0.191 | 0.469 |
| Phenomenology | Mental | 0.276 | 0.633 |
| Expertise | Pragmatic | 0.590 | 0.213 |
| Formality | Pragmatic | 0.464 | 0.425 |
| Helpfulness | Pragmatic | 0.346 | 0.249 |
| SysPrompt | SysPrompt | 1.002 | 0.293 |

## Per-Dimension Data (Operational Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Biological | Control | 3.487 | 1.454 |
| Shapes | Control | 0.565 | 0.298 |
| Social | Mental | 9.183 | 1.602 |
| Attention | Mental | 8.817 | 1.163 |
| Prediction | Mental | 8.696 | 1.240 |
| Agency | Mental | 7.906 | 1.351 |
| Cognitive | Mental | 7.500 | 0.966 |
| Roles | Mental | 7.672 | 1.230 |
| Intentions | Mental | 7.491 | 1.406 |
| Emotions | Mental | 6.696 | 1.172 |
| Embodiment | Mental | 5.838 | 1.435 |
| Phenomenology | Mental | 5.739 | 1.307 |
| Expertise | Pragmatic | 4.327 | 0.124 |
| Formality | Pragmatic | 1.569 | 0.606 |
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
