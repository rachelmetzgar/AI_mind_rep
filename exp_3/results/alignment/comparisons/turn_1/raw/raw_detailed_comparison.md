# Exp 3: Raw Alignment — Partner Identity vs Control

*Generated: 2026-03-04 12:13*

## Summary

| Version | Metacognitive R² (Mental, ×10⁻³) | Operational R² (Mental, ×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.338 | 3.191 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.474 | 4.906 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension Data (Metacognitive Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Human vs AI (general) | Control | 0.319 | 0.760 |
| Biological | Control | 0.273 | 0.265 |
| Shapes | Control | 0.272 | 0.475 |
| Phenomenology | Mental | 0.253 | 0.376 |
| Emotions | Mental | 0.247 | 0.366 |
| Agency | Mental | 0.271 | 0.444 |
| Intentions | Mental | 0.302 | 0.560 |
| Prediction | Mental | 0.336 | 0.464 |
| Cognitive | Mental | 0.403 | 0.525 |
| Social | Mental | 0.493 | 0.564 |
| Attention | Mental | 0.401 | 0.497 |
| Embodiment | Physical | 0.250 | 0.350 |
| Roles | Physical | 0.282 | 0.459 |
| Animacy | Physical | 0.306 | 0.441 |
| Formality | Pragmatic | 0.322 | 0.239 |
| Expertise | Pragmatic | 0.330 | 0.439 |
| Helpfulness | Pragmatic | 0.375 | 0.437 |
| SysPrompt | SysPrompt | 0.638 | 0.569 |

## Per-Dimension Data (Operational Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Human vs AI (general) | Control | 3.993 | 9.654 |
| Biological | Control | 2.058 | 4.084 |
| Shapes | Control | 0.520 | 0.641 |
| Phenomenology | Mental | 2.553 | 4.541 |
| Emotions | Mental | 2.679 | 3.947 |
| Agency | Mental | 3.309 | 5.253 |
| Intentions | Mental | 2.846 | 4.652 |
| Prediction | Mental | 3.693 | 5.612 |
| Cognitive | Mental | 3.100 | 4.323 |
| Social | Mental | 3.654 | 5.813 |
| Attention | Mental | 3.695 | 5.107 |
| Embodiment | Physical | 2.560 | 5.092 |
| Roles | Physical | 3.474 | 6.367 |
| Animacy | Physical | 2.878 | 6.167 |
| Formality | Pragmatic | 0.201 | 0.284 |
| Expertise | Pragmatic | 0.318 | 0.569 |
| Helpfulness | Pragmatic | 0.452 | 0.317 |
| SysPrompt | SysPrompt | 5.145 | 3.795 |

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
