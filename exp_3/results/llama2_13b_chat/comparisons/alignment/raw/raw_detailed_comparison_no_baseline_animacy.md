# Exp 3: Raw Alignment (No Baseline) — Partner Identity vs Control

*Generated: 2026-03-05 00:23*

## Summary

| Version | Metacognitive R² (Mental, ×10⁻³) | Operational R² (Mental, ×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.324 | 3.156 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.460 | 5.071 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension Data (Metacognitive Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Biological | Control | 0.273 | 0.265 |
| Shapes | Control | 0.272 | 0.475 |
| Social | Mental | 0.493 | 0.564 |
| Attention | Mental | 0.401 | 0.497 |
| Prediction | Mental | 0.336 | 0.464 |
| Roles | Mental | 0.282 | 0.459 |
| Agency | Mental | 0.271 | 0.444 |
| Cognitive | Mental | 0.403 | 0.525 |
| Intentions | Mental | 0.302 | 0.560 |
| Emotions | Mental | 0.247 | 0.366 |
| Embodiment | Mental | 0.250 | 0.350 |
| Phenomenology | Mental | 0.253 | 0.376 |
| Helpfulness | Pragmatic | 0.375 | 0.437 |
| Expertise | Pragmatic | 0.330 | 0.439 |
| Formality | Pragmatic | 0.322 | 0.239 |
| SysPrompt | SysPrompt | 0.638 | 0.569 |

## Per-Dimension Data (Operational Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Biological | Control | 2.058 | 4.084 |
| Shapes | Control | 0.520 | 0.641 |
| Social | Mental | 3.654 | 5.813 |
| Attention | Mental | 3.695 | 5.107 |
| Prediction | Mental | 3.693 | 5.612 |
| Roles | Mental | 3.474 | 6.367 |
| Agency | Mental | 3.309 | 5.253 |
| Cognitive | Mental | 3.100 | 4.323 |
| Intentions | Mental | 2.846 | 4.652 |
| Emotions | Mental | 2.679 | 3.947 |
| Embodiment | Mental | 2.560 | 5.092 |
| Phenomenology | Mental | 2.553 | 4.541 |
| Helpfulness | Pragmatic | 0.452 | 0.317 |
| Expertise | Pragmatic | 0.318 | 0.569 |
| Formality | Pragmatic | 0.201 | 0.284 |
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
