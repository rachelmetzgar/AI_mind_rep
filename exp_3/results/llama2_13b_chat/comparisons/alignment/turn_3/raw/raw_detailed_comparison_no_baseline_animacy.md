# Exp 3: Raw Alignment (No Baseline) — Partner Identity vs Control

*Generated: 2026-03-05 00:23*

## Summary

| Version | Metacognitive R² (Mental, ×10⁻³) | Operational R² (Mental, ×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.357 | 5.366 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.316 | 1.039 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension Data (Metacognitive Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Biological | Control | 0.245 | 0.293 |
| Shapes | Control | 0.464 | 0.176 |
| Social | Mental | 0.503 | 0.298 |
| Attention | Mental | 0.496 | 0.347 |
| Prediction | Mental | 0.311 | 0.337 |
| Cognitive | Mental | 0.435 | 0.355 |
| Agency | Mental | 0.294 | 0.272 |
| Intentions | Mental | 0.437 | 0.394 |
| Roles | Mental | 0.273 | 0.304 |
| Emotions | Mental | 0.302 | 0.232 |
| Phenomenology | Mental | 0.246 | 0.302 |
| Embodiment | Mental | 0.278 | 0.317 |
| Expertise | Pragmatic | 0.483 | 0.245 |
| Formality | Pragmatic | 0.457 | 0.272 |
| Helpfulness | Pragmatic | 0.399 | 0.237 |
| SysPrompt | SysPrompt | 0.629 | 0.238 |

## Per-Dimension Data (Operational Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Biological | Control | 2.494 | 0.814 |
| Shapes | Control | 0.517 | 0.345 |
| Social | Mental | 6.591 | 0.743 |
| Attention | Mental | 6.424 | 1.022 |
| Prediction | Mental | 6.145 | 0.959 |
| Cognitive | Mental | 5.530 | 1.522 |
| Agency | Mental | 5.456 | 1.195 |
| Intentions | Mental | 5.194 | 1.460 |
| Roles | Mental | 5.204 | 0.741 |
| Emotions | Mental | 4.735 | 0.683 |
| Phenomenology | Mental | 4.273 | 0.986 |
| Embodiment | Mental | 4.108 | 1.073 |
| Expertise | Pragmatic | 2.351 | 0.941 |
| Formality | Pragmatic | 1.280 | 0.507 |
| Helpfulness | Pragmatic | 0.570 | 0.582 |
| SysPrompt | SysPrompt | 4.982 | 0.457 |

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
