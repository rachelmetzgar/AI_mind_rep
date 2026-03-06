# Exp 3: Raw Alignment (No Baseline) — Partner Identity vs Control

*Generated: 2026-03-05 00:24*

## Summary

| Version | Metacognitive R² (Mental, ×10⁻³) | Operational R² (Mental, ×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.619 | 2.685 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.189 | 0.306 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension Data (Metacognitive Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Biological | Control | 0.327 | 0.207 |
| Shapes | Control | 0.304 | 0.206 |
| Attention | Mental | 0.908 | 0.228 |
| Social | Mental | 0.709 | 0.168 |
| Cognitive | Mental | 0.862 | 0.253 |
| Prediction | Mental | 0.717 | 0.178 |
| Intentions | Mental | 0.675 | 0.169 |
| Agency | Mental | 0.540 | 0.143 |
| Roles | Mental | 0.581 | 0.165 |
| Phenomenology | Mental | 0.443 | 0.212 |
| Emotions | Mental | 0.451 | 0.171 |
| Embodiment | Mental | 0.308 | 0.207 |
| Expertise | Pragmatic | 0.455 | 0.283 |
| Formality | Pragmatic | 0.314 | 0.253 |
| Helpfulness | Pragmatic | 0.297 | 0.257 |
| SysPrompt | SysPrompt | 0.893 | 0.196 |

## Per-Dimension Data (Operational Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Biological | Control | 1.039 | 0.221 |
| Shapes | Control | 0.245 | 0.190 |
| Attention | Mental | 3.540 | 0.275 |
| Social | Mental | 3.366 | 0.291 |
| Cognitive | Mental | 3.142 | 0.276 |
| Prediction | Mental | 2.987 | 0.313 |
| Intentions | Mental | 2.839 | 0.383 |
| Agency | Mental | 2.577 | 0.379 |
| Roles | Mental | 2.521 | 0.257 |
| Phenomenology | Mental | 2.108 | 0.267 |
| Emotions | Mental | 1.999 | 0.340 |
| Embodiment | Mental | 1.771 | 0.275 |
| Expertise | Pragmatic | 1.133 | 0.214 |
| Formality | Pragmatic | 0.330 | 0.237 |
| Helpfulness | Pragmatic | 0.270 | 0.203 |
| SysPrompt | SysPrompt | 2.939 | 0.076 |

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
