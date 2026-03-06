# Exp 3: Raw Alignment (No Baseline) — Partner Identity vs Control

*Generated: 2026-03-05 00:23*

## Summary

| Version | Metacognitive R² (Mental, ×10⁻³) | Operational R² (Mental, ×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.403 | 4.615 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.208 | 0.329 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension Data (Metacognitive Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Biological | Control | 0.192 | 0.253 |
| Shapes | Control | 0.638 | 0.236 |
| Attention | Mental | 0.677 | 0.238 |
| Social | Mental | 0.486 | 0.211 |
| Cognitive | Mental | 0.784 | 0.222 |
| Prediction | Mental | 0.388 | 0.227 |
| Intentions | Mental | 0.496 | 0.194 |
| Agency | Mental | 0.292 | 0.177 |
| Roles | Mental | 0.223 | 0.194 |
| Emotions | Mental | 0.239 | 0.151 |
| Phenomenology | Mental | 0.196 | 0.197 |
| Embodiment | Mental | 0.174 | 0.238 |
| Animacy | Mental | 0.477 | 0.244 |
| Expertise | Pragmatic | 0.559 | 0.210 |
| Formality | Pragmatic | 0.372 | 0.270 |
| Helpfulness | Pragmatic | 0.674 | 0.270 |
| SysPrompt | SysPrompt | 0.895 | 0.189 |

## Per-Dimension Data (Operational Probe R² ×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Biological | Control | 2.034 | 0.242 |
| Shapes | Control | 0.575 | 0.229 |
| Attention | Mental | 6.217 | 0.304 |
| Social | Mental | 5.611 | 0.354 |
| Cognitive | Mental | 5.270 | 0.337 |
| Prediction | Mental | 5.490 | 0.354 |
| Intentions | Mental | 4.765 | 0.454 |
| Agency | Mental | 4.817 | 0.425 |
| Roles | Mental | 4.553 | 0.264 |
| Emotions | Mental | 4.112 | 0.289 |
| Phenomenology | Mental | 3.752 | 0.330 |
| Embodiment | Mental | 3.270 | 0.338 |
| Animacy | Mental | 2.908 | 0.174 |
| Expertise | Pragmatic | 1.801 | 0.256 |
| Formality | Pragmatic | 0.968 | 0.207 |
| Helpfulness | Pragmatic | 0.616 | 0.175 |
| SysPrompt | SysPrompt | 5.185 | 0.170 |

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
