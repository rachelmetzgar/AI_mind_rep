# Exp 3: Residual Alignment (Entity Baseline Projected Out) — Cross-Version Comparison

*Generated: 2026-03-04 11:39*

> Same as raw alignment, but with the entity baseline direction (dim 0: 'this is a human/AI') projected out of each concept vector before computing alignment. This removes shared entity-level variance, isolating concept-specific alignment.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.732 | 1.818 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.204 | 0.332 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.451 | 0.171 |
| Emotions | Mental | 0.545 | 0.133 |
| Agency | Mental | 0.685 | 0.185 |
| Intentions | Mental | 0.823 | 0.187 |
| Prediction | Mental | 0.656 | 0.253 |
| Cognitive | Mental | 0.814 | 0.211 |
| Social | Mental | 0.928 | 0.253 |
| Attention | Mental | 0.959 | 0.240 |
| Embodiment | Physical | 0.303 | 0.183 |
| Roles | Physical | 0.410 | 0.192 |
| Animacy | Physical | 0.468 | 0.223 |
| Formality | Pragmatic | 0.454 | 0.263 |
| Expertise | Pragmatic | 0.582 | 0.206 |
| Helpfulness | Pragmatic | 0.622 | 0.256 |
| Baseline | Control | 0.283 | 0.210 |
| Biological | Control | 0.251 | 0.173 |
| Shapes | Control | 0.621 | 0.222 |
| SysPrompt (labeled) | SysPrompt | 0.873 | 0.169 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.945 | 0.258 |
| Emotions | Mental | 1.337 | 0.241 |
| Agency | Mental | 1.692 | 0.445 |
| Intentions | Mental | 1.821 | 0.417 |
| Prediction | Mental | 2.026 | 0.328 |
| Cognitive | Mental | 2.166 | 0.334 |
| Social | Mental | 1.910 | 0.346 |
| Attention | Mental | 2.647 | 0.286 |
| Embodiment | Physical | 0.542 | 0.300 |
| Roles | Physical | 1.109 | 0.261 |
| Animacy | Physical | 0.675 | 0.198 |
| Formality | Pragmatic | 0.818 | 0.235 |
| Expertise | Pragmatic | 1.312 | 0.286 |
| Helpfulness | Pragmatic | 0.753 | 0.199 |
| Baseline | Control | 2.491 | 0.151 |
| Biological | Control | 0.455 | 0.201 |
| Shapes | Control | 0.649 | 0.246 |
| SysPrompt (labeled) | SysPrompt | 2.449 | 0.148 |

## Entity Overlap (Mean |cosine| with baseline)

| Dimension | Entity Overlap |
|---|---|
| Phenomenology | 0.5669 |
| Emotions | 0.4941 |
| Agency | 0.5390 |
| Intentions | 0.4166 |
| Prediction | 0.5678 |
| Cognitive | 0.2396 |
| Social | 0.5568 |
| Attention | 0.4183 |
| Embodiment | 0.6324 |
| Roles | 0.6834 |
| Animacy | 0.7429 |
| Formality | 0.2335 |
| Expertise | 0.1731 |
| Helpfulness | 0.1964 |
| Baseline | 0.9756 |
| Biological | 0.5146 |
| Shapes | 0.1273 |
| SysPrompt (labeled) | 0.2829 |

## Methods

- **Analysis**: Residual Alignment (Entity Baseline Projected Out)
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
