# Exp 3: Raw Contrast Alignment — Cross-Version Comparison

*Generated: 2026-03-09 14:57*

> Raw cosine alignment between human-AI contrast vectors and probe weight vectors. No entity baseline subtraction. Concept direction = mean(human) - mean(AI) per layer.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.362 | 0.595 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.204 | 0.208 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.369 | 0.239 |
| Emotions | Mental | 0.310 | 0.164 |
| Agency | Mental | 0.268 | 0.214 |
| Intentions | Mental | 0.154 | 0.210 |
| Prediction | Mental | 0.364 | 0.188 |
| Cognitive | Mental | 0.241 | 0.202 |
| Social | Mental | 0.373 | 0.228 |
| Attention | Mental | 0.699 | 0.207 |
| Beliefs | Mental | 0.419 | 0.177 |
| Desires | Mental | 0.364 | 0.218 |
| Goals | Mental | 0.419 | 0.196 |
| Embodiment | Physical | 0.276 | 0.237 |
| Roles | Physical | 0.506 | 0.242 |
| Animacy | Physical | 0.120 | 0.201 |
| Formality | Pragmatic | 0.346 | 0.233 |
| Expertise | Pragmatic | 0.441 | 0.173 |
| Helpfulness | Pragmatic | 0.395 | 0.274 |
| Biological | Control | 0.123 | 0.281 |
| Shapes | Control | 0.219 | 0.165 |
| Granite/Sandstone | Control | 0.250 | 0.193 |
| Squares/Triangles | Control | 0.322 | 0.199 |
| Horizontal/Vertical | Control | 0.141 | 0.206 |
| SysPrompt (labeled) | SysPrompt | 0.355 | 0.178 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.207 | 0.193 |
| Emotions | Mental | 0.343 | 0.248 |
| Agency | Mental | 0.398 | 0.248 |
| Intentions | Mental | 0.546 | 0.149 |
| Prediction | Mental | 0.438 | 0.224 |
| Cognitive | Mental | 0.246 | 0.163 |
| Social | Mental | 0.662 | 0.259 |
| Attention | Mental | 2.175 | 0.120 |
| Beliefs | Mental | 0.609 | 0.214 |
| Desires | Mental | 0.565 | 0.260 |
| Goals | Mental | 0.353 | 0.207 |
| Embodiment | Physical | 0.135 | 0.134 |
| Roles | Physical | 0.466 | 0.114 |
| Animacy | Physical | 0.206 | 0.210 |
| Formality | Pragmatic | 0.342 | 0.219 |
| Expertise | Pragmatic | 0.448 | 0.291 |
| Helpfulness | Pragmatic | 0.740 | 0.262 |
| Biological | Control | 0.272 | 0.253 |
| Shapes | Control | 0.431 | 0.305 |
| Granite/Sandstone | Control | 0.396 | 0.209 |
| Squares/Triangles | Control | 0.567 | 0.207 |
| Horizontal/Vertical | Control | 0.289 | 0.236 |
| SysPrompt (labeled) | SysPrompt | 0.429 | 0.248 |

## Methods

- **Analysis**: Raw Contrast Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/2a_alignment_analysis.py`
