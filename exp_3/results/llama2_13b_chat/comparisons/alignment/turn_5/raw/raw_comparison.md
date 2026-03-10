# Exp 3: Raw Contrast Alignment — Cross-Version Comparison

*Generated: 2026-03-08 14:55*

> Raw cosine alignment between human-AI contrast vectors and probe weight vectors. No entity baseline subtraction. Concept direction = mean(human) - mean(AI) per layer.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.644 | 2.827 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.198 | 0.324 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.443 | 0.212 |
| Emotions | Mental | 0.451 | 0.171 |
| Agency | Mental | 0.540 | 0.143 |
| Intentions | Mental | 0.675 | 0.169 |
| Prediction | Mental | 0.717 | 0.178 |
| Cognitive | Mental | 0.862 | 0.253 |
| Social | Mental | 0.709 | 0.168 |
| Attention | Mental | 0.908 | 0.228 |
| Beliefs | Mental | 0.547 | 0.282 |
| Desires | Mental | 0.493 | 0.212 |
| Goals | Mental | 0.744 | 0.164 |
| Embodiment | Physical | 0.308 | 0.207 |
| Roles | Physical | 0.581 | 0.165 |
| Animacy | Physical | 0.425 | 0.245 |
| Formality | Pragmatic | 0.314 | 0.253 |
| Expertise | Pragmatic | 0.455 | 0.283 |
| Helpfulness | Pragmatic | 0.297 | 0.257 |
| Baseline | Control | 0.653 | 0.187 |
| Biological | Control | 0.327 | 0.207 |
| Shapes | Control | 0.304 | 0.206 |
| Granite/Sandstone | Control | 0.176 | 0.194 |
| Squares/Triangles | Control | 0.221 | 0.221 |
| Horizontal/Vertical | Control | 0.195 | 0.314 |
| SysPrompt (labeled) | SysPrompt | 0.893 | 0.196 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 2.108 | 0.267 |
| Emotions | Mental | 1.999 | 0.340 |
| Agency | Mental | 2.577 | 0.379 |
| Intentions | Mental | 2.839 | 0.383 |
| Prediction | Mental | 2.987 | 0.313 |
| Cognitive | Mental | 3.142 | 0.276 |
| Social | Mental | 3.366 | 0.291 |
| Attention | Mental | 3.540 | 0.275 |
| Beliefs | Mental | 3.005 | 0.373 |
| Desires | Mental | 2.545 | 0.377 |
| Goals | Mental | 2.996 | 0.288 |
| Embodiment | Physical | 1.771 | 0.275 |
| Roles | Physical | 2.521 | 0.257 |
| Animacy | Physical | 1.401 | 0.182 |
| Formality | Pragmatic | 0.330 | 0.237 |
| Expertise | Pragmatic | 1.133 | 0.214 |
| Helpfulness | Pragmatic | 0.270 | 0.203 |
| Baseline | Control | 2.886 | 0.143 |
| Biological | Control | 1.039 | 0.221 |
| Shapes | Control | 0.245 | 0.190 |
| Granite/Sandstone | Control | 0.078 | 0.166 |
| Squares/Triangles | Control | 0.259 | 0.204 |
| Horizontal/Vertical | Control | 0.231 | 0.260 |
| SysPrompt (labeled) | SysPrompt | 2.939 | 0.076 |

## Methods

- **Analysis**: Raw Contrast Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/2a_alignment_analysis.py`
