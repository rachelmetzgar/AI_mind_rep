# Exp 3: Raw Contrast Alignment — Cross-Version Comparison

*Generated: 2026-03-04 11:39*

> Raw cosine alignment between human-AI contrast vectors and probe weight vectors. No entity baseline subtraction. Concept direction = mean(human) - mean(AI) per layer.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.445 | 5.004 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.202 | 0.356 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.196 | 0.197 |
| Emotions | Mental | 0.239 | 0.151 |
| Agency | Mental | 0.292 | 0.177 |
| Intentions | Mental | 0.496 | 0.194 |
| Prediction | Mental | 0.388 | 0.227 |
| Cognitive | Mental | 0.784 | 0.222 |
| Social | Mental | 0.486 | 0.211 |
| Attention | Mental | 0.677 | 0.238 |
| Embodiment | Physical | 0.174 | 0.238 |
| Roles | Physical | 0.223 | 0.194 |
| Animacy | Physical | 0.477 | 0.244 |
| Formality | Pragmatic | 0.372 | 0.270 |
| Expertise | Pragmatic | 0.559 | 0.210 |
| Helpfulness | Pragmatic | 0.674 | 0.270 |
| Baseline | Control | 0.586 | 0.239 |
| Biological | Control | 0.192 | 0.253 |
| Shapes | Control | 0.638 | 0.236 |
| SysPrompt (labeled) | SysPrompt | 0.895 | 0.189 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 3.752 | 0.330 |
| Emotions | Mental | 4.112 | 0.289 |
| Agency | Mental | 4.817 | 0.425 |
| Intentions | Mental | 4.765 | 0.454 |
| Prediction | Mental | 5.490 | 0.354 |
| Cognitive | Mental | 5.270 | 0.337 |
| Social | Mental | 5.611 | 0.354 |
| Attention | Mental | 6.217 | 0.304 |
| Embodiment | Physical | 3.270 | 0.338 |
| Roles | Physical | 4.553 | 0.264 |
| Animacy | Physical | 2.908 | 0.174 |
| Formality | Pragmatic | 0.968 | 0.207 |
| Expertise | Pragmatic | 1.801 | 0.256 |
| Helpfulness | Pragmatic | 0.616 | 0.175 |
| Baseline | Control | 4.741 | 0.178 |
| Biological | Control | 2.034 | 0.242 |
| Shapes | Control | 0.575 | 0.229 |
| SysPrompt (labeled) | SysPrompt | 5.185 | 0.170 |

## Methods

- **Analysis**: Raw Contrast Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
