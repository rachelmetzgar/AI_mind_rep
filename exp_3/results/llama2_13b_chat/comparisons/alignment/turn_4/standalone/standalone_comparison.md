# Exp 3: Standalone Concept Alignment — Cross-Version Comparison

*Generated: 2026-03-04 11:40*

> Alignment between standalone mean activation vectors and probe weights. Standalone prompts describe concepts without entity framing (no 'human' or 'AI' words). Concept vector = mean activation across all prompts for that concept. Tests whether alignment is driven by concept content rather than entity labels.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.537 | 2.601 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.240 | 0.143 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.624 | 0.238 |
| Emotions | Mental | 0.613 | 0.248 |
| Agency | Mental | 0.562 | 0.246 |
| Intentions | Mental | 0.540 | 0.243 |
| Prediction | Mental | 0.541 | 0.239 |
| Cognitive | Mental | 0.471 | 0.235 |
| Social | Mental | 0.400 | 0.232 |
| Attention | Mental | 0.547 | 0.238 |
| Embodiment | Physical | 0.616 | 0.251 |
| Roles | Physical | 0.507 | 0.233 |
| Animacy | Physical | 0.612 | 0.245 |
| Formality | Pragmatic | 0.481 | 0.221 |
| Expertise | Pragmatic | 0.433 | 0.221 |
| Helpfulness | Pragmatic | 0.450 | 0.222 |
| Human (concept) | Entity | 0.626 | 0.226 |
| AI (concept) | Entity | 0.605 | 0.215 |
| Biological | Control | 0.548 | 0.245 |
| Shapes | Control | 0.530 | 0.240 |
| SysPrompt (talkto human) | SysPrompt | 0.291 | 0.270 |
| SysPrompt (talkto AI) | SysPrompt | 0.419 | 0.251 |
| SysPrompt (bare human) | SysPrompt | 0.353 | 0.257 |
| SysPrompt (bare AI) | SysPrompt | 0.443 | 0.246 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 2.807 | 0.139 |
| Emotions | Mental | 2.670 | 0.141 |
| Agency | Mental | 2.634 | 0.154 |
| Intentions | Mental | 2.572 | 0.150 |
| Prediction | Mental | 2.549 | 0.138 |
| Cognitive | Mental | 2.578 | 0.139 |
| Social | Mental | 2.263 | 0.140 |
| Attention | Mental | 2.738 | 0.144 |
| Embodiment | Physical | 2.927 | 0.136 |
| Roles | Physical | 2.643 | 0.151 |
| Animacy | Physical | 3.012 | 0.149 |
| Formality | Pragmatic | 2.301 | 0.140 |
| Expertise | Pragmatic | 2.464 | 0.151 |
| Helpfulness | Pragmatic | 2.323 | 0.152 |
| Human (concept) | Entity | 2.830 | 0.184 |
| AI (concept) | Entity | 3.440 | 0.165 |
| Biological | Control | 3.004 | 0.163 |
| Shapes | Control | 2.972 | 0.140 |
| SysPrompt (talkto human) | SysPrompt | 1.850 | 0.185 |
| SysPrompt (talkto AI) | SysPrompt | 2.832 | 0.179 |
| SysPrompt (bare human) | SysPrompt | 2.080 | 0.217 |
| SysPrompt (bare AI) | SysPrompt | 3.071 | 0.208 |

## Methods

- **Analysis**: Standalone Concept Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
