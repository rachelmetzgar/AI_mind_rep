# Exp 3: Standalone Concept Alignment — Cross-Version Comparison

*Generated: 2026-03-04 11:36*

> Alignment between standalone mean activation vectors and probe weights. Standalone prompts describe concepts without entity framing (no 'human' or 'AI' words). Concept vector = mean activation across all prompts for that concept. Tests whether alignment is driven by concept content rather than entity labels.

## Summary (Mental Dimensions)

| Version | Metacognitive R² (×10⁻³) | Operational R² (×10⁻³) | Description |
|---------|---|---|---|
| Partner Identity | 0.330 | 0.327 | Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues |
| Control | 0.305 | 0.354 | Nonsense codewords replacing identity labels — no meaningful identity cues |

## Per-Dimension: Metacognitive Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.321 | 0.305 |
| Emotions | Mental | 0.329 | 0.310 |
| Agency | Mental | 0.332 | 0.309 |
| Intentions | Mental | 0.330 | 0.298 |
| Prediction | Mental | 0.344 | 0.315 |
| Cognitive | Mental | 0.327 | 0.300 |
| Social | Mental | 0.342 | 0.305 |
| Attention | Mental | 0.317 | 0.299 |
| Embodiment | Physical | 0.322 | 0.313 |
| Roles | Physical | 0.317 | 0.286 |
| Animacy | Physical | 0.313 | 0.303 |
| Formality | Pragmatic | 0.317 | 0.284 |
| Expertise | Pragmatic | 0.325 | 0.290 |
| Helpfulness | Pragmatic | 0.325 | 0.283 |
| Human (concept) | Entity | 0.333 | 0.299 |
| AI (concept) | Entity | 0.315 | 0.287 |
| Biological | Control | 0.311 | 0.310 |
| Shapes | Control | 0.311 | 0.313 |
| SysPrompt (talkto human) | SysPrompt | 0.376 | 0.351 |
| SysPrompt (talkto AI) | SysPrompt | 0.393 | 0.357 |
| SysPrompt (bare human) | SysPrompt | 0.342 | 0.305 |
| SysPrompt (bare AI) | SysPrompt | 0.350 | 0.319 |

## Per-Dimension: Operational Probe R² (×10⁻³)

| Dimension | Category | Partner Identity | Control |
|---|---|---|---|
| Phenomenology | Mental | 0.333 | 0.296 |
| Emotions | Mental | 0.322 | 0.284 |
| Agency | Mental | 0.320 | 0.333 |
| Intentions | Mental | 0.325 | 0.312 |
| Prediction | Mental | 0.325 | 0.361 |
| Cognitive | Mental | 0.344 | 0.423 |
| Social | Mental | 0.324 | 0.476 |
| Attention | Mental | 0.326 | 0.343 |
| Embodiment | Physical | 0.333 | 0.312 |
| Roles | Physical | 0.323 | 0.391 |
| Animacy | Physical | 0.326 | 0.285 |
| Formality | Pragmatic | 0.346 | 0.447 |
| Expertise | Pragmatic | 0.375 | 0.484 |
| Helpfulness | Pragmatic | 0.369 | 0.466 |
| Human (concept) | Entity | 0.274 | 0.437 |
| AI (concept) | Entity | 0.381 | 0.566 |
| Biological | Control | 0.304 | 0.309 |
| Shapes | Control | 0.313 | 0.292 |
| SysPrompt (talkto human) | SysPrompt | 0.211 | 0.507 |
| SysPrompt (talkto AI) | SysPrompt | 0.359 | 0.535 |
| SysPrompt (bare human) | SysPrompt | 0.261 | 0.492 |
| SysPrompt (bare AI) | SysPrompt | 0.375 | 0.547 |

## Methods

- **Analysis**: Standalone Concept Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, logistic per-layer (metacognitive + operational)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
