# Exp 3: Standalone Concept Alignment — Cross-Version Comparison

*Generated: 2026-02-26 15:47*

> Alignment between standalone mean activation vectors and probe weights. Standalone prompts describe concepts without entity framing (no 'human' or 'AI' words). Concept vector = mean activation across all prompts for that concept. Tests whether alignment is driven by concept content rather than entity labels.

## Summary (Mental Dimensions)

| Version | Reading R² (×10⁻³) | Control R² (×10⁻³) | Description |
|---------|---|---|---|
| Labels | 0.495 | 0.973 | Partner identified as 'human' or 'AI' (no names) |
| Balanced Names | 0.633 | 3.412 | Gender-balanced names (e.g., Alex/Jordan) |
| Balanced GPT | 1.361 | 17.112 | Balanced names with GPT-4 replacing 'AI' partner |
| Names (orig.) | 1.081 | 3.212 | Original Sam/Casey names (deprecated due to name confound) |
| Nonsense Codeword | 0.314 | 1.781 | Nonsense codewords replacing identity labels |
| Nonsense Ignore | 0.299 | 0.615 | Nonsense labels with instruction to ignore them |

## Per-Dimension: Reading Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore |
|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.499 | 0.691 | 1.426 | 1.192 | 0.317 | 0.290 |
| Emotions | Mental | 0.491 | 0.636 | 1.379 | 1.065 | 0.319 | 0.298 |
| Agency | Mental | 0.496 | 0.638 | 1.384 | 1.092 | 0.310 | 0.297 |
| Intentions | Mental | 0.483 | 0.627 | 1.347 | 0.998 | 0.311 | 0.294 |
| Prediction | Mental | 0.515 | 0.631 | 1.361 | 1.076 | 0.320 | 0.304 |
| Cognitive | Mental | 0.502 | 0.620 | 1.321 | 1.077 | 0.306 | 0.301 |
| Social | Mental | 0.480 | 0.541 | 1.267 | 0.996 | 0.315 | 0.304 |
| Attention | Mental | 0.496 | 0.679 | 1.406 | 1.152 | 0.311 | 0.304 |
| Embodiment | Physical | 0.515 | 0.690 | 1.457 | 1.200 | 0.308 | 0.291 |
| Roles | Physical | 0.487 | 0.621 | 1.386 | 1.072 | 0.307 | 0.295 |
| Animacy | Physical | 0.505 | 0.727 | 1.460 | 1.219 | 0.302 | 0.280 |
| Formality | Pragmatic | 0.500 | 0.619 | 1.346 | 1.139 | 0.321 | 0.301 |
| Expertise | Pragmatic | 0.500 | 0.624 | 1.356 | 1.105 | 0.310 | 0.292 |
| Helpfulness | Pragmatic | 0.480 | 0.592 | 1.327 | 1.066 | 0.315 | 0.287 |
| Human (concept) | Entity | 0.523 | 0.677 | 1.551 | 1.167 | 0.317 | 0.316 |
| AI (concept) | Entity | 0.551 | 0.743 | 1.653 | 1.209 | 0.322 | 0.323 |
| Biological | Control | 0.502 | 0.720 | 1.411 | 1.215 | 0.302 | 0.298 |
| Shapes | Control | 0.522 | 0.773 | 1.401 | 1.349 | 0.317 | 0.310 |
| SysPrompt (talkto human) | SysPrompt | 0.483 | 0.449 | 1.120 | 0.791 | 0.352 | 0.397 |
| SysPrompt (talkto AI) | SysPrompt | 0.523 | 0.541 | 1.376 | 0.955 | 0.341 | 0.391 |
| SysPrompt (bare human) | SysPrompt | 0.489 | 0.485 | 1.235 | 0.771 | 0.322 | 0.373 |
| SysPrompt (bare AI) | SysPrompt | 0.502 | 0.572 | 1.517 | 0.977 | 0.311 | 0.367 |

## Per-Dimension: Control Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore |
|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.978 | 3.552 | 17.252 | 3.353 | 1.789 | 0.613 |
| Emotions | Mental | 0.975 | 3.471 | 17.167 | 3.240 | 1.780 | 0.599 |
| Agency | Mental | 0.978 | 3.448 | 17.129 | 3.254 | 1.781 | 0.602 |
| Intentions | Mental | 0.965 | 3.415 | 17.063 | 3.134 | 1.788 | 0.604 |
| Prediction | Mental | 0.987 | 3.396 | 17.090 | 3.217 | 1.781 | 0.614 |
| Cognitive | Mental | 0.972 | 3.359 | 17.063 | 3.185 | 1.773 | 0.637 |
| Social | Mental | 0.965 | 3.169 | 16.956 | 3.010 | 1.778 | 0.631 |
| Attention | Mental | 0.966 | 3.487 | 17.175 | 3.305 | 1.775 | 0.622 |
| Embodiment | Physical | 0.996 | 3.569 | 17.266 | 3.389 | 1.782 | 0.612 |
| Roles | Physical | 0.958 | 3.407 | 17.127 | 3.223 | 1.781 | 0.616 |
| Animacy | Physical | 0.968 | 3.644 | 17.326 | 3.376 | 1.776 | 0.602 |
| Formality | Pragmatic | 0.966 | 3.228 | 16.986 | 3.139 | 1.781 | 0.633 |
| Expertise | Pragmatic | 0.965 | 3.289 | 17.049 | 3.110 | 1.764 | 0.655 |
| Helpfulness | Pragmatic | 0.966 | 3.195 | 16.994 | 3.037 | 1.771 | 0.652 |
| Human (concept) | Entity | 0.964 | 3.541 | 17.360 | 3.355 | 1.772 | 0.559 |
| AI (concept) | Entity | 0.956 | 3.824 | 17.669 | 3.532 | 1.786 | 0.570 |
| Biological | Control | 0.985 | 3.643 | 17.272 | 3.302 | 1.780 | 0.596 |
| Shapes | Control | 0.987 | 3.608 | 17.204 | 3.442 | 1.790 | 0.607 |
| SysPrompt (talkto human) | SysPrompt | 0.944 | 2.820 | 16.709 | 2.733 | 1.799 | 0.598 |
| SysPrompt (talkto AI) | SysPrompt | 0.969 | 3.282 | 17.221 | 3.203 | 1.784 | 0.591 |
| SysPrompt (bare human) | SysPrompt | 0.933 | 2.861 | 16.700 | 2.545 | 1.776 | 0.597 |
| SysPrompt (bare AI) | SysPrompt | 0.933 | 3.339 | 17.303 | 3.115 | 1.764 | 0.582 |

## Methods

- **Analysis**: Standalone Concept Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, turn 5, logistic per-layer (reading + control)
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
