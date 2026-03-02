# Exp 3: Standalone Concept Alignment — Cross-Version Comparison

*Generated: 2026-03-02 14:48*

> Alignment between standalone mean activation vectors and probe weights. Standalone prompts describe concepts without entity framing (no 'human' or 'AI' words). Concept vector = mean activation across all prompts for that concept. Tests whether alignment is driven by concept content rather than entity labels.

## Summary (Mental Dimensions)

| Version | Reading R² (×10⁻³) | Control R² (×10⁻³) | Description |
|---------|---|---|---|
| Labels | 0.508 | 0.403 | Partner identified as 'human' or 'AI' (no names) |
| Balanced Names | 0.665 | 2.466 | Gender-balanced names (e.g., Alex/Jordan) |
| Balanced GPT | 1.491 | 2.020 | Balanced names with GPT-4 replacing 'AI' partner |
| Names (orig.) | 1.196 | 2.532 | Original Sam/Casey names (deprecated due to name confound) |
| Nonsense Codeword | 0.216 | 0.157 | Nonsense codewords replacing identity labels |
| Nonsense Ignore | 0.198 | 0.281 | Nonsense labels with instruction to ignore them |
| Labels + Turnwise | 0.636 | 0.661 | Labels + turn-level 'Human:'/'AI:' prefix each turn |
| You-Are Labels | 0.255 | 0.272 | 'You are talking to a Human/an AI' framing |
| You-Are Bal. GPT | 1.532 | 2.318 | 'You are talking to' + named partners (Gregory/Rebecca, ChatGPT/GPT-4) |
| You-Are Lab. Turn. | 0.270 | 0.603 | 'You are talking to' framing + turn-level prefix |

## Per-Dimension: Reading Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore | Labels + Turnwise | You-Are Labels | You-Are Bal. GPT | You-Are Lab. Turn. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.512 | 0.732 | 1.565 | 1.325 | 0.217 | 0.188 | 0.678 | 0.274 | 1.558 | 0.222 |
| Emotions | Mental | 0.505 | 0.669 | 1.512 | 1.177 | 0.223 | 0.191 | 0.620 | 0.261 | 1.527 | 0.230 |
| Agency | Mental | 0.509 | 0.671 | 1.518 | 1.209 | 0.213 | 0.193 | 0.617 | 0.251 | 1.594 | 0.269 |
| Intentions | Mental | 0.495 | 0.659 | 1.475 | 1.098 | 0.214 | 0.191 | 0.623 | 0.244 | 1.491 | 0.263 |
| Prediction | Mental | 0.531 | 0.663 | 1.491 | 1.190 | 0.221 | 0.204 | 0.646 | 0.254 | 1.576 | 0.284 |
| Cognitive | Mental | 0.515 | 0.650 | 1.445 | 1.192 | 0.209 | 0.203 | 0.628 | 0.246 | 1.512 | 0.305 |
| Social | Mental | 0.489 | 0.559 | 1.382 | 1.097 | 0.217 | 0.206 | 0.600 | 0.245 | 1.441 | 0.345 |
| Attention | Mental | 0.507 | 0.718 | 1.541 | 1.278 | 0.213 | 0.203 | 0.681 | 0.265 | 1.558 | 0.239 |
| Embodiment | Physical | 0.529 | 0.730 | 1.602 | 1.337 | 0.213 | 0.192 | 0.672 | 0.271 | 1.603 | 0.240 |
| Roles | Physical | 0.498 | 0.649 | 1.519 | 1.185 | 0.208 | 0.196 | 0.639 | 0.249 | 1.535 | 0.278 |
| Animacy | Physical | 0.518 | 0.770 | 1.607 | 1.359 | 0.208 | 0.180 | 0.673 | 0.270 | 1.606 | 0.226 |
| Formality | Pragmatic | 0.511 | 0.649 | 1.473 | 1.265 | 0.225 | 0.207 | 0.652 | 0.258 | 1.477 | 0.313 |
| Expertise | Pragmatic | 0.513 | 0.653 | 1.486 | 1.227 | 0.217 | 0.202 | 0.626 | 0.251 | 1.523 | 0.326 |
| Helpfulness | Pragmatic | 0.491 | 0.615 | 1.453 | 1.181 | 0.222 | 0.199 | 0.630 | 0.250 | 1.437 | 0.337 |
| Human (concept) | Entity | 0.531 | 0.707 | 1.704 | 1.285 | 0.216 | 0.198 | 0.643 | 0.275 | 1.775 | 0.239 |
| AI (concept) | Entity | 0.559 | 0.783 | 1.818 | 1.336 | 0.213 | 0.203 | 0.748 | 0.272 | 1.934 | 0.251 |
| Biological | Control | 0.514 | 0.761 | 1.544 | 1.352 | 0.207 | 0.191 | 0.691 | 0.250 | 1.528 | 0.229 |
| Shapes | Control | 0.539 | 0.829 | 1.534 | 1.508 | 0.217 | 0.204 | 0.696 | 0.274 | 1.542 | 0.228 |
| SysPrompt (talkto human) | SysPrompt | 0.487 | 0.438 | 1.206 | 0.839 | 0.244 | 0.257 | 0.630 | 0.288 | 1.061 | 0.302 |
| SysPrompt (talkto AI) | SysPrompt | 0.533 | 0.547 | 1.503 | 1.032 | 0.232 | 0.247 | 0.666 | 0.291 | 1.358 | 0.295 |
| SysPrompt (bare human) | SysPrompt | 0.500 | 0.479 | 1.346 | 0.809 | 0.227 | 0.217 | 0.529 | 0.245 | 1.106 | 0.350 |
| SysPrompt (bare AI) | SysPrompt | 0.512 | 0.581 | 1.673 | 1.051 | 0.213 | 0.211 | 0.568 | 0.245 | 1.407 | 0.336 |

## Per-Dimension: Control Probe R² (×10⁻³)

| Dimension | Category | Labels | Balanced Names | Balanced GPT | Names (orig.) | Nonsense Codeword | Nonsense Ignore | Labels + Turnwise | You-Are Labels | You-Are Bal. GPT | You-Are Lab. Turn. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Phenomenology | Mental | 0.404 | 2.619 | 2.166 | 2.680 | 0.159 | 0.279 | 0.688 | 0.303 | 2.361 | 0.592 |
| Emotions | Mental | 0.402 | 2.521 | 2.072 | 2.542 | 0.154 | 0.264 | 0.637 | 0.282 | 2.307 | 0.570 |
| Agency | Mental | 0.407 | 2.505 | 2.043 | 2.577 | 0.158 | 0.267 | 0.622 | 0.268 | 2.374 | 0.571 |
| Intentions | Mental | 0.393 | 2.460 | 1.961 | 2.429 | 0.161 | 0.269 | 0.627 | 0.262 | 2.250 | 0.576 |
| Prediction | Mental | 0.420 | 2.443 | 1.986 | 2.537 | 0.152 | 0.279 | 0.687 | 0.257 | 2.357 | 0.614 |
| Cognitive | Mental | 0.406 | 2.429 | 1.985 | 2.535 | 0.157 | 0.302 | 0.667 | 0.265 | 2.339 | 0.621 |
| Social | Mental | 0.394 | 2.182 | 1.822 | 2.299 | 0.157 | 0.294 | 0.676 | 0.250 | 2.186 | 0.664 |
| Attention | Mental | 0.394 | 2.569 | 2.127 | 2.661 | 0.159 | 0.292 | 0.686 | 0.293 | 2.370 | 0.614 |
| Embodiment | Physical | 0.423 | 2.672 | 2.247 | 2.763 | 0.168 | 0.277 | 0.690 | 0.297 | 2.446 | 0.618 |
| Roles | Physical | 0.385 | 2.477 | 2.071 | 2.564 | 0.159 | 0.281 | 0.654 | 0.275 | 2.328 | 0.636 |
| Animacy | Physical | 0.393 | 2.762 | 2.305 | 2.756 | 0.161 | 0.262 | 0.647 | 0.303 | 2.456 | 0.579 |
| Formality | Pragmatic | 0.388 | 2.256 | 1.877 | 2.457 | 0.154 | 0.303 | 0.714 | 0.278 | 2.135 | 0.671 |
| Expertise | Pragmatic | 0.392 | 2.363 | 1.962 | 2.468 | 0.154 | 0.313 | 0.685 | 0.271 | 2.297 | 0.670 |
| Helpfulness | Pragmatic | 0.385 | 2.248 | 1.873 | 2.365 | 0.156 | 0.304 | 0.717 | 0.266 | 2.099 | 0.688 |
| Human (concept) | Entity | 0.390 | 2.595 | 2.357 | 2.664 | 0.148 | 0.233 | 0.616 | 0.271 | 2.666 | 0.638 |
| AI (concept) | Entity | 0.381 | 2.913 | 2.731 | 2.873 | 0.147 | 0.249 | 0.818 | 0.286 | 3.137 | 0.801 |
| Biological | Control | 0.408 | 2.761 | 2.288 | 2.668 | 0.175 | 0.266 | 0.621 | 0.290 | 2.380 | 0.533 |
| Shapes | Control | 0.410 | 2.683 | 2.179 | 2.788 | 0.163 | 0.284 | 0.641 | 0.288 | 2.434 | 0.544 |
| SysPrompt (talkto human) | SysPrompt | 0.372 | 1.708 | 1.550 | 1.888 | 0.163 | 0.286 | 0.611 | 0.231 | 1.816 | 0.618 |
| SysPrompt (talkto AI) | SysPrompt | 0.399 | 2.267 | 2.180 | 2.464 | 0.154 | 0.278 | 0.702 | 0.228 | 2.528 | 0.651 |
| SysPrompt (bare human) | SysPrompt | 0.367 | 1.874 | 1.642 | 1.705 | 0.192 | 0.283 | 0.470 | 0.223 | 1.830 | 0.596 |
| SysPrompt (bare AI) | SysPrompt | 0.363 | 2.436 | 2.363 | 2.391 | 0.174 | 0.265 | 0.575 | 0.217 | 2.561 | 0.644 |

## Methods

- **Analysis**: Standalone Concept Alignment
- **Model**: LLaMA-2-13B-Chat
- **Probes**: Exp 2, turn 5, logistic per-layer (reading + control)
- **Layer range**: Layers 6–40 (35 of 41). Layers 0–5 excluded (embedding artifact + prompt-format confound).
- **Bootstrap**: 1,000 iterations (prompt resampling)
- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`
