#!/usr/bin/env python3
"""
Experiment 3, Phase 4: Concept Vector Causal Steering — V1 Generation

Mean-vector steering: injects concept contrast directions (human - AI mean
difference) into LLaMA-2-13B-Chat conversation generation. Tests whether
concept-specific directions causally shift behavior.

Key difference from 5_concept_intervention.py: uses raw concept mean-difference
vectors instead of trained concept probes. This tests whether the untrained
geometric structure is causally relevant.

Intervention formula:
    h'[layer, last_token] = h[layer, last_token] + sign * N * direction[layer]
    where sign = +1 (human) or -1 (AI), direction is unit-normalized

Layer selection strategies (--strategies):
    exp2_peak        : Top 15 layers from Exp 2 metacognitive probe accuracy
    upper_half       : Layers 20-40 (21 layers)
    concept_aligned  : Top 15 layers by |cosine similarity| between concept
                       vector and Exp 2 metacognitive probe weight (per-dim)

Output:
    results/versions/{version}/concept_steering/v1/{dim_name}/{strategy}/
        N_{strength}_results.csv
        N_{strength}_results.txt
        N_{strength}_config.json

Usage:
    python 4_concept_steering_generate.py --version balanced_gpt --dim_id 0
    python 4_concept_steering_generate.py --version balanced_gpt --dim_id 0 --strategies exp2_peak upper_half
    python 4_concept_steering_generate.py --version balanced_gpt --dim_id 0 --strengths 2 4

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import csv
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Local imports ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import llama_v2_prompt
from config import config, set_version, add_version_argument, ensure_dir


# ========================== INLINE TRACEDICT ========================== #

class TraceDict:
    """Minimal replacement for baukit.TraceDict using forward hooks."""
    def __init__(self, model, layers, edit_output=None):
        self.model = model
        self.layers = layers
        self.edit_output = edit_output
        self.handles = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name in self.layers and self.edit_output is not None:
                def make_hook(ln):
                    def hook_fn(mod, inp, output):
                        return self.edit_output(output, ln)
                    return hook_fn
                h = module.register_forward_hook(make_hook(name))
                self.handles.append(h)
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ========================== CONFIG ========================== #

MODEL_NAME = config.MODEL_NAME
INPUT_DIM = config.INPUT_DIM
N_LAYERS = config.N_LAYERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# V1 generation parameters
V1_MAX_NEW_TOKENS = config.GEN_V1.max_new_tokens
V1_TEMPERATURE = config.GEN_V1.temperature
V1_TOP_P = config.GEN_V1.top_p
V1_DO_SAMPLE = config.GEN_V1.do_sample
V1_BATCH_SIZE = config.GEN_V1.batch_size

# Strategy definitions
STRATEGY_NAMES = ["exp2_peak", "upper_half", "concept_aligned"]
DEFAULT_STRENGTHS = [2, 4]
EXP2_PEAK_TOP_K = 15

# Paths — set after set_version()
CONCEPT_VECTOR_ROOT = None
RESULT_DIR = None
CAUSAL_QUESTION_PATH = None
CONCEPT_ALIGNED_JSON = None


def _init_paths(version):
    """Initialize version-dependent paths."""
    global CONCEPT_VECTOR_ROOT, RESULT_DIR, CAUSAL_QUESTION_PATH, CONCEPT_ALIGNED_JSON
    set_version(version)
    CONCEPT_VECTOR_ROOT = config.PATHS.concept_activations_contrasts
    RESULT_DIR = config.RESULTS.root / "versions" / version / "concept_steering" / "v1"
    CAUSAL_QUESTION_PATH = config.PATHS.causality_questions
    CONCEPT_ALIGNED_JSON = config.RESULTS.alignment / "concept_aligned_layers.json"


# ========================== DIMENSION DISCOVERY ========================== #

def discover_dimensions(concept_vector_root):
    """Scan concept_activations/contrasts for available dimensions.

    Returns dict: {dim_id: dim_name} where dim_name is the directory name.
    """
    dims = {}
    for entry in sorted(os.listdir(concept_vector_root)):
        path = os.path.join(concept_vector_root, entry)
        if not os.path.isdir(path):
            continue
        parts = entry.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            dim_id = int(parts[0])
        except ValueError:
            continue
        vec_path = os.path.join(path, "concept_vector_per_layer.npz")
        if os.path.isfile(vec_path):
            dims[dim_id] = entry
    return dims


# ========================== VECTOR LOADING ========================== #

def load_concept_direction(concept_vector_root, dim_name):
    """Load and unit-normalize concept direction vectors.

    The concept_direction array is shape (N_LAYERS, INPUT_DIM) where
    concept_direction[i] = human_mean[i] - ai_mean[i] at layer i.
    Positive direction = more human-like.

    Returns dict: {layer_idx: unit_normalized_direction_tensor} for all layers.
    """
    vec_path = os.path.join(concept_vector_root, dim_name, "concept_vector_per_layer.npz")
    data = np.load(vec_path)
    directions_raw = data["concept_direction"]  # shape (N_LAYERS, INPUT_DIM)

    directions = {}
    for layer_idx in range(directions_raw.shape[0]):
        vec = torch.tensor(directions_raw[layer_idx], dtype=torch.float32, device=DEVICE)
        norm = vec.norm()
        if norm > 1e-8:
            vec = vec / norm
        directions[layer_idx] = vec

    return directions


# ========================== LAYER STRATEGIES ========================== #

def get_exp2_peak_layers(top_k=EXP2_PEAK_TOP_K):
    """Top layers from Exp 2 metacognitive probe accuracy.

    Uses the accuracy_summary.pkl from the exp2 metacognitive probe directory
    (set by set_version). Returns layer indices in hidden_states indexing.
    """
    pkl_path = config.PATHS.exp2_metacognitive / "accuracy_summary.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Metacognitive accuracy summary not found: {pkl_path}"
        )
    with open(pkl_path, "rb") as f:
        summary = pickle.load(f)

    accs = summary["acc"]
    indexed = [(i, a) for i, a in enumerate(accs)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    layers = [i for i, _ in indexed[:top_k]]

    print(f"  [exp2_peak] Top {top_k} metacognitive probe layers: {sorted(layers)}")
    print(f"  Accuracy range: {indexed[top_k-1][1]:.4f} - {indexed[0][1]:.4f}")
    return set(layers)


def get_upper_half_layers():
    """Layers 20-40 (21 layers, hidden_states indexing)."""
    layers = set(range(20, 41))
    print(f"  [upper_half] Layers 20-40 ({len(layers)} layers)")
    return layers


def get_concept_aligned_layers(dim_name, top_k=15):
    """Top layers by |cosine| between concept vector and exp2 metacognitive probe.

    Reads from pre-computed JSON (output of 2h_concept_aligned_layers.py).
    """
    if not CONCEPT_ALIGNED_JSON.exists():
        raise FileNotFoundError(
            f"Concept-aligned layers JSON not found: {CONCEPT_ALIGNED_JSON}\n"
            f"Run 2h_concept_aligned_layers.py first."
        )
    with open(CONCEPT_ALIGNED_JSON) as f:
        lookup = json.load(f)

    if dim_name not in lookup:
        raise KeyError(
            f"Dimension '{dim_name}' not found in {CONCEPT_ALIGNED_JSON}"
        )

    layers = lookup[dim_name][:top_k]
    print(f"  [concept_aligned] Top {top_k} layers for {dim_name}: {sorted(layers)}")
    return set(layers)


def resolve_layers(strategy, dim_name):
    """Resolve layer indices (hidden_states indexing) for a strategy."""
    if strategy == "exp2_peak":
        return get_exp2_peak_layers()
    elif strategy == "upper_half":
        return get_upper_half_layers()
    elif strategy == "concept_aligned":
        return get_concept_aligned_layers(dim_name)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ========================== MODEL ========================== #

def load_model_and_tokenizer():
    """Load LLaMA-2-Chat-13B model and tokenizer."""
    print("Loading LLaMA-2-Chat-13B model and tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=True, padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, local_files_only=True,
    )
    model.half().to(DEVICE).eval()
    if "<pad>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded.")
    return model, tokenizer


# ========================== STEERING ========================== #

def make_concept_edit_fn(directions_by_layer, sign, N):
    """Create hook function for concept vector steering.

    Args:
        directions_by_layer: dict {layer_idx: unit-normalized direction tensor}
            Layer indices use hidden_states indexing (0 = embedding, 1 = layer 0 output, ...).
        sign: +1.0 for human-steering, -1.0 for AI-steering
        N: intervention strength (scalar multiplier)
    """
    def edit_fn(output, layer_name):
        if "model.layers." not in layer_name:
            return output
        try:
            layer_num = int(layer_name.split("model.layers.")[-1].split(".")[0])
        except ValueError:
            return output

        layer_idx = layer_num + 1  # hidden_states offset
        if layer_idx not in directions_by_layer:
            return output

        direction = directions_by_layer[layer_idx]
        hidden = output[0]
        if hidden.ndim != 3:
            return output

        # Apply steering to last token only
        delta = (sign * N * direction).to(hidden.dtype)
        hidden = hidden.clone()
        hidden[:, -1, :] += delta

        output = list(output)
        output[0] = hidden
        return tuple(output)

    return edit_fn


def get_layer_names(model, layer_indices):
    """Get model module names for given layer indices (hidden_states indexing).

    Maps layer_idx (hidden_states offset) to model.layers.{layer_idx - 1}.
    """
    names = []
    for name, _ in model.named_modules():
        if name.startswith("model.layers.") and name.split(".")[-1].isdigit():
            layer_num = int(name.split("model.layers.")[-1])
            if (layer_num + 1) in layer_indices:
                names.append(name)
    return names


def null_edit(output, layer_name):
    """No-op edit function for baseline generation."""
    return output


# ========================== V1 GENERATION ========================== #

def v1_generate_batched(model, tokenizer, prompts, layer_names, edit_fn,
                        batch_size=V1_BATCH_SIZE):
    """Generate responses with optional steering via TraceDict hooks."""
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i+batch_size]
        formatted = [
            llama_v2_prompt([
                {"role": "system", "content": "You are a helpful, respectful and honest assistant."},
                {"role": "user", "content": p},
            ])
            for p in batch
        ]
        with TraceDict(model, layer_names, edit_output=edit_fn) as _:
            with torch.no_grad():
                inputs = tokenizer(
                    formatted, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048,
                ).to(DEVICE)
                tokens = model.generate(
                    **inputs, max_new_tokens=V1_MAX_NEW_TOKENS,
                    do_sample=V1_DO_SAMPLE,
                    temperature=V1_TEMPERATURE,
                    top_p=V1_TOP_P,
                )
        for seq in tokens:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            if "[/INST]" in text:
                text = text.split("[/INST]", 1)[1]
            responses.append(text.strip())
    return responses


def v1_save(result_dir, dim_name, strategy, N, questions, baseline,
            human_resp, ai_resp, layer_indices, layer_names):
    """Save V1 results: CSV, TXT examples, config JSON."""
    os.makedirs(result_dir, exist_ok=True)

    # CSV — same format as exp2 V1
    csv_path = os.path.join(result_dir, f"N_{N}_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "question_idx", "question", "condition", "response",
        ])
        w.writeheader()
        for idx, q in enumerate(questions):
            for cond, resps in [
                ("baseline", baseline),
                ("human", human_resp),
                ("ai", ai_resp),
            ]:
                w.writerow({
                    "question_idx": idx, "question": q,
                    "condition": cond, "response": resps[idx],
                })

    # TXT examples
    txt_path = os.path.join(result_dir, f"N_{N}_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Concept vector steering: {dim_name}\n")
        f.write(f"Strategy: {strategy} | N = {N} (unit-normalized direction)\n")
        f.write(f"Active layers (hidden_states idx): {sorted(layer_indices)}\n")
        f.write(f"Questions: {len(questions)}\n")
        f.write("=" * 80 + "\n\n")
        for i in range(len(questions)):
            f.write(f"Q{i}: {questions[i]}\n\n")
            f.write("-" * 50 + "\nBaseline:\n")
            f.write(f"ASSISTANT: {baseline[i]}\n\n")
            f.write("-" * 50 + f"\nHuman-steered (N={N}):\n")
            f.write(f"ASSISTANT: {human_resp[i]}\n\n")
            f.write("-" * 50 + f"\nAI-steered (N={N}):\n")
            f.write(f"ASSISTANT: {ai_resp[i]}\n\n")
            f.write("=" * 80 + "\n\n")

    # Config JSON
    cfg_path = os.path.join(result_dir, f"N_{N}_config.json")
    gen_config = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp3_concept_vector_steering",
        "dimension": dim_name,
        "strategy": strategy,
        "config": {
            "model": "LLaMA-2-Chat-13B",
            "model_path": MODEL_NAME,
            "vector_type": "concept_mean_difference_unit_normalized",
            "vector_source": str(CONCEPT_VECTOR_ROOT / dim_name / "concept_vector_per_layer.npz"),
            "layer_strategy": strategy,
            "intervention_strength": N,
            "active_layers": sorted(layer_indices),
            "n_active_layers": len(layer_indices),
            "modified_layer_names": layer_names,
            "max_new_tokens": V1_MAX_NEW_TOKENS,
            "temperature": V1_TEMPERATURE,
            "top_p": V1_TOP_P,
            "do_sample": V1_DO_SAMPLE,
            "n_questions": len(questions),
        },
    }
    with open(cfg_path, "w") as f:
        json.dump(gen_config, f, indent=2)

    print(f"  Saved to {result_dir}")


# ========================== MAIN ========================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 3: Concept vector causal steering (V1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Layer strategies:
  exp2_peak       : Top 15 layers from Exp 2 metacognitive probe accuracy
  upper_half      : Layers 20-40 (21 layers)
  concept_aligned : Top 15 layers by |cosine| with metacognitive probe (per-dim)

Examples:
  python 4_concept_steering_generate.py --version balanced_gpt --dim_id 0
  python 4_concept_steering_generate.py --version balanced_gpt --dim_id 14 --strategies exp2_peak upper_half
  python 4_concept_steering_generate.py --version balanced_gpt --dim_id 0 --strengths 2 4
        """,
    )
    add_version_argument(p)
    p.add_argument(
        "--dim_id", type=int, required=True,
        help="Concept dimension ID (e.g., 0=baseline, 14=biological, 15=shapes, 17=attention).",
    )
    p.add_argument(
        "--strategies", type=str, nargs="+",
        default=STRATEGY_NAMES, choices=STRATEGY_NAMES,
        help=f"Layer selection strategies to run. Default: all three.",
    )
    p.add_argument(
        "--strengths", type=int, nargs="+",
        default=DEFAULT_STRENGTHS,
        help=f"Intervention strengths (default: {DEFAULT_STRENGTHS}).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    _init_paths(args.version)

    # Discover available dimensions
    dim_registry = discover_dimensions(CONCEPT_VECTOR_ROOT)
    if args.dim_id not in dim_registry:
        print(f"[ERROR] dim_id={args.dim_id} not found.")
        print(f"  Available: {sorted(dim_registry.keys())}")
        sys.exit(1)
    dim_name = dim_registry[args.dim_id]

    print(f"\n{'#'*70}")
    print(f"  Concept Vector Steering V1")
    print(f"  Version:    {args.version}")
    print(f"  Dimension:  {args.dim_id} ({dim_name})")
    print(f"  Strategies: {args.strategies}")
    print(f"  Strengths:  {args.strengths}")
    print(f"  Output:     {RESULT_DIR}")
    print(f"{'#'*70}\n")

    # Load concept direction vectors (all layers)
    all_directions = load_concept_direction(CONCEPT_VECTOR_ROOT, dim_name)
    print(f"Loaded concept direction for {dim_name}: {len(all_directions)} layers")

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Load test questions
    if not os.path.isfile(CAUSAL_QUESTION_PATH):
        print(f"[ERROR] Test questions not found: {CAUSAL_QUESTION_PATH}")
        sys.exit(1)
    with open(CAUSAL_QUESTION_PATH) as f:
        questions = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(questions)} test questions.")

    # Generate baseline (shared across all strategies/strengths)
    print("\n=== Generating baseline responses ===")
    baseline = v1_generate_batched(model, tokenizer, questions, [], null_edit)

    # Run each strategy x strength
    for strategy in args.strategies:
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy}")
        print(f"{'='*60}")

        try:
            layer_indices = resolve_layers(strategy, dim_name)
        except (FileNotFoundError, KeyError) as e:
            print(f"  [SKIP] {strategy}: {e}")
            continue

        # Filter directions to selected layers
        active_directions = {
            idx: all_directions[idx]
            for idx in layer_indices
            if idx in all_directions
        }
        layer_names = get_layer_names(model, set(active_directions.keys()))

        print(f"  Active layers: {sorted(active_directions.keys())} "
              f"({len(active_directions)} layers)")
        print(f"  Model hooks: {layer_names}")

        for N in args.strengths:
            result_dir = str(RESULT_DIR / dim_name / strategy)

            # Skip if already done
            csv_path = os.path.join(result_dir, f"N_{N}_results.csv")
            if os.path.isfile(csv_path):
                print(f"  [SKIP] Already exists: {csv_path}")
                continue

            print(f"\n  --- {dim_name} | {strategy} | N={N} ---")

            print(f"  Human-steered (N={N})...")
            human_resp = v1_generate_batched(
                model, tokenizer, questions, layer_names,
                make_concept_edit_fn(active_directions, +1.0, N),
            )

            print(f"  AI-steered (N={N})...")
            ai_resp = v1_generate_batched(
                model, tokenizer, questions, layer_names,
                make_concept_edit_fn(active_directions, -1.0, N),
            )

            v1_save(
                result_dir, dim_name, strategy, N,
                questions, baseline, human_resp, ai_resp,
                sorted(active_directions.keys()), layer_names,
            )

    print(f"\nV1 concept vector steering complete for {dim_name}.")


if __name__ == "__main__":
    main()
