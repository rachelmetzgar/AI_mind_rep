#!/usr/bin/env python3
"""
Causal intervention — GENERATION ONLY.
Experiment 2: LLaMA-2-Chat-13B with naturalistic conversation data.

Two modes, selected via --mode flag:

  V1: Single-prompt causality test.
      Generates baseline/human-steered/AI-steered responses for each question.
      Saves intervention_responses.csv per probe type × strength.

  V2: Steering-based recreation of Experiment 1 conversations.
      Two-agent back-and-forth matching Exp1 structure.
      Per-subject via SLURM array. Both probe types, 3 conditions each.

Both modes sweep intervention strengths [2, 4, 8, 16] by default.

Layer selection strategies (--layer_strategy):
  narrow   : Contiguous 10-layer window (layers 20-29), matching Viegas et al.
             Only probes within this window AND above MIN_PROBE_ACCURACY are used.
  wide     : All layers above MIN_PROBE_ACCURACY (default 0.70).
             Same as narrow but no window constraint.
  peak_15  : Top 15 layers by accuracy (non-contiguous).
             Selects the 15 highest-accuracy probes from accuracy_summary.pkl.
  all_70   : Alias kept for clarity — identical to 'wide'.
             Every layer whose probe accuracy >= 0.70 is included.

Multiple strategies can be run in one invocation:
    --layer_strategy narrow wide

Output directory structure:
    intervention_results/V{1,2}/{layer_strategy}/{control,reading}_probes/is_{N}/...

Probe configurations (applied within each layer strategy):
  - control_probes:          control probes, layers selected by strategy
  - reading_probes_matched:  reading probes, restricted to control probe layers
  - reading_probes_peak:     reading probes, only layers where control probes fail

Judging is done separately via 3_causality_judge.py.

Usage:
    python code/pipeline/2_causality_generate.py --version labels --mode V1 --layer_strategy narrow
    python code/pipeline/2_causality_generate.py --version labels --mode V1 --layer_strategy narrow wide peak_15 all_70
    python code/pipeline/2_causality_generate.py --version labels --mode V1 --layer_strategy wide --strengths 4 8
    python code/pipeline/2_causality_generate.py --version labels --mode V2 --layer_strategy narrow --subject_idx 0

Attribute: conversation partner type
    label 0 = AI partner
    label 1 = Human partner

Env: llama2_env

Rachel C. Metzgar · Feb 2026
"""

import os
import re
import sys
import csv
import json
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from contextlib import nullcontext

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


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


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config as cfg, set_version, add_version_argument, ensure_dir
from src.dataset import llama_v2_prompt
from src.probes import LinearProbeClassification


# ========================== RUNTIME CONFIG ========================== #
# These are set in main() after set_version() is called.
# Module-level references kept for code clarity.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = DEVICE

# Populated by _init_paths() after set_version()
CONTROL_PROBE_DIR = None
READING_PROBE_DIR = None
CAUSAL_QUESTION_PATH = None
RESULT_DIR_V1 = None
RESULT_DIR_V2 = None
PROMPTS_DIR = None
CONFIG_DIR = None
LOG_DIR = None
LOG_FILE = None
PROBE_CONFIGS = None


def _init_paths():
    """Set module-level path variables from config after set_version()."""
    global CONTROL_PROBE_DIR, READING_PROBE_DIR, CAUSAL_QUESTION_PATH
    global RESULT_DIR_V1, RESULT_DIR_V2, PROMPTS_DIR, CONFIG_DIR
    global LOG_DIR, LOG_FILE, PROBE_CONFIGS

    CONTROL_PROBE_DIR = str(cfg.PATHS.probe_checkpoints / "turn_5" / "control_probe")
    READING_PROBE_DIR = str(cfg.PATHS.probe_checkpoints / "turn_5" / "reading_probe")

    CAUSAL_QUESTION_PATH = str(cfg.PATHS.causality_questions)
    RESULT_DIR_V1 = str(cfg.PATHS.intervention_results / "V1")
    RESULT_DIR_V2 = str(cfg.PATHS.intervention_results / "V2")

    PROMPTS_DIR = cfg.PATHS.exp1_prompts
    CONFIG_DIR = cfg.PATHS.exp1_configs

    LOG_DIR = cfg.RESULTS.version_logs
    LOG_FILE = LOG_DIR / "steering_generation_progress.log"

    PROBE_CONFIGS = [
        {"label": "control_probes",         "probe_dir": CONTROL_PROBE_DIR, "layer_mode": "all"},
    #    {"label": "reading_probes_matched", "probe_dir": READING_PROBE_DIR, "layer_mode": "matched"},
        {"label": "reading_probes_peak",    "probe_dir": READING_PROBE_DIR, "layer_mode": "all"},
    ]


# Intervention config from central config
DEFAULT_STRENGTHS = cfg.INTERVENTION.default_strengths
MIN_PROBE_ACCURACY = cfg.INTERVENTION.min_probe_accuracy
LAYER_STRATEGIES = cfg.LAYER_STRATEGIES
V1_MAX_NEW_TOKENS = cfg.INTERVENTION.v1_max_new_tokens
V1_GEN_TEMPERATURE = cfg.INTERVENTION.v1_temperature
V1_GEN_TOP_P = cfg.INTERVENTION.v1_top_p
V1_GEN_DO_SAMPLE = cfg.INTERVENTION.v1_do_sample
V2_MAX_NEW_TOKENS = cfg.INTERVENTION.v2_max_new_tokens
V2_GEN_TEMPERATURE = cfg.INTERVENTION.v2_temperature
V2_GEN_TOP_P = cfg.INTERVENTION.v2_top_p
PAIRS_TOTAL = cfg.INTERVENTION.v2_pairs_total
HISTORY_PAIRS = cfg.INTERVENTION.v2_history_pairs
CONDITIONS = cfg.INTERVENTION.v2_conditions
SYSTEM_PROMPT = cfg.INTERVENTION.v2_system_prompt


# ========================== ACCURACY AUTO-LOADING ========================== #

def load_accuracy_lookup(probe_dir: str) -> Optional[Dict[int, float]]:
    """Load per-layer accuracy from accuracy_summary.pkl."""
    pkl_path = os.path.join(probe_dir, "accuracy_summary.pkl")
    if not os.path.isfile(pkl_path):
        print(f"  [WARNING] No accuracy_summary.pkl in {probe_dir} — loading all probes unfiltered")
        return None
    with open(pkl_path, "rb") as f:
        summary = pickle.load(f)
    acc_list = summary.get("acc", [])
    if not acc_list:
        print(f"  [WARNING] Empty accuracy list in {pkl_path}")
        return None
    lookup = {i: acc for i, acc in enumerate(acc_list)}
    print(f"  Loaded accuracies for {len(lookup)} layers from {pkl_path}")
    return lookup


# ========================== LAYER STRATEGY RESOLUTION ========================== #

def find_best_contiguous_window(accuracy_lookup: Dict[int, float],
                                 window_size: int,
                                 min_accuracy: float) -> Tuple[int, int]:
    """Find the contiguous window of `window_size` layers with highest mean accuracy.

    Only considers layers that individually meet min_accuracy.
    Returns (start_layer, end_layer) inclusive.
    """
    if not accuracy_lookup:
        # Fallback to Viegas default
        return (20, 29)

    max_layer = max(accuracy_lookup.keys())
    best_mean = -1
    best_start = 20  # fallback

    for start in range(0, max_layer - window_size + 2):
        end = start + window_size - 1
        layers_in_window = [l for l in range(start, end + 1) if l in accuracy_lookup]
        if len(layers_in_window) < window_size:
            continue
        accs = [accuracy_lookup[l] for l in layers_in_window]
        # All layers in window must meet threshold
        if all(a >= min_accuracy for a in accs):
            mean_acc = np.mean(accs)
            if mean_acc > best_mean:
                best_mean = mean_acc
                best_start = start

    return (best_start, best_start + window_size - 1)


def resolve_strategy_layers(strategy_name: str,
                             accuracy_lookup: Dict[int, float],
                             print_fn=print) -> Set[int]:
    """Given a strategy name and accuracy data, return the set of layer indices to use."""

    strategy = LAYER_STRATEGIES[strategy_name]
    min_acc = strategy["min_accuracy"]
    top_k = strategy["top_k"]
    window_size = strategy.get("window_size")

    if accuracy_lookup is None:
        print_fn(f"  [WARNING] No accuracy data — using all available layers for '{strategy_name}'")
        return set()

    # Step 1: Filter by min accuracy
    eligible = {l: acc for l, acc in accuracy_lookup.items() if acc >= min_acc}
    print_fn(f"  [{strategy_name}] {len(eligible)} layers >= {min_acc} accuracy")

    if not eligible:
        return set()

    # Step 2: Apply window constraint (for 'narrow')
    if window_size is not None:
        window_start, window_end = find_best_contiguous_window(
            accuracy_lookup, window_size, min_acc
        )
        eligible = {l: acc for l, acc in eligible.items() if window_start <= l <= window_end}
        print_fn(f"  [{strategy_name}] Best {window_size}-layer window: "
                 f"layers {window_start}-{window_end} "
                 f"(mean acc={np.mean(list(eligible.values())):.3f})")

    # Step 3: Apply top_k (for 'peak_15')
    if top_k is not None and len(eligible) > top_k:
        sorted_layers = sorted(eligible.keys(), key=lambda l: eligible[l], reverse=True)
        kept = sorted_layers[:top_k]
        eligible = {l: eligible[l] for l in kept}
        print_fn(f"  [{strategy_name}] Kept top {top_k} layers: {sorted(eligible.keys())}")

    print_fn(f"  [{strategy_name}] Final layers: {sorted(eligible.keys())} "
             f"(n={len(eligible)}, mean acc={np.mean(list(eligible.values())):.3f})")

    return set(eligible.keys())


# ========================== PROBE LOADING ========================== #

def load_probes(probe_dir: str, device: str, input_dim: int = None,
                allowed_layers: Optional[Set[int]] = None,
                accuracy_lookup: Optional[Dict[int, float]] = None,
                label: str = "probe", raise_if_empty: bool = False):
    """Load probes, optionally restricted to allowed_layers.

    If allowed_layers is provided, only probes at those layer indices are loaded.
    If accuracy_lookup is provided but allowed_layers is None, loads all probes
    (no filtering — filtering should happen via allowed_layers from the strategy).
    """
    if input_dim is None:
        input_dim = cfg.INPUT_DIM
    probes = {}
    if not os.path.isdir(probe_dir):
        msg = f"Probe directory not found: {probe_dir}"
        if raise_if_empty:
            raise RuntimeError(msg)
        print(f"  [WARNING] {msg}")
        return probes

    for fname in os.listdir(probe_dir):
        if not fname.endswith(".pth") or not fname.startswith("human_ai_probe_at_layer_"):
            continue
        if fname.endswith("_final.pth"):
            continue

        layer_idx = int(fname.split("_layer_")[-1].split(".pth")[0])

        # Filter by strategy-resolved layers
        if allowed_layers is not None and layer_idx not in allowed_layers:
            continue

        probe = LinearProbeClassification(
            device=device, probe_class=1, input_dim=input_dim, logistic=True
        )
        probe.load_state_dict(
            torch.load(os.path.join(probe_dir, fname), map_location=device)
        )
        probe.eval()
        probes[layer_idx] = probe

    if not probes:
        msg = f"No matching probes found in {probe_dir} (allowed_layers={allowed_layers})"
        if raise_if_empty:
            raise RuntimeError(msg)
        print(f"  [WARNING] {msg}")
    else:
        print(f"Loaded {len(probes)} {label} from {probe_dir}")
        print(f"  Layer indices: {sorted(probes.keys())}")

    return probes


# ========================== INTERVENTION UTILS ========================== #

def optimize_one_inter_rep(inter_rep, cf_target, probe, N):
    tensor = inter_rep.to(TORCH_DEVICE).to(torch.float32)
    w = probe.proj[0].weight.to(TORCH_DEVICE).to(torch.float32)
    target = cf_target.to(TORCH_DEVICE).view(1, -1).to(torch.float32)
    return tensor + N * (target @ w)


def make_edit_fn(probes_by_layer, cf_target, N):
    def edit_fn(output, layer_name):
        if "model.layers." not in layer_name:
            return output
        try:
            layer_num = int(layer_name.split("model.layers.")[-1].split(".")[0])
        except ValueError:
            return output
        probe_idx = layer_num + 1
        if probe_idx not in probes_by_layer:
            return output
        probe = probes_by_layer[probe_idx]
        hidden = output[0]
        if hidden.ndim != 3:
            return output
        last_tok = hidden[:, -1, :]
        updated = torch.cat([
            optimize_one_inter_rep(last_tok[i:i+1, :], cf_target, probe, N=N)
            for i in range(last_tok.size(0))
        ], dim=0).to(hidden.dtype)
        hidden[:, -1, :] = updated
        output = list(output)
        output[0] = hidden
        return tuple(output)
    return edit_fn


def get_layer_names_from_probes(model, probes_by_layer):
    """Derive intervention layer names from whichever probes were loaded.

    Probe keys use hidden_states indexing (layer_num + 1), so we map back:
    probe_idx corresponds to model.layers.{probe_idx - 1}.
    """
    layer_names = []
    for name, _ in model.named_modules():
        if name.startswith("model.layers.") and name.split(".")[-1].isdigit():
            layer_num = int(name.split("model.layers.")[-1])
            probe_idx = layer_num + 1  # hidden_states offset
            if probe_idx in probes_by_layer:
                layer_names.append(name)
    return layer_names


def null_edit(output, layer_name):
    return output


def load_model_and_tokenizer():
    print("Loading LLaMA-2-Chat-13B model and tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME, local_files_only=True, padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME, local_files_only=True)
    model.half().to(TORCH_DEVICE).eval()
    if "<pad>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded.")
    return model, tokenizer


# ========================== PROBE RESOLUTION ========================== #

def resolve_probe_configs_for_strategy(strategy_name: str, print_fn=print):
    """Load all probe sets and resolve layer_mode filters for a given strategy.

    Returns list of dicts: {"label", "probes", "config", "strategy_layers"}
    """
    print_fn(f"\n{'='*60}")
    print_fn(f"  Layer strategy: {strategy_name}")
    print_fn(f"  {LAYER_STRATEGIES[strategy_name]['description']}")
    print_fn(f"{'='*60}")

    # Step 1: Load accuracy lookups for both probe types
    ctrl_acc = load_accuracy_lookup(CONTROL_PROBE_DIR)
    read_acc = load_accuracy_lookup(READING_PROBE_DIR)

    # Step 2: Resolve strategy layers for control probes
    ctrl_strategy_layers = resolve_strategy_layers(
        strategy_name, ctrl_acc, print_fn=print_fn
    )

    # Step 3: Resolve strategy layers for reading probes
    read_strategy_layers = resolve_strategy_layers(
        strategy_name, read_acc, print_fn=print_fn
    )

    # Step 4: Load and filter probes per config
    control_probes = load_probes(
        CONTROL_PROBE_DIR, DEVICE,
        allowed_layers=ctrl_strategy_layers,
        label="control_probes"
    )
    control_layers = set(control_probes.keys()) if control_probes else set()
    print_fn(f"\n  Control probe layers for '{strategy_name}': {sorted(control_layers)}")

    resolved = []
    for pcfg in PROBE_CONFIGS:
        label = pcfg["label"]
        mode = pcfg["layer_mode"]

        # Load probes with strategy filtering
        if pcfg["probe_dir"] == CONTROL_PROBE_DIR:
            probes = dict(control_probes)  # reuse
        else:
            probes = load_probes(
                pcfg["probe_dir"], DEVICE,
                allowed_layers=read_strategy_layers,
                label=label
            )

        if not probes:
            print_fn(f"  [SKIP] No probes loaded for {label}")
            continue

        # Apply layer_mode filter (on top of strategy filter)
        if mode == "matched":
            if not control_layers:
                print_fn(f"  [SKIP] {label}: no control layers to match against")
                continue
            probes = {k: v for k, v in probes.items() if k in control_layers}
            print_fn(f"  {label} (matched): keeping layers {sorted(probes.keys())}")
        elif mode == "exclude":
            probes = {k: v for k, v in probes.items() if k not in control_layers}
            print_fn(f"  {label} (exclude): keeping layers {sorted(probes.keys())}")
        else:  # "all"
            print_fn(f"  {label} (all): using layers {sorted(probes.keys())}")

        if not probes:
            print_fn(f"  [SKIP] {label}: no layers remain after filtering")
            continue

        resolved.append({
            "label": label,
            "probes": probes,
            "config": pcfg,
            "strategy_name": strategy_name,
            "strategy_layers_ctrl": sorted(ctrl_strategy_layers),
            "strategy_layers_read": sorted(read_strategy_layers),
        })

    return resolved


# ============================================================================ #
#                     V1: SINGLE-PROMPT GENERATION                             #
# ============================================================================ #

def v1_collect_responses_batched(model, tokenizer, prompts, modified_layer_names,
                                  edit_function, batch_size=5):
    print(f"Intervening on layers: {modified_layer_names}")
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i+batch_size]
        formatted = [llama_v2_prompt([
            {"role": "system", "content": "You are a helpful, respectful and honest assistant."},
            {"role": "user", "content": p},
        ]) for p in batch]
        with TraceDict(model, modified_layer_names, edit_output=edit_function) as _:
            with torch.no_grad():
                inputs = tokenizer(
                    formatted, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048
                ).to(TORCH_DEVICE)
                tokens = model.generate(
                    **inputs, max_new_tokens=V1_MAX_NEW_TOKENS,
                    do_sample=V1_GEN_DO_SAMPLE,
                    temperature=V1_GEN_TEMPERATURE, top_p=V1_GEN_TOP_P,
                )
        for seq in tokens:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            if "[/INST]" in text:
                text = text.split("[/INST]", 1)[1]
            responses.append(text.strip())
    return responses


def v1_save(result_dir, probe_label, strategy_name, N, questions, baseline,
            human_resp, ai_resp, probes_by_layer, modified_layer_names,
            strategy_info):
    os.makedirs(result_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(result_dir, "intervention_responses.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "question_idx", "question", "condition", "response",
        ])
        w.writeheader()
        for idx, q in enumerate(questions):
            for cond, resps in [("baseline", baseline), ("human", human_resp), ("ai", ai_resp)]:
                w.writerow({
                    "question_idx": idx, "question": q,
                    "condition": cond, "response": resps[idx],
                })

    # Config JSON
    gen_config = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp2b_naturalistic",
        "probe_type": probe_label,
        "layer_strategy": strategy_name,
        "layer_strategy_description": LAYER_STRATEGIES[strategy_name]["description"],
        "config": {
            "model": "LLaMA-2-Chat-13B",
            "model_path": cfg.MODEL_NAME,
            "layer_selection": f"automatic ({strategy_name} strategy)",
            "intervention_strength": N,
            "min_probe_accuracy": LAYER_STRATEGIES[strategy_name]["min_accuracy"],
            "probe_layers_used": sorted(probes_by_layer.keys()),
            "n_probe_layers": len(probes_by_layer),
            "modified_layer_names": modified_layer_names,
            "max_new_tokens": V1_MAX_NEW_TOKENS,
            "gen_temperature": V1_GEN_TEMPERATURE,
            "gen_top_p": V1_GEN_TOP_P,
            "gen_do_sample": V1_GEN_DO_SAMPLE,
            "n_questions": len(questions),
        },
        "strategy_info": strategy_info,
    }
    with open(os.path.join(result_dir, "generation_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)

    # TXT examples
    with open(os.path.join(result_dir, "human_ai_causal_examples.txt"), "w") as f:
        f.write(f"Strategy: {strategy_name} | Probe type: {probe_label} | N = {N}\n")
        f.write(f"Active layers: {sorted(probes_by_layer.keys())} ({len(probes_by_layer)} layers)\n")
        f.write(f"Temp = {V1_GEN_TEMPERATURE} | Questions: {len(questions)}\n")
        f.write("=" * 80 + "\n\n")
        for i in range(len(questions)):
            f.write(f"Q{i}: {questions[i]}\n\n")
            f.write("-" * 50 + f"\nBaseline:\nASSISTANT: {baseline[i]}\n\n")
            f.write("-" * 50 + f"\nHuman-steered:\nASSISTANT: {human_resp[i]}\n\n")
            f.write("-" * 50 + f"\nAI-steered:\nASSISTANT: {ai_resp[i]}\n\n")
            f.write("=" * 80 + "\n\n")
    print(f"  Saved to {result_dir}")


def main_v1(strengths, layer_strategies):
    os.makedirs(RESULT_DIR_V1, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()

    with open(CAUSAL_QUESTION_PATH, "r") as f:
        questions = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(questions)} causal test prompts.")

    # Baseline is shared across all strategies (no intervention)
    print("\n=== Generating baseline responses ===")
    baseline = v1_collect_responses_batched(model, tokenizer, questions, [], null_edit)

    for strategy_name in layer_strategies:
        resolved = resolve_probe_configs_for_strategy(strategy_name)

        for entry in resolved:
            probe_label = entry["label"]
            probes_by_layer = entry["probes"]
            modified_layer_names = get_layer_names_from_probes(model, probes_by_layer)

            strategy_info = {
                "ctrl_layers": entry["strategy_layers_ctrl"],
                "read_layers": entry["strategy_layers_read"],
            }

            for N in strengths:
                print(f"\n{'='*60}")
                print(f"  {strategy_name} | {probe_label} | N = {N}")
                print(f"  Active layers: {sorted(probes_by_layer.keys())} "
                      f"({len(probes_by_layer)} layers)")
                print(f"{'='*60}")

                result_dir = os.path.join(
                    RESULT_DIR_V1, strategy_name, probe_label, f"is_{N}"
                )

                # Skip if already completed
                if os.path.isfile(os.path.join(result_dir, "generation_config.json")):
                    print(f"  [SKIP] Already exists: {result_dir}")
                    continue

                print(f"\n=== Human-steered ({probe_label}, N={N}) ===")
                human_resp = v1_collect_responses_batched(
                    model, tokenizer, questions, modified_layer_names,
                    make_edit_fn(probes_by_layer, torch.tensor([1.0]), N),
                )

                print(f"\n=== AI-steered ({probe_label}, N={N}) ===")
                ai_resp = v1_collect_responses_batched(
                    model, tokenizer, questions, modified_layer_names,
                    make_edit_fn(probes_by_layer, torch.tensor([-1.0]), N),
                )

                v1_save(
                    result_dir, probe_label, strategy_name, N,
                    questions, baseline, human_resp, ai_resp,
                    probes_by_layer, modified_layer_names, strategy_info,
                )

    print("\nV1 generation complete.")


# ============================================================================ #
#                     V2: EXP1 RECREATION GENERATION                           #
# ============================================================================ #

def log_message(msg, log_file=LOG_FILE):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def load_prompt_text(prompts_dir, topic):
    p = prompts_dir / f"{topic}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip(), p.name


def truncate_history(history, keep_pairs):
    systems = [m for m in history if m["role"] == "system"]
    others = [m for m in history if m["role"] != "system"]
    return systems + (others[-2*keep_pairs:] if keep_pairs > 0 else others[-2:])


def clean_meta_narration(text):
    if not isinstance(text, str):
        return text
    c = text.strip()
    c = re.sub(
        r'^.*?(?:Here\'s my (?:first message|response|next message|next question|first response))[:\!]*\s*\n+',
        '', c, flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    c = re.sub(
        r'^Sure,?\s+I\'d (?:be happy|love) to\s+\w+(?:\s+\w+){0,10}[!.]\s*\n*',
        '', c, flags=re.IGNORECASE,
    ).strip()
    if c.startswith('""') and c.endswith('""'):
        c = c[2:-2].strip()
    elif c.startswith('"') and c.endswith('"') and c.count('"') == 2:
        c = c[1:-1].strip()
    elif c.startswith('"'):
        c = c[1:].strip()
        if c.endswith('"'):
            c = c[:-1].strip()
    return c


def v2_generate_single_turn(model, tokenizer, messages, modified_layer_names, edit_function):
    formatted = llama_v2_prompt(messages)
    ctx = (
        TraceDict(model, modified_layer_names, edit_output=edit_function)
        if modified_layer_names
        else nullcontext()
    )
    with ctx:
        with torch.no_grad():
            inputs = tokenizer(
                formatted, return_tensors="pt", truncation=True, max_length=2048,
            ).to(TORCH_DEVICE)
            tokens = model.generate(
                **inputs, max_new_tokens=V2_MAX_NEW_TOKENS,
                do_sample=(V2_GEN_TEMPERATURE > 0),
                temperature=V2_GEN_TEMPERATURE if V2_GEN_TEMPERATURE > 0 else None,
                top_p=V2_GEN_TOP_P,
            )
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    if "[/INST]" in text:
        text = text.rsplit("[/INST]", 1)[-1]
    if "[INST]" in text:
        text = text.split("[INST]")[0].strip()
    return clean_meta_narration(text.strip())


def run_topic_dialogue_steered(model, tokenizer, topic_text, modified_layer_names, edit_function):
    topic_intro = (
        f"The conversation topic is: '{topic_text}'.\n\n"
        f"Please begin by producing only your first message to start the conversation.\n"
        f"Do not simulate both sides of the dialogue."
    )
    sub_hist = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": topic_intro},
    ]
    llm_hist = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": topic_intro},
    ]
    exchanges, pair_index = [], 1

    sub_msg = v2_generate_single_turn(
        model, tokenizer,
        truncate_history(sub_hist, HISTORY_PAIRS),
        modified_layer_names, edit_function,
    )
    sub_hist.append({"role": "assistant", "content": sub_msg})
    llm_hist.append({"role": "user", "content": sub_msg})

    while pair_index <= PAIRS_TOTAL:
        llm_msg = v2_generate_single_turn(
            model, tokenizer,
            truncate_history(llm_hist, HISTORY_PAIRS),
            [], null_edit,
        )
        llm_hist.append({"role": "assistant", "content": llm_msg})
        exchanges.append({
            "pair_index": pair_index,
            "transcript_sub": sub_msg,
            "transcript_llm": llm_msg,
        })
        if pair_index == PAIRS_TOTAL:
            break
        sub_hist.append({"role": "user", "content": f"Partner: {llm_msg}"})
        sub_msg = v2_generate_single_turn(
            model, tokenizer,
            truncate_history(sub_hist, HISTORY_PAIRS),
            modified_layer_names, edit_function,
        )
        sub_hist.append({"role": "assistant", "content": sub_msg})
        llm_hist.append({"role": "user", "content": sub_msg})
        pair_index += 1

    return exchanges


def v2_run_subject_for_probe_and_strength(model, tokenizer, df_config, probes_by_layer,
                                           modified_layer_names, probe_label, strategy_name,
                                           N, subject_id):
    result_dir = os.path.join(
        RESULT_DIR_V2, strategy_name, probe_label, f"is_{N}", "per_subject"
    )
    os.makedirs(result_dir, exist_ok=True)
    out_csv = Path(result_dir) / f"{subject_id}.csv"

    # Skip if already completed
    if out_csv.exists() and out_csv.stat().st_size > 0:
        log_message(f"  [SKIP] Already exists: {out_csv}")
        return out_csv

    edit_functions = {
        "baseline": null_edit,
        "human": make_edit_fn(probes_by_layer, torch.tensor([1.0]), N),
        "ai": make_edit_fn(probes_by_layer, torch.tensor([-1.0]), N),
    }
    layer_names = {
        "baseline": [],
        "human": modified_layer_names,
        "ai": modified_layer_names,
    }
    fieldnames = [
        "subject", "run", "order", "trial", "condition", "topic", "topic_file",
        "pair_index", "transcript_sub", "transcript_llm",
    ]
    total_convos = len(df_config) * len(CONDITIONS)
    convo_count = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, row in df_config.iterrows():
            run_num, order, trial = int(row["run"]), int(row["order"]), int(row["trial"])
            topic = str(row["topic"]).strip()
            topic_text, topic_file = load_prompt_text(PROMPTS_DIR, topic)
            for condition in CONDITIONS:
                convo_count += 1
                log_message(
                    f"[{subject_id}] {strategy_name}/{probe_label} N={N} | "
                    f"Conv {convo_count}/{total_convos} | "
                    f"run={run_num} order={order} topic={topic} [{condition}]"
                )
                try:
                    for ex in run_topic_dialogue_steered(
                        model, tokenizer, topic_text,
                        layer_names[condition], edit_functions[condition],
                    ):
                        writer.writerow({
                            "subject": subject_id, "run": run_num, "order": order,
                            "trial": trial, "condition": condition, "topic": topic,
                            "topic_file": topic_file, "pair_index": ex["pair_index"],
                            "transcript_sub": ex["transcript_sub"],
                            "transcript_llm": ex["transcript_llm"],
                        })
                except Exception as e:
                    log_message(
                        f"[ERROR] {subject_id} | topic={topic} cond={condition}: {e}"
                    )
        f.flush()
    log_message(f"  Wrote {out_csv} ({convo_count} conversations)")
    return out_csv


def main_v2(subject_idx, strengths, layer_strategies):
    start_time = time.time()
    os.makedirs(RESULT_DIR_V2, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    subject_id = f"s{subject_idx + 1:03d}"
    log_message(f"\n=== [START] Subject {subject_id} ===")

    config_path = CONFIG_DIR / f"conds_{subject_id}.csv"
    if not config_path.exists():
        log_message(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    import pandas as pd
    df_config = pd.read_csv(config_path, encoding="utf-8").reset_index(drop=True)
    df_config["trial"] = df_config.index + 1
    log_message(f"Loaded {len(df_config)} trials from {config_path}")

    model, tokenizer = load_model_and_tokenizer()

    for strategy_name in layer_strategies:
        resolved = resolve_probe_configs_for_strategy(strategy_name, print_fn=log_message)
        if not resolved:
            log_message(f"[WARNING] No probes resolved for strategy '{strategy_name}'. Skipping.")
            continue

        for entry in resolved:
            probe_label = entry["label"]
            probes_by_layer = entry["probes"]
            modified_layer_names = get_layer_names_from_probes(model, probes_by_layer)
            log_message(
                f"\n{strategy_name}/{probe_label}: "
                f"active probe layers = {sorted(probes_by_layer.keys())} "
                f"({len(probes_by_layer)} layers)"
            )

            for N in strengths:
                log_message(
                    f"\n{'='*60}\n"
                    f"  {strategy_name} | {probe_label} | N = {N} | Subject {subject_id}\n"
                    f"{'='*60}"
                )
                v2_run_subject_for_probe_and_strength(
                    model, tokenizer, df_config, probes_by_layer,
                    modified_layer_names, probe_label, strategy_name, N, subject_id,
                )

    log_message(f"=== [END] {subject_id} | {(time.time() - start_time) / 60:.1f} min ===\n")


# ============================================================================ #
#                          CLI ENTRY POINT                                     #
# ============================================================================ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Causal intervention generation (V1 or V2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Layer strategies:
  narrow   : Best contiguous 10-layer window (auto-selected), Viegas-style
  wide     : All layers >= 0.70 accuracy
  peak_15  : Top 15 layers by accuracy (non-contiguous)
  all_70   : Same as 'wide' (all layers >= 0.70)

Examples:
  python 2_causality_generate.py --version labels --mode V1 --layer_strategy narrow
  python 2_causality_generate.py --version labels --mode V1 --layer_strategy narrow wide --strengths 4 8
  python 2_causality_generate.py --version labels --mode V2 --layer_strategy narrow --subject_idx 0
        """,
    )
    add_version_argument(p)
    p.add_argument("--mode", type=str, required=True, choices=["V1", "V2"],
                   help="V1 = single-turn test questions; V2 = multi-turn Exp 1 recreation.")
    p.add_argument(
        "--layer_strategy", type=str, nargs="+", default=["narrow"],
        choices=ALL_STRATEGIES,
        help=f"Layer selection strategies to run. Choices: {ALL_STRATEGIES}. "
             f"Multiple allowed. Default: narrow",
    )
    p.add_argument(
        "--subject_idx", type=int, default=None,
        help="(V2) Subject index, 0-based. Falls back to $SLURM_ARRAY_TASK_ID, then 0.",
    )
    p.add_argument(
        "--strengths", type=int, nargs="+", default=None,
        help=f"Intervention strengths to sweep (default: {DEFAULT_STRENGTHS}).",
    )
    p.add_argument(
        "--strength", type=int, default=None,
        help="Single intervention strength (for SLURM). Overrides --strengths.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Force regeneration even if output files already exist.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_version(args.version)
    _init_paths()

    # Resolve strengths
    if args.strength is not None:
        strengths = [args.strength]
    elif args.strengths is not None:
        strengths = args.strengths
    else:
        strengths = DEFAULT_STRENGTHS

    layer_strategies = args.layer_strategy

    print(f"\n{'#'*60}")
    print(f"  Experiment 2b Causal Intervention Generation")
    print(f"  Data version: {args.version}")
    print(f"  Mode:       {args.mode}")
    print(f"  Strategies: {layer_strategies}")
    print(f"  Strengths:  {strengths}")
    print(f"  Probe dir:  {cfg.PATHS.probe_checkpoints}")
    print(f"{'#'*60}\n")

    if args.mode == "V1":
        main_v1(strengths, layer_strategies)
    elif args.mode == "V2":
        idx = (
            args.subject_idx
            if args.subject_idx is not None
            else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        )
        main_v2(idx, strengths, layer_strategies)


if __name__ == "__main__":
    main()