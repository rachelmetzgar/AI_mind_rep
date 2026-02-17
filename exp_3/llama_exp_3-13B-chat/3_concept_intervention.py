#!/usr/bin/env python3
"""
Experiment 3, Phase 3: Concept Injection into Conversations — GENERATION ONLY.

Injects concept vectors (from per-dimension concept probes trained in Phase 2)
into conversational generation, following the same pipeline structure as Exp 2b's
causal intervention script.

Key design decisions:
    - Unit-normalized vectors — N is step size in activation-space units,
      making dose-response curves comparable across dimensions and vs Exp 2b
    - Automatic layer selection via accuracy_summary.pkl + MIN_PROBE_ACCURACY
    - One config per dimension (all passing layers)
    - Judging done separately (via existing judge script)
    - V2 multi-turn: PAIRS_TOTAL=4, HISTORY_PAIRS=4, clean_meta_narration

Two generation modes:
    --mode V1   : single-turn test questions (sweeps N)
    --mode V2   : multi-turn Exp 1 recreation (single N per SLURM task)

Accepts --dim_id to specify which concept dimension to inject.
Accepts --dim_ids to run multiple dimensions sequentially.

Output:
    data/intervention_results/V{1,2}/{dim_name}/is_{N}/...

Usage:
    python 3_concept_intervention.py --mode V1 --dim_id 7
    python 3_concept_intervention.py --mode V1 --dim_id 7 --strengths 4 8
    python 3_concept_intervention.py --mode V2 --dim_id 7 --subject_idx 0
    python 3_concept_intervention.py --mode V1 --dim_ids 1 5 7 11 13

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
from datetime import datetime
from contextlib import nullcontext

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Local imports ---
sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import llama_v2_prompt
from src.probes import LinearProbeClassification

# Import dimension registry from script 1
from importlib.util import spec_from_file_location, module_from_spec
_s1_spec = spec_from_file_location(
    "script1",
    os.path.join(os.path.dirname(__file__), "1_elicit_concept_vectors.py"),
)
_s1_mod = module_from_spec(_s1_spec)
_s1_spec.loader.exec_module(_s1_mod)
DIMENSION_REGISTRY = _s1_mod.DIMENSION_REGISTRY


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

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

CONCEPT_PROBE_ROOT = "data/concept_probes"

INPUT_DIM = 5120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = DEVICE

DEFAULT_STRENGTHS = [1, 2, 4, 8]
MIN_PROBE_ACCURACY = 0.70

# --- V1 config ---
CAUSAL_QUESTION_PATH = "data/causality_test_questions/human_ai.txt"
RESULT_DIR_V1 = "data/intervention_results/V1"
V1_MAX_NEW_TOKENS = 768
V1_GEN_TEMPERATURE = 0.7
V1_GEN_TOP_P = 0.9
V1_GEN_DO_SAMPLE = True

# --- V2 config ---
RESULT_DIR_V2 = "data/intervention_results/V2"
EXP1_ROOT = Path("/jukebox/graziano/rachel/ai_mind_rep/exp_1/code/data_gen")
PROMPTS_DIR = EXP1_ROOT / "utils" / "prompts"
CONFIG_DIR = EXP1_ROOT / "utils" / "config"
V2_MAX_NEW_TOKENS = 500
V2_GEN_TEMPERATURE = 0.8
V2_GEN_TOP_P = 1.0
PAIRS_TOTAL = 4
HISTORY_PAIRS = 4
CONDITIONS = ["baseline", "human", "ai"]
SYSTEM_PROMPT = "You are engaging in a real-time spoken conversation."
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "concept_steering_progress.log"


# ========================== PROBE LOADING ========================== #

def load_accuracy_lookup(probe_dir):
    """Load accuracy_summary.pkl and return layer->accuracy dict."""
    pkl_path = os.path.join(probe_dir, "accuracy_summary.pkl")
    if not os.path.isfile(pkl_path):
        print(f"  [WARNING] No accuracy_summary.pkl in {probe_dir} "
              f"— loading all probes unfiltered")
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


def load_concept_probes(probe_dir, device, min_accuracy=MIN_PROBE_ACCURACY,
                        label="concept"):
    """
    Load concept probes from a dimension-specific directory.
    Probes are named concept_probe_layer_{N}.pth (best checkpoint).
    Weights are unit-normalized after loading so N controls step size
    in activation-space units, making results comparable across dimensions.
    """
    probes = {}
    if not os.path.isdir(probe_dir):
        print(f"  [WARNING] Concept probe directory not found: {probe_dir}")
        return probes

    accuracy_lookup = load_accuracy_lookup(probe_dir)

    for fname in sorted(os.listdir(probe_dir)):
        if not fname.startswith("concept_probe_layer_") or not fname.endswith(".pth"):
            continue
        if fname.endswith("_final.pth"):
            continue
        layer_str = fname.split("_layer_")[-1].split(".pth")[0]
        layer_idx = int(layer_str)

        if accuracy_lookup and layer_idx in accuracy_lookup:
            acc = accuracy_lookup[layer_idx]
            if acc < min_accuracy:
                continue

        probe = LinearProbeClassification(
            device=device, probe_class=1, input_dim=INPUT_DIM, logistic=True,
        )
        probe.load_state_dict(
            torch.load(os.path.join(probe_dir, fname), map_location=device)
        )

        # Unit-normalize the weight vector
        with torch.no_grad():
            w = probe.proj[0].weight
            probe.proj[0].weight.copy_(w / (w.norm() + 1e-8))

        probe.eval()
        probes[layer_idx] = probe

    if not probes:
        print(f"  [WARNING] No passing concept probes found in {probe_dir}")
    else:
        print(f"Loaded {len(probes)} {label} (unit-normalized, "
              f"≥{min_accuracy} acc)")
        print(f"  Layer indices: {sorted(probes.keys())}")

    return probes


# ========================== INTERVENTION UTILS ========================== #

def optimize_one_inter_rep(inter_rep, cf_target, probe, N):
    """Apply steering: hidden + N * (target @ weight)."""
    tensor = inter_rep.to(TORCH_DEVICE).to(torch.float32)
    w = probe.proj[0].weight.to(TORCH_DEVICE).to(torch.float32)
    target = cf_target.to(TORCH_DEVICE).view(1, -1).to(torch.float32)
    return tensor + N * (target @ w)


def make_edit_fn(probes_by_layer, cf_target, N):
    """Create a hook-compatible edit function for TraceDict."""
    def edit_fn(output, layer_name):
        if "model.layers." not in layer_name:
            return output
        try:
            layer_num = int(
                layer_name.split("model.layers.")[-1].split(".")[0]
            )
        except ValueError:
            return output
        probe_idx = layer_num + 1  # hidden_states offset
        if probe_idx not in probes_by_layer:
            return output
        probe = probes_by_layer[probe_idx]
        hidden = output[0]
        if hidden.ndim != 3:
            return output
        last_tok = hidden[:, -1, :]
        updated = torch.cat([
            optimize_one_inter_rep(
                last_tok[i:i+1, :], cf_target, probe, N=N
            )
            for i in range(last_tok.size(0))
        ], dim=0).to(hidden.dtype)
        hidden[:, -1, :] = updated
        output = list(output)
        output[0] = hidden
        return tuple(output)
    return edit_fn


def get_layer_names_from_probes(model, probes_by_layer):
    """Derive intervention layer names from loaded probes."""
    layer_names = []
    for name, _ in model.named_modules():
        if name.startswith("model.layers.") and name.split(".")[-1].isdigit():
            layer_num = int(name.split("model.layers.")[-1])
            probe_idx = layer_num + 1
            if probe_idx in probes_by_layer:
                layer_names.append(name)
    return layer_names


def null_edit(output, layer_name):
    return output


def load_model_and_tokenizer():
    print("Loading LLaMA-2-Chat-13B model and tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=True, padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, local_files_only=True,
    )
    model.half().to(TORCH_DEVICE).eval()
    if "<pad>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded.")
    return model, tokenizer


# ============================================================================ #
#                     V1: SINGLE-PROMPT GENERATION                             #
# ============================================================================ #

def v1_collect_responses_batched(model, tokenizer, prompts,
                                 modified_layer_names, edit_function,
                                 batch_size=5):
    """Generate responses with optional steering via TraceDict hooks."""
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i+batch_size]
        formatted = [
            llama_v2_prompt([{"role": "user", "content": p}])
            for p in batch
        ]
        with TraceDict(
            model, modified_layer_names, edit_output=edit_function
        ) as _:
            with torch.no_grad():
                inputs = tokenizer(
                    formatted, return_tensors="pt", padding=True,
                    truncation=True, max_length=2048,
                ).to(TORCH_DEVICE)
                tokens = model.generate(
                    **inputs, max_new_tokens=V1_MAX_NEW_TOKENS,
                    do_sample=V1_GEN_DO_SAMPLE,
                    temperature=V1_GEN_TEMPERATURE,
                    top_p=V1_GEN_TOP_P,
                )
        for seq in tokens:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            if "[/INST]" in text:
                text = text.split("[/INST]", 1)[1]
            responses.append(text.strip())
    return responses


def v1_save(result_dir, dim_name, N, questions, baseline, human_resp,
            ai_resp, probes_by_layer, modified_layer_names):
    """Save V1 generation results: CSV, config JSON, examples TXT."""
    os.makedirs(result_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(result_dir, "intervention_responses.csv")
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

    # Config JSON
    config = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp3_concept_injection",
        "dimension": dim_name,
        "config": {
            "model": "LLaMA-2-Chat-13B",
            "vector_type": "concept_probe_unit_normalized",
            "layer_selection": "automatic (MIN_PROBE_ACCURACY filter)",
            "intervention_strength": N,
            "min_probe_accuracy": MIN_PROBE_ACCURACY,
            "probe_layers_used": sorted(probes_by_layer.keys()),
            "modified_layer_names": modified_layer_names,
            "max_new_tokens": V1_MAX_NEW_TOKENS,
            "gen_temperature": V1_GEN_TEMPERATURE,
            "gen_top_p": V1_GEN_TOP_P,
            "gen_do_sample": V1_GEN_DO_SAMPLE,
            "n_questions": len(questions),
        },
    }
    with open(os.path.join(result_dir, "generation_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # TXT examples
    txt_path = os.path.join(result_dir, "human_ai_causal_examples.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Concept injection: {dim_name}, N = {N} (unit-normalized)\n")
        f.write(f"Active layers: {sorted(probes_by_layer.keys())}\n")
        f.write(f"Questions: {len(questions)}\n" + "=" * 80 + "\n\n")
        for i in range(len(questions)):
            f.write(f"Q{i}: {questions[i]}\n\n")
            f.write("-" * 50 + "\nBaseline:\n")
            f.write(f"ASSISTANT: {baseline[i]}\n\n")
            f.write("-" * 50 + f"\nHuman-steered (N={N}):\n")
            f.write(f"ASSISTANT: {human_resp[i]}\n\n")
            f.write("-" * 50 + f"\nAI-steered (N={N}):\n")
            f.write(f"ASSISTANT: {ai_resp[i]}\n\n")
            f.write("=" * 80 + "\n\n")

    print(f"  Saved to {result_dir}")


def run_v1_for_dimension(model, tokenizer, dim_name, strengths):
    """Run V1 single-turn generation for one concept dimension."""
    os.makedirs(RESULT_DIR_V1, exist_ok=True)

    with open(CAUSAL_QUESTION_PATH, "r") as f:
        questions = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(questions)} causal test prompts.")

    # Load concept probes for this dimension
    concept_probe_dir = os.path.join(CONCEPT_PROBE_ROOT, dim_name)
    probes_by_layer = load_concept_probes(
        concept_probe_dir, DEVICE, label=dim_name,
    )
    if not probes_by_layer:
        print(f"[ERROR] No concept probes for {dim_name}. Skipping.")
        return

    modified_layer_names = get_layer_names_from_probes(model, probes_by_layer)
    print(f"Intervention layers: {modified_layer_names}")

    # Baseline (once, shared across all N values)
    print("\n=== Generating baseline responses ===")
    baseline = v1_collect_responses_batched(
        model, tokenizer, questions, [], null_edit,
    )

    for N in strengths:
        print(f"\n{'='*60}")
        print(f"  {dim_name}  |  N = {N}")
        print(f"{'='*60}")
        result_dir = os.path.join(RESULT_DIR_V1, dim_name, f"is_{N}")

        print(f"  Generating human-steered (N={N})...")
        human_resp = v1_collect_responses_batched(
            model, tokenizer, questions, modified_layer_names,
            make_edit_fn(probes_by_layer, torch.tensor([1.0]), N),
        )

        print(f"  Generating AI-steered (N={N})...")
        ai_resp = v1_collect_responses_batched(
            model, tokenizer, questions, modified_layer_names,
            make_edit_fn(probes_by_layer, torch.tensor([-1.0]), N),
        )

        v1_save(
            result_dir, dim_name, N, questions,
            baseline, human_resp, ai_resp,
            probes_by_layer, modified_layer_names,
        )

    print(f"\nV1 generation complete for {dim_name}.")


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
    return systems + (
        others[-2*keep_pairs:] if keep_pairs > 0 else others[-2:]
    )


def clean_meta_narration(text):
    """Strip meta-commentary where the model breaks character."""
    if not isinstance(text, str):
        return text
    c = text.strip()
    c = re.sub(
        r'^.*?(?:Here\'s my (?:first message|response|next message|'
        r'next question|first response))[:\!]*\s*\n+',
        '', c, flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    c = re.sub(
        r'^Sure,?\s+I\'d (?:be happy|love) to\s+\w+(?:\s+\w+){0,10}'
        r'[!.]\s*\n*',
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


def v2_generate_single_turn(model, tokenizer, messages,
                             modified_layer_names, edit_function):
    """Generate one conversational turn with optional steering."""
    formatted = llama_v2_prompt(messages)
    ctx = (
        TraceDict(model, modified_layer_names, edit_output=edit_function)
        if modified_layer_names
        else nullcontext()
    )
    with ctx:
        with torch.no_grad():
            inputs = tokenizer(
                formatted, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(TORCH_DEVICE)
            tokens = model.generate(
                **inputs, max_new_tokens=V2_MAX_NEW_TOKENS,
                do_sample=(V2_GEN_TEMPERATURE > 0),
                temperature=(
                    V2_GEN_TEMPERATURE if V2_GEN_TEMPERATURE > 0 else None
                ),
                top_p=V2_GEN_TOP_P,
            )
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    if "[/INST]" in text:
        text = text.rsplit("[/INST]", 1)[-1]
    if "[INST]" in text:
        text = text.split("[INST]")[0].strip()
    return clean_meta_narration(text.strip())


def run_topic_dialogue_steered(model, tokenizer, topic_text,
                                modified_layer_names, edit_function):
    """Run a multi-turn dialogue for one topic/condition."""
    topic_intro = (
        f"The conversation topic is: '{topic_text}'.\n\n"
        f"Please begin by producing only your first message to start "
        f"the conversation.\n"
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
    exchanges = []
    pair_index = 1

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


def run_v2_for_dimension(model, tokenizer, dim_name, subject_idx, strengths):
    """Run V2 multi-turn generation for one dimension and one subject."""
    os.makedirs(RESULT_DIR_V2, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    subject_id = f"s{subject_idx + 1:03d}"
    log_message(f"\n=== [START] Subject {subject_id}, Dimension {dim_name} ===")

    config_path = CONFIG_DIR / f"conds_{subject_id}.csv"
    if not config_path.exists():
        log_message(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    import pandas as pd
    df_config = pd.read_csv(
        config_path, encoding="utf-8",
    ).reset_index(drop=True)
    df_config["trial"] = df_config.index + 1
    log_message(f"Loaded {len(df_config)} trials from {config_path}")

    # Load concept probes
    concept_probe_dir = os.path.join(CONCEPT_PROBE_ROOT, dim_name)
    probes_by_layer = load_concept_probes(
        concept_probe_dir, DEVICE, label=dim_name,
    )
    if not probes_by_layer:
        log_message(f"[ERROR] No concept probes for {dim_name}. Exiting.")
        sys.exit(1)

    modified_layer_names = get_layer_names_from_probes(model, probes_by_layer)
    log_message(
        f"{dim_name}: active probe layers = {sorted(probes_by_layer.keys())}"
    )

    start_time = time.time()

    for N in strengths:
        log_message(
            f"\n{'='*60}\n  {dim_name}  |  N = {N}  |  "
            f"Subject {subject_id}\n{'='*60}"
        )

        result_dir = os.path.join(
            RESULT_DIR_V2, dim_name, f"is_{N}", "per_subject",
        )
        os.makedirs(result_dir, exist_ok=True)
        out_csv = Path(result_dir) / f"{subject_id}.csv"

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
            "subject", "run", "order", "trial", "condition",
            "topic", "topic_file", "pair_index",
            "transcript_sub", "transcript_llm",
        ]
        total_convos = len(df_config) * len(CONDITIONS)
        convo_count = 0

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for _, row in df_config.iterrows():
                run_num = int(row["run"])
                order = int(row["order"])
                trial = int(row["trial"])
                topic = str(row["topic"]).strip()
                topic_text, topic_file = load_prompt_text(
                    PROMPTS_DIR, topic,
                )

                for condition in CONDITIONS:
                    convo_count += 1
                    log_message(
                        f"[{subject_id}] {dim_name} N={N} | "
                        f"Conv {convo_count}/{total_convos} | "
                        f"run={run_num} order={order} "
                        f"topic={topic} [{condition}]"
                    )
                    try:
                        exchanges = run_topic_dialogue_steered(
                            model, tokenizer, topic_text,
                            layer_names[condition],
                            edit_functions[condition],
                        )
                        for ex in exchanges:
                            writer.writerow({
                                "subject": subject_id,
                                "run": run_num,
                                "order": order,
                                "trial": trial,
                                "condition": condition,
                                "topic": topic,
                                "topic_file": topic_file,
                                "pair_index": ex["pair_index"],
                                "transcript_sub": ex["transcript_sub"],
                                "transcript_llm": ex["transcript_llm"],
                            })
                    except Exception as e:
                        log_message(
                            f"[ERROR] {subject_id} | "
                            f"topic={topic} cond={condition}: {e}"
                        )
            f.flush()

        log_message(f"  Wrote {out_csv} ({convo_count} conversations)")

    log_message(
        f"=== [END] {subject_id} {dim_name} | "
        f"{(time.time() - start_time) / 60:.1f} min ===\n"
    )


# ============================================================================ #
#                          CLI ENTRY POINT                                     #
# ============================================================================ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 3: Concept injection generation (V1 or V2)."
    )
    p.add_argument(
        "--mode", type=str, required=True, choices=["V1", "V2"],
        help="V1 = single-turn + judge sweep; "
             "V2 = multi-turn Exp 1 recreation.",
    )
    p.add_argument(
        "--dim_id", type=int, default=None,
        help="Single dimension ID (1-13).",
    )
    p.add_argument(
        "--dim_ids", type=int, nargs="+", default=None,
        help="Multiple dimension IDs to run sequentially.",
    )
    p.add_argument(
        "--subject_idx", type=int, default=None,
        help="(V2) Subject index, 0-based. "
             "Falls back to $SLURM_ARRAY_TASK_ID.",
    )
    p.add_argument(
        "--strengths", type=int, nargs="+", default=None,
        help=f"Intervention strengths to sweep "
             f"(default: {DEFAULT_STRENGTHS}).",
    )
    p.add_argument(
        "--strength", type=int, default=None,
        help="Single intervention strength (overrides --strengths).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve strengths
    if args.strength is not None:
        strengths = [args.strength]
    elif args.strengths is not None:
        strengths = args.strengths
    else:
        strengths = DEFAULT_STRENGTHS

    # Resolve dimension(s)
    if args.dim_id is not None and args.dim_ids is not None:
        print("[ERROR] Specify --dim_id OR --dim_ids, not both.")
        sys.exit(1)
    if args.dim_id is not None:
        dim_ids = [args.dim_id]
    elif args.dim_ids is not None:
        dim_ids = args.dim_ids
    else:
        print("[ERROR] Must specify --dim_id or --dim_ids.")
        sys.exit(1)

    for d in dim_ids:
        if d not in DIMENSION_REGISTRY:
            print(f"[ERROR] Unknown dim_id={d}. "
                  f"Valid: {list(DIMENSION_REGISTRY.keys())}")
            sys.exit(1)

    # Load model once
    model, tokenizer = load_model_and_tokenizer()

    # Run each dimension
    for dim_id in dim_ids:
        _, dim_name = DIMENSION_REGISTRY[dim_id]
        print(f"\n{'#'*70}")
        print(f"# Dimension {dim_id}: {dim_name}")
        print(f"{'#'*70}")

        if args.mode == "V1":
            run_v1_for_dimension(model, tokenizer, dim_name, strengths)
        elif args.mode == "V2":
            idx = (
                args.subject_idx
                if args.subject_idx is not None
                else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
            )
            run_v2_for_dimension(
                model, tokenizer, dim_name, idx, strengths,
            )

    print(f"\n✅ Concept injection complete.")


if __name__ == "__main__":
    main()