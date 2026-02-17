#!/usr/bin/env python3
"""
Phase 4: Causal intervention — GENERATION ONLY.
Experiment 2b: LLaMA-2-Chat-13B with naturalistic conversation data.

Two versions, selected via --version flag:

  V1: Single-prompt causality test.
      Generates baseline/human-steered/AI-steered responses for each question.
      Saves intervention_responses.csv per probe type × strength.

  V2: Steering-based recreation of Experiment 1 conversations.
      Two-agent back-and-forth matching Exp1 structure.
      Per-subject via SLURM array. Both probe types, 3 conditions each.

Both versions sweep intervention strengths [2, 4, 8, 16] by default.

Output directory structure:
    intervention_results/V{1,2}/{control,reading}_probes/is_{N}/...

Layer selection is automatic: probes are loaded from accuracy_summary.pkl
and filtered by MIN_PROBE_ACCURACY. Intervention layers are derived from
whichever probes pass the filter — no hardcoded layer ranges needed.

Three probe configurations are run:
  - control_probes:          control probes, all layers ≥ threshold
  - reading_probes_matched:  reading probes, restricted to control probe layers
  - reading_probes_peak:     reading probes, only layers where control probes fail

Judging is done separately via 3b_causality_judge.py.

Usage:
    python 3b_causality_generate.py --version V1
    python 3b_causality_generate.py --version V1 --strengths 8 16
    python 3b_causality_generate.py --version V2 --subject_idx 0
    sbatch --array=0-49 3b_causality_generate.sh  # V2 via SLURM

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
from typing import List, Dict, Tuple, Optional
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


from src.dataset import llama_v2_prompt
from src.probes import LinearProbeClassification


# ========================== SHARED CONFIG ========================== #

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

CONTROL_PROBE_DIR = "data/probe_checkpoints/control_probe"
READING_PROBE_DIR = "data/probe_checkpoints/reading_probe"

INPUT_DIM = 5120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = DEVICE

DEFAULT_STRENGTHS = [2, 4, 8, 16]
MIN_PROBE_ACCURACY = 0.70

# Probe configurations.
# "layer_mode" controls which layers are used for intervention:
#   "all"     → use all layers that pass the accuracy filter
#   "matched" → intersect with control probe layers (apples-to-apples)
#   "exclude" → use only layers where control probes FAIL (dissociation test)
PROBE_CONFIGS = [
    {"label": "control_probes",         "probe_dir": CONTROL_PROBE_DIR, "layer_mode": "all"},
    {"label": "reading_probes_matched", "probe_dir": READING_PROBE_DIR, "layer_mode": "matched"},
    {"label": "reading_probes_peak",    "probe_dir": READING_PROBE_DIR, "layer_mode": "exclude"},
]

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
LOG_FILE = LOG_DIR / "steering_generation_progress.log"


# ========================== ACCURACY AUTO-LOADING ========================== #

def load_accuracy_lookup(probe_dir):
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


# ========================== PROBE LOADING ========================== #

def load_probes(probe_dir, device, input_dim=INPUT_DIM, min_accuracy=MIN_PROBE_ACCURACY,
                accuracy_lookup=None, label="probe", raise_if_empty=False):
    probes = {}
    if not os.path.isdir(probe_dir):
        msg = f"Probe directory not found: {probe_dir}"
        if raise_if_empty: raise RuntimeError(msg)
        print(f"  [WARNING] {msg}"); return probes
    if accuracy_lookup is None:
        accuracy_lookup = load_accuracy_lookup(probe_dir)
    for fname in os.listdir(probe_dir):
        if not fname.endswith(".pth") or not fname.startswith("human_ai_probe_at_layer_"):
            continue
        if fname.endswith("_final.pth"): continue
        layer_idx = int(fname.split("_layer_")[-1].split(".pth")[0])
        if accuracy_lookup and layer_idx in accuracy_lookup:
            acc = accuracy_lookup[layer_idx]
            if acc < min_accuracy:
                print(f"  Skipping layer {layer_idx} {label} (acc={acc:.3f} < {min_accuracy})")
                continue
        probe = LinearProbeClassification(device=device, probe_class=1, input_dim=input_dim, logistic=True)
        probe.load_state_dict(torch.load(os.path.join(probe_dir, fname), map_location=device))
        probe.eval()
        probes[layer_idx] = probe
    if not probes:
        msg = f"No matching probes found in {probe_dir}"
        if raise_if_empty: raise RuntimeError(msg)
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
        if "model.layers." not in layer_name: return output
        try: layer_num = int(layer_name.split("model.layers.")[-1].split(".")[0])
        except ValueError: return output
        probe_idx = layer_num + 1
        if probe_idx not in probes_by_layer: return output
        probe = probes_by_layer[probe_idx]
        hidden = output[0]
        if hidden.ndim != 3: return output
        last_tok = hidden[:, -1, :]
        updated = torch.cat([
            optimize_one_inter_rep(last_tok[i:i+1, :], cf_target, probe, N=N)
            for i in range(last_tok.size(0))
        ], dim=0).to(hidden.dtype)
        hidden[:, -1, :] = updated
        output = list(output); output[0] = hidden
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


def null_edit(output, layer_name): return output

def load_model_and_tokenizer():
    print("Loading LLaMA-2-Chat-13B model and tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(TORCH_DEVICE).eval()
    if "<pad>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    print("Model loaded.")
    return model, tokenizer


# ========================== PROBE RESOLUTION ========================== #

def resolve_probe_configs(print_fn=print):
    """Load all probe sets and resolve layer_mode filters.

    Returns list of dicts: {"label", "probes", "config"}
    where "probes" contains only the layers to be used for intervention.
    """
    # Step 1: Load control probes (needed to resolve "matched" / "exclude")
    control_probes = load_probes(CONTROL_PROBE_DIR, DEVICE, label="control_probes")
    control_layers = set(control_probes.keys()) if control_probes else set()
    print_fn(f"\nControl probe layers (≥{MIN_PROBE_ACCURACY}): {sorted(control_layers)}")

    resolved = []
    for pcfg in PROBE_CONFIGS:
        label = pcfg["label"]
        mode = pcfg["layer_mode"]

        # Load probes (control probes already loaded, reuse them)
        if pcfg["probe_dir"] == CONTROL_PROBE_DIR and control_probes is not None:
            probes = dict(control_probes)  # shallow copy
        else:
            probes = load_probes(pcfg["probe_dir"], DEVICE, label=label)

        if not probes:
            print_fn(f"  [SKIP] No probes loaded for {label}")
            continue

        # Apply layer_mode filter
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

        resolved.append({"label": label, "probes": probes, "config": pcfg})

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
        formatted = [llama_v2_prompt([{"role": "user", "content": p}]) for p in batch]
        with TraceDict(model, modified_layer_names, edit_output=edit_function) as _:
            with torch.no_grad():
                inputs = tokenizer(formatted, return_tensors="pt", padding=True,
                                   truncation=True, max_length=2048).to(TORCH_DEVICE)
                tokens = model.generate(**inputs, max_new_tokens=V1_MAX_NEW_TOKENS,
                                        do_sample=V1_GEN_DO_SAMPLE,
                                        temperature=V1_GEN_TEMPERATURE, top_p=V1_GEN_TOP_P)
        for seq in tokens:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            if "[/INST]" in text: text = text.split("[/INST]", 1)[1]
            responses.append(text.strip())
    return responses


def v1_save(result_dir, probe_label, N, questions, baseline, human_resp, ai_resp,
            probes_by_layer, modified_layer_names):
    os.makedirs(result_dir, exist_ok=True)
    # CSV
    csv_path = os.path.join(result_dir, "intervention_responses.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question_idx", "question", "condition", "response"])
        w.writeheader()
        for idx, q in enumerate(questions):
            for cond, resps in [("baseline", baseline), ("human", human_resp), ("ai", ai_resp)]:
                w.writerow({"question_idx": idx, "question": q, "condition": cond, "response": resps[idx]})
    # Config JSON
    config = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp2b_naturalistic", "probe_type": probe_label,
        "config": {
            "model": "LLaMA-2-Chat-13B", "model_path": MODEL_NAME,
            "layer_selection": "automatic (MIN_PROBE_ACCURACY filter)",
            "intervention_strength": N, "min_probe_accuracy": MIN_PROBE_ACCURACY,
            "probe_layers_used": sorted(probes_by_layer.keys()),
            "modified_layer_names": modified_layer_names,
            "max_new_tokens": V1_MAX_NEW_TOKENS, "gen_temperature": V1_GEN_TEMPERATURE,
            "gen_top_p": V1_GEN_TOP_P, "gen_do_sample": V1_GEN_DO_SAMPLE,
            "n_questions": len(questions),
        },
    }
    with open(os.path.join(result_dir, "generation_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    # TXT examples
    with open(os.path.join(result_dir, "human_ai_causal_examples.txt"), "w") as f:
        f.write(f"Probe type: {probe_label}, N = {N}, temp = {V1_GEN_TEMPERATURE}\n")
        f.write(f"Active layers: {sorted(probes_by_layer.keys())}\n")
        f.write(f"Questions: {len(questions)}\n" + "=" * 80 + "\n\n")
        for i in range(len(questions)):
            f.write(f"Q{i}: {questions[i]}\n\n")
            f.write("-"*50 + f"\nBaseline:\nASSISTANT: {baseline[i]}\n\n")
            f.write("-"*50 + f"\nHuman-steered:\nASSISTANT: {human_resp[i]}\n\n")
            f.write("-"*50 + f"\nAI-steered:\nASSISTANT: {ai_resp[i]}\n\n")
            f.write("=" * 80 + "\n\n")
    print(f"  Saved to {result_dir}")


def main_v1(strengths):
    os.makedirs(RESULT_DIR_V1, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()
    with open(CAUSAL_QUESTION_PATH, "r") as f:
        questions = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(questions)} causal test prompts.")

    print("\n=== Generating baseline responses ===")
    baseline = v1_collect_responses_batched(model, tokenizer, questions, [], null_edit)

    resolved = resolve_probe_configs()
    for entry in resolved:
        probe_label = entry["label"]
        probes_by_layer = entry["probes"]
        modified_layer_names = get_layer_names_from_probes(model, probes_by_layer)

        for N in strengths:
            print(f"\n{'='*60}\n  {probe_label}  |  N = {N}\n{'='*60}")
            print(f"  Active layers: {sorted(probes_by_layer.keys())}")
            result_dir = os.path.join(RESULT_DIR_V1, probe_label, f"is_{N}")

            print(f"\n=== Human-steered ({probe_label}, N={N}) ===")
            human_resp = v1_collect_responses_batched(
                model, tokenizer, questions, modified_layer_names,
                make_edit_fn(probes_by_layer, torch.tensor([1.0]), N))

            print(f"\n=== AI-steered ({probe_label}, N={N}) ===")
            ai_resp = v1_collect_responses_batched(
                model, tokenizer, questions, modified_layer_names,
                make_edit_fn(probes_by_layer, torch.tensor([-1.0]), N))

            v1_save(result_dir, probe_label, N, questions, baseline, human_resp, ai_resp,
                    probes_by_layer, modified_layer_names)
    print("\nV1 generation complete.")


# ============================================================================ #
#                     V2: EXP1 RECREATION GENERATION                           #
# ============================================================================ #

def log_message(msg, log_file=LOG_FILE):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(log_file, "a") as f: f.write(line + "\n")

def load_prompt_text(prompts_dir, topic):
    p = prompts_dir / f"{topic}.txt"
    if not p.exists(): raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip(), p.name

def truncate_history(history, keep_pairs):
    systems = [m for m in history if m["role"] == "system"]
    others = [m for m in history if m["role"] != "system"]
    return systems + (others[-2*keep_pairs:] if keep_pairs > 0 else others[-2:])

def clean_meta_narration(text):
    if not isinstance(text, str): return text
    c = text.strip()
    c = re.sub(r'^.*?(?:Here\'s my (?:first message|response|next message|next question|first response))[:\!]*\s*\n+',
               '', c, flags=re.IGNORECASE|re.DOTALL).strip()
    c = re.sub(r'^Sure,?\s+I\'d (?:be happy|love) to\s+\w+(?:\s+\w+){0,10}[!.]\s*\n*', '', c, flags=re.IGNORECASE).strip()
    if c.startswith('""') and c.endswith('""'): c = c[2:-2].strip()
    elif c.startswith('"') and c.endswith('"') and c.count('"') == 2: c = c[1:-1].strip()
    elif c.startswith('"'):
        c = c[1:].strip()
        if c.endswith('"'): c = c[:-1].strip()
    return c

def v2_generate_single_turn(model, tokenizer, messages, modified_layer_names, edit_function):
    formatted = llama_v2_prompt(messages)
    ctx = TraceDict(model, modified_layer_names, edit_output=edit_function) if modified_layer_names else nullcontext()
    with ctx:
        with torch.no_grad():
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(TORCH_DEVICE)
            tokens = model.generate(**inputs, max_new_tokens=V2_MAX_NEW_TOKENS,
                                    do_sample=(V2_GEN_TEMPERATURE > 0),
                                    temperature=V2_GEN_TEMPERATURE if V2_GEN_TEMPERATURE > 0 else None,
                                    top_p=V2_GEN_TOP_P)
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    if "[/INST]" in text: text = text.rsplit("[/INST]", 1)[-1]
    if "[INST]" in text: text = text.split("[INST]")[0].strip()
    return clean_meta_narration(text.strip())

def run_topic_dialogue_steered(model, tokenizer, topic_text, modified_layer_names, edit_function):
    topic_intro = (f"The conversation topic is: '{topic_text}'.\n\n"
                   f"Please begin by producing only your first message to start the conversation.\n"
                   f"Do not simulate both sides of the dialogue.")
    sub_hist = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": topic_intro}]
    llm_hist = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": topic_intro}]
    exchanges, pair_index = [], 1

    sub_msg = v2_generate_single_turn(model, tokenizer, truncate_history(sub_hist, HISTORY_PAIRS),
                                       modified_layer_names, edit_function)
    sub_hist.append({"role": "assistant", "content": sub_msg})
    llm_hist.append({"role": "user", "content": sub_msg})

    while pair_index <= PAIRS_TOTAL:
        llm_msg = v2_generate_single_turn(model, tokenizer, truncate_history(llm_hist, HISTORY_PAIRS), [], null_edit)
        llm_hist.append({"role": "assistant", "content": llm_msg})
        exchanges.append({"pair_index": pair_index, "transcript_sub": sub_msg, "transcript_llm": llm_msg})
        if pair_index == PAIRS_TOTAL: break
        sub_hist.append({"role": "user", "content": f"Partner: {llm_msg}"})
        sub_msg = v2_generate_single_turn(model, tokenizer, truncate_history(sub_hist, HISTORY_PAIRS),
                                           modified_layer_names, edit_function)
        sub_hist.append({"role": "assistant", "content": sub_msg})
        llm_hist.append({"role": "user", "content": sub_msg})
        pair_index += 1
    return exchanges

def v2_run_subject_for_probe_and_strength(model, tokenizer, df_config, probes_by_layer,
                                           modified_layer_names, probe_label, N, subject_id):
    result_dir = os.path.join(RESULT_DIR_V2, probe_label, f"is_{N}", "per_subject")
    os.makedirs(result_dir, exist_ok=True)
    out_csv = Path(result_dir) / f"{subject_id}.csv"

    edit_functions = {"baseline": null_edit,
                      "human": make_edit_fn(probes_by_layer, torch.tensor([1.0]), N),
                      "ai": make_edit_fn(probes_by_layer, torch.tensor([-1.0]), N)}
    layer_names = {"baseline": [], "human": modified_layer_names, "ai": modified_layer_names}
    fieldnames = ["subject","run","order","trial","condition","topic","topic_file","pair_index",
                  "transcript_sub","transcript_llm"]
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
                log_message(f"[{subject_id}] {probe_label} N={N} | Conv {convo_count}/{total_convos} | "
                           f"run={run_num} order={order} topic={topic} [{condition}]")
                try:
                    for ex in run_topic_dialogue_steered(model, tokenizer, topic_text,
                                                         layer_names[condition], edit_functions[condition]):
                        writer.writerow({"subject": subject_id, "run": run_num, "order": order,
                                        "trial": trial, "condition": condition, "topic": topic,
                                        "topic_file": topic_file, "pair_index": ex["pair_index"],
                                        "transcript_sub": ex["transcript_sub"],
                                        "transcript_llm": ex["transcript_llm"]})
                except Exception as e:
                    log_message(f"[ERROR] {subject_id} | topic={topic} cond={condition}: {e}")
        f.flush()
    log_message(f"  Wrote {out_csv} ({convo_count} conversations)")
    return out_csv

def main_v2(subject_idx, strengths):
    start_time = time.time()
    os.makedirs(RESULT_DIR_V2, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    subject_id = f"s{subject_idx + 1:03d}"
    log_message(f"\n=== [START] Subject {subject_id} ===")

    config_path = CONFIG_DIR / f"conds_{subject_id}.csv"
    if not config_path.exists():
        log_message(f"[ERROR] Config not found: {config_path}"); sys.exit(1)

    import pandas as pd
    df_config = pd.read_csv(config_path, encoding="utf-8").reset_index(drop=True)
    df_config["trial"] = df_config.index + 1
    log_message(f"Loaded {len(df_config)} trials from {config_path}")

    model, tokenizer = load_model_and_tokenizer()

    resolved = resolve_probe_configs(print_fn=log_message)
    if not resolved:
        log_message("[ERROR] No probes loaded. Exiting."); sys.exit(1)

    for entry in resolved:
        probe_label = entry["label"]
        probes_by_layer = entry["probes"]
        modified_layer_names = get_layer_names_from_probes(model, probes_by_layer)
        log_message(f"\n{probe_label}: active probe layers = {sorted(probes_by_layer.keys())}")

        for N in strengths:
            log_message(f"\n{'='*60}\n  {probe_label}  |  N = {N}  |  Subject {subject_id}\n{'='*60}")
            v2_run_subject_for_probe_and_strength(model, tokenizer, df_config, probes_by_layer,
                                                   modified_layer_names, probe_label, N, subject_id)

    log_message(f"=== [END] {subject_id} | {(time.time()-start_time)/60:.1f} min ===\n")


# ============================================================================ #
#                          CLI ENTRY POINT                                     #
# ============================================================================ #

def parse_args():
    p = argparse.ArgumentParser(description="Causal intervention generation (V1 or V2).")
    p.add_argument("--version", type=str, required=True, choices=["V1", "V2"])
    p.add_argument("--subject_idx", type=int, default=None,
                   help="(V2) Subject index, 0-based. Falls back to $SLURM_ARRAY_TASK_ID, then 0.")
    p.add_argument("--strengths", type=int, nargs="+", default=None,
                   help=f"Intervention strengths to sweep sequentially (default: {DEFAULT_STRENGTHS}).")
    p.add_argument("--strength", type=int, default=None,
                   help="Single intervention strength (for SLURM parallelization). Overrides --strengths.")
    return p.parse_args()

def main():
    args = parse_args()

    # Resolve strengths: --strength (singular) overrides --strengths (plural)
    if args.strength is not None:
        strengths = [args.strength]
    elif args.strengths is not None:
        strengths = args.strengths
    else:
        strengths = DEFAULT_STRENGTHS

    if args.version == "V1":
        main_v1(strengths)
    elif args.version == "V2":
        idx = args.subject_idx if args.subject_idx is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        main_v2(idx, strengths)

if __name__ == "__main__":
    main()