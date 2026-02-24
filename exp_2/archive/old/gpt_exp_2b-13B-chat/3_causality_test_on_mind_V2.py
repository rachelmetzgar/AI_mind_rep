#!/usr/bin/env python3
"""
Phase 4, Version 2: Steering-based recreation of Experiment 1 conversations.

Recreates Experiment 1's conversational structure using activation steering
instead of system-prompt identity manipulation. Two-agent back-and-forth:
    - Subject agent: LLaMA-2-Chat-13B with activation steering applied
    - Partner agent: LLaMA-2-Chat-13B running vanilla (no steering)

Matches Experiment 1 exactly:
    - Same per-subject condition configs (conds_sXXX.csv)
    - Same topic prompt .txt files
    - Same 4-exchange conversational structure
    - Same history truncation logic
    - Same turn-taking flow (subject starts, alternating thereafter)

The ONLY difference: subject's partner-identity belief comes from activation
steering rather than a system prompt. Neither agent receives any identity
information in their prompts.

Each subject runs all 40 topics under 3 conditions:
    baseline (no steering), human-steered, ai-steered
    = 120 conversations per subject, 480 exchanges per subject

Updated Feb 2026: FROM_LAYER=27, N=16 to match V1 causality script.
Updated Feb 2026: 4 turns (was 5), meta-narration cleaning, [INST] leak fix.

Run via SLURM array (one subject per task):
    sbatch --array=0-49 3b_causality_exp1_recreation.sh

Env: llama2_env

Rachel C. Metzgar · Feb 2026
"""

import os
import re
import sys
import csv
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from contextlib import nullcontext

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Local imports ---
from src.dataset import llama_v2_prompt
from src.probes import LinearProbeClassification


# Inline TraceDict to avoid baukit's torchvision dependency
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

CONTROL_PROBE_DIR = "data/probe_checkpoints/control_probe"
RESULT_DIR = "data/intervention_results/V2"

# Experiment 1 paths (read-only references)
EXP1_ROOT = Path("/jukebox/graziano/rachel/ai_mind_rep/exp_1/code/data_gen")
PROMPTS_DIR = EXP1_ROOT / "utils" / "prompts"
CONFIG_DIR = EXP1_ROOT / "utils" / "config"

INPUT_DIM = 5120  # LLaMA-2-13B hidden size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = DEVICE

# Intervention hyperparameters — matched to V1
# Layers 28-30 reliable (0.770, 0.797, 0.800), 33-39 all >= 0.782
# Collapsed layers (24, 27, 31, 32) filtered out by MIN_PROBE_ACCURACY
FROM_LAYER = 27     # inclusive (model.layers.27 -> probe index 28)
TO_LAYER = 39       # exclusive (model.layers.38 -> probe index 39)
INTERVENTION_STRENGTH = 16  # matched to V1; 13B-chat RLHF needs strong steering
USE_RESIDUAL_STREAM = True

# Minimum control probe accuracy to use for intervention
MIN_PROBE_ACCURACY = 0.70

# Control probe best-validation accuracies from Exp 2b training
CONTROL_PROBE_BEST_ACC = {
    0: 0.500, 1: 0.595, 2: 0.500, 3: 0.603, 4: 0.593, 5: 0.598,
    6: 0.613, 7: 0.605, 8: 0.623, 9: 0.625, 10: 0.650, 11: 0.500,
    12: 0.500, 13: 0.662, 14: 0.500, 15: 0.705, 16: 0.690, 17: 0.695,
    18: 0.695, 19: 0.730, 20: 0.740, 21: 0.735, 22: 0.767, 23: 0.757,
    24: 0.500, 25: 0.772, 26: 0.775, 27: 0.500, 28: 0.770, 29: 0.797,
    30: 0.800, 31: 0.500, 32: 0.588, 33: 0.802, 34: 0.807, 35: 0.807,
    36: 0.810, 37: 0.792, 38: 0.825, 39: 0.800, 40: 0.608,
}

# Generation hyperparameters
MAX_NEW_TOKENS = 500   # match Experiment 1
GEN_TEMPERATURE = 0.8  # match Experiment 1
GEN_TOP_P = 1.0

# Conversation structure
PAIRS_TOTAL = 4       # 4 exchange pairs (turn 5 degenerates — [INST] leakage + repetition)
HISTORY_PAIRS = 4     # keep last 4 pairs in context window

# Conditions to generate per topic
CONDITIONS = ["baseline", "human", "ai"]

# Neutral system prompt (same as Experiment 1's partner prompt)
# NO identity information for either agent
SYSTEM_PROMPT = "You are engaging in a real-time spoken conversation."

# Get subject index from CLI (SLURM array compatibility)
if len(sys.argv) > 1:
    SUBJ_IDX = int(sys.argv[1])
else:
    SUBJ_IDX = 0

SUBJECT_ID = f"s{SUBJ_IDX + 1:03d}"

# Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "steering_generation_progress.log"


# ========================== HELPERS ========================== #

def log_message(msg: str, log_file: Path = LOG_FILE):
    """Append a timestamped message to the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_prompt_text(prompts_dir: Path, topic: str) -> Tuple[str, str]:
    """Load topic prompt from .txt file. Returns (text, filename)."""
    p = prompts_dir / f"{topic}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip(), p.name


def truncate_history(
    history: List[Dict[str, str]], keep_pairs: int
) -> List[Dict[str, str]]:
    """Keep all system messages + last N user/assistant pairs."""
    systems = [m for m in history if m["role"] == "system"]
    others = [m for m in history if m["role"] != "system"]
    if keep_pairs > 0:
        return systems + others[-2 * keep_pairs:]
    else:
        return systems + others[-2:]


def clean_meta_narration(text):
    """
    Strip LLM meta-narration preambles from generated conversation turns.

    LLaMA-2-Chat often produces framing text like:
        "Sure! Here's my first message:\\n\\n\\"Hey there!..."
        "Great! Here's my response:\\n\\n\\"I completely agree..."
        "Sure, I'd be happy to talk about my favorite car!..."

    This contaminates linguistic measures (inflates word count, adds
    discourse markers that aren't part of the actual conversation).
    """
    if not isinstance(text, str):
        return text

    cleaned = text.strip()

    # Broad pattern: anything ending with "Here's my [type]:" + newlines
    cleaned = re.sub(
        r'^.*?(?:Here\'s my (?:first message|response|next message|'
        r'next question|first response))[:\!]*\s*\n+',
        '', cleaned, flags=re.IGNORECASE | re.DOTALL
    ).strip()

    # "Sure, I'd be happy to [verb] [words]! [actual content]"
    cleaned = re.sub(
        r'^Sure,?\s+I\'d (?:be happy|love) to\s+\w+(?:\s+\w+){0,10}[!.]\s*\n*',
        '', cleaned, flags=re.IGNORECASE
    ).strip()

    # Strip wrapping escaped quotes: ""content"" -> content
    if cleaned.startswith('""') and cleaned.endswith('""'):
        cleaned = cleaned[2:-2].strip()
    elif cleaned.startswith('"') and cleaned.endswith('"') and cleaned.count('"') == 2:
        cleaned = cleaned[1:-1].strip()
    # Strip leading quote left over after preamble removal
    elif cleaned.startswith('"'):
        cleaned = cleaned[1:].strip()
        if cleaned.endswith('"'):
            cleaned = cleaned[:-1].strip()

    return cleaned


# ========================== PROBE LOADING ========================== #

def load_partner_control_probes(
    probe_dir,
    device,
    input_dim=INPUT_DIM,
    min_accuracy=MIN_PROBE_ACCURACY,
    accuracy_lookup=None,
):
    """
    Load control probes, skipping any below min_accuracy threshold.
    """
    probes = {}
    for fname in os.listdir(probe_dir):
        if not fname.endswith(".pth") or not fname.startswith("human_ai_probe_at_layer_"):
            continue
        if fname.endswith("_final.pth"):
            continue
        layer_str = fname.split("_layer_")[-1].split(".pth")[0]
        layer_idx = int(layer_str)

        if accuracy_lookup and layer_idx in accuracy_lookup:
            acc = accuracy_lookup[layer_idx]
            if acc < min_accuracy:
                log_message(f"  Skipping layer {layer_idx} probe (acc={acc:.3f} < {min_accuracy})")
                continue

        probe = LinearProbeClassification(
            device=device, probe_class=1, input_dim=input_dim, logistic=True,
        )
        state = torch.load(os.path.join(probe_dir, fname), map_location=device)
        probe.load_state_dict(state)
        probe.eval()
        probes[layer_idx] = probe
    if not probes:
        raise RuntimeError(f"No matching probes found in {probe_dir}")
    log_message(f"Loaded {len(probes)} control probes from {probe_dir}")
    log_message(f"  Layer indices: {sorted(probes.keys())}")
    return probes


# ========================== INTERVENTION UTILS ========================== #

def optimize_one_inter_rep(inter_rep, cf_target, probe, N=INTERVENTION_STRENGTH):
    tensor = inter_rep.to(TORCH_DEVICE).to(torch.float32)
    w = probe.proj[0].weight.to(TORCH_DEVICE).to(torch.float32)
    target = cf_target.to(TORCH_DEVICE).view(1, -1).to(torch.float32)
    direction = target @ w
    return tensor + N * direction


def make_edit_fn(probes_by_layer, cf_target, N):
    def edit_fn(output, layer_name):
        if "model.layers." not in layer_name:
            return output
        layer_str = layer_name.split("model.layers.")[-1].split(".")[0]
        try:
            layer_num = int(layer_str)
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
        updated_batch = []
        for i in range(last_tok.size(0)):
            cur = last_tok[i:i+1, :]
            updated = optimize_one_inter_rep(cur, cf_target, probe, N=N)
            updated_batch.append(updated)
        updated_batch = torch.cat(updated_batch, dim=0).to(hidden.dtype)
        hidden[:, -1, :] = updated_batch
        output = list(output)
        output[0] = hidden
        return tuple(output)
    return edit_fn


def get_modified_layer_names(model, from_idx, to_idx):
    which_layers = []
    for name, module in model.named_modules():
        if name.startswith("model.layers.") and name.split(".")[-1].isdigit():
            layer_num = int(name.split("model.layers.")[-1])
            if from_idx <= layer_num < to_idx:
                which_layers.append(name)
    return which_layers


# ========================== GENERATION ========================== #

def generate_single_turn(
    model, tokenizer, messages,
    modified_layer_names, edit_function,
):
    formatted = llama_v2_prompt(messages)

    if modified_layer_names:
        ctx = TraceDict(model, modified_layer_names, edit_output=edit_function)
    else:
        ctx = nullcontext()

    with ctx:
        with torch.no_grad():
            inputs = tokenizer(
                formatted, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(TORCH_DEVICE)

            tokens = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=(GEN_TEMPERATURE > 0),
                temperature=GEN_TEMPERATURE if GEN_TEMPERATURE > 0 else None,
                top_p=GEN_TOP_P,
            )

    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    if "[/INST]" in text:
        text = text.rsplit("[/INST]", 1)[-1]

    # Strip [INST] leakage (model echoing back prompt format in later turns)
    if "[INST]" in text:
        text = text.split("[INST]")[0].strip()

    return clean_meta_narration(text.strip())


def run_topic_dialogue_steered(
    model, tokenizer, topic_text,
    modified_layer_names, edit_function,
):
    def null_edit(output, layer_name):
        return output

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

    exchanges = []
    pair_index = 1

    sub_input = truncate_history(sub_hist, HISTORY_PAIRS)
    sub_msg = generate_single_turn(
        model, tokenizer, sub_input,
        modified_layer_names, edit_function,
    )
    sub_hist.append({"role": "assistant", "content": sub_msg})
    llm_hist.append({"role": "user", "content": sub_msg})

    while pair_index <= PAIRS_TOTAL:
        llm_input = truncate_history(llm_hist, HISTORY_PAIRS)
        llm_msg = generate_single_turn(
            model, tokenizer, llm_input, [], null_edit,
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
        sub_input = truncate_history(sub_hist, HISTORY_PAIRS)
        sub_msg = generate_single_turn(
            model, tokenizer, sub_input,
            modified_layer_names, edit_function,
        )
        sub_hist.append({"role": "assistant", "content": sub_msg})
        llm_hist.append({"role": "user", "content": sub_msg})

        pair_index += 1

    return exchanges


# ========================== MAIN ========================== #

def main():
    start_time = time.time()
    os.makedirs(RESULT_DIR, exist_ok=True)

    log_message(f"\n=== [START] Subject {SUBJECT_ID} ===")

    config_path = CONFIG_DIR / f"conds_{SUBJECT_ID}.csv"
    if not config_path.exists():
        log_message(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    import pandas as pd
    df_config = pd.read_csv(config_path, encoding="utf-8").reset_index(drop=True)
    df_config["trial"] = df_config.index + 1
    log_message(f"Loaded {len(df_config)} trials from {config_path}")

    log_message("Loading LLaMA-2-Chat-13B...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=True, padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(TORCH_DEVICE).eval()

    if "<pad>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    log_message("Model loaded.")

    probes_by_layer = load_partner_control_probes(
        CONTROL_PROBE_DIR, DEVICE, input_dim=INPUT_DIM,
        min_accuracy=MIN_PROBE_ACCURACY,
        accuracy_lookup=CONTROL_PROBE_BEST_ACC,
    )

    modified_layer_names = get_modified_layer_names(model, FROM_LAYER, TO_LAYER)

    log_message(f"Intervention config:")
    log_message(f"  Layer range: model.layers.{FROM_LAYER} to model.layers.{TO_LAYER-1}")
    log_message(f"  Probe indices used: {sorted(k for k in probes_by_layer if FROM_LAYER < k <= TO_LAYER)}")
    log_message(f"  N = {INTERVENTION_STRENGTH}")

    def null_edit(output, layer_name):
        return output

    edit_functions = {
        "baseline": null_edit,
        "human": make_edit_fn(probes_by_layer, torch.tensor([1.0]), INTERVENTION_STRENGTH),
        "ai": make_edit_fn(probes_by_layer, torch.tensor([-1.0]), INTERVENTION_STRENGTH),
    }
    layer_names_per_condition = {
        "baseline": [],
        "human": modified_layer_names,
        "ai": modified_layer_names,
    }

    out_dir = Path(RESULT_DIR) / "per_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{SUBJECT_ID}.csv"

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

            topic_text, topic_file = load_prompt_text(PROMPTS_DIR, topic)

            for condition in CONDITIONS:
                convo_count += 1
                log_message(
                    f"[{SUBJECT_ID}] Conv {convo_count}/{total_convos} | "
                    f"run={run_num} order={order} topic={topic} [{condition}]"
                )

                try:
                    exchanges = run_topic_dialogue_steered(
                        model, tokenizer, topic_text,
                        layer_names_per_condition[condition],
                        edit_functions[condition],
                    )

                    for ex in exchanges:
                        writer.writerow({
                            "subject": SUBJECT_ID,
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
                    log_message(f"[ERROR] {SUBJECT_ID} | topic={topic} cond={condition}: {e}")
                    continue

        f.flush()

    elapsed = time.time() - start_time
    log_message(
        f"=== [END] {SUBJECT_ID} | {convo_count} conversations | "
        f"{elapsed/60:.1f} min | Output: {out_csv} ===\n"
    )


if __name__ == "__main__":
    main()