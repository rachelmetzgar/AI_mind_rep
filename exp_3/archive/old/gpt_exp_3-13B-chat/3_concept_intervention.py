#!/usr/bin/env python3
"""
Experiment 3, Phase 3: Concept Injection into Conversations

Injects the model's general "human" vs "AI" concept vector (derived from
non-conversational elicitation, NOT from conversation data) into conversational
generation to test whether the abstract concept drives the same behavioral
shifts as the conversational partner-identity representation.

Aligned with Experiment 2b causality script for direct comparison:
    - Same generation parameters (temp=0.7, top_p=0.9, sampling)
    - Same intervention strength (N=16)
    - Same GPT-4o-mini judge pipeline
    - Same inline TraceDict (no baukit dependency)
    - Same batched generation approach

Supports two steering vector sources:
    --vector_source probe   : use concept probe weight vectors
    --vector_source mean    : use mean-difference concept vectors (default)

Supports two generation modes:
    --mode v1   : single-turn test questions (same as 2b V1) + GPT judge
    --mode v2   : multi-turn Exp 1 recreation (same as 2b V2)

Output:
    data/intervention_results/concept_{probe|mean}_{v1|v2}/
        intervention_responses.csv
        intervention_results.json      (V1: includes GPT judge results)
        human_ai_causal_examples.txt   (V1: readable examples)
        per_subject/s001.csv ...       (V2)

Env: llama2_env

Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from contextlib import nullcontext

import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai

# --- Local imports ---
from src.dataset import llama_v2_prompt
from src.probes import LinearProbeClassification


# ========================== INLINE TRACEDICT ========================== #
# Avoids baukit dependency (matches Exp 2b approach)

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


# ========================== CLI ========================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp 3: Concept injection intervention."
    )
    parser.add_argument(
        "--mode", required=True, choices=["v1", "v2"],
        help="V1 = single-turn test questions + GPT judge; V2 = multi-turn Exp 1 recreation.",
    )
    parser.add_argument(
        "--vector_source", default="mean", choices=["probe", "mean"],
        help="Use concept probe weights or mean-difference vectors (default: mean).",
    )
    parser.add_argument(
        "--subject_idx", type=int, default=None,
        help="Subject index for V2 mode (0-49). If not set, uses SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--intervention_strength", type=float, default=None,
        help="Override default intervention strength N.",
    )
    return parser.parse_args()


# ========================== CONFIG ========================== #

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

# Concept vectors
CONCEPT_PROBE_DIR = "data/concept_probes"
CONCEPT_VEC_PATH = "data/concept_activations/concept_vector_per_layer.npz"

# V1 test questions (same as 2b)
CAUSAL_QUESTION_PATH = "data/causality_test_questions/human_ai.txt"

# V2 Experiment 1 references
EXP1_ROOT = Path("/jukebox/graziano/rachel/ai_mind_rep/exp_1/code/data_gen")
PROMPTS_DIR = EXP1_ROOT / "utils" / "prompts"
CONFIG_DIR = EXP1_ROOT / "utils" / "config"

INPUT_DIM = 5120  # LLaMA-2-13B hidden size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = DEVICE

# Intervention hyperparameters — matched to Exp 2b
FROM_LAYER = 27         # inclusive (model.layers.27 -> probe index 28)
TO_LAYER = 39           # exclusive (model.layers.38 -> probe index 39)
INTERVENTION_STRENGTH = 16  # matched to Exp 2b (13B-chat RLHF needs strong steering)
USE_RESIDUAL_STREAM = True

# Generation hyperparameters — matched to Exp 2b V1
MAX_NEW_TOKENS = 768
GEN_TEMPERATURE = 0.7   # sampling lets subtle probability shifts manifest
GEN_TOP_P = 0.9
GEN_DO_SAMPLE = True

# V2 generation (match Experiment 1)
V2_MAX_TOKENS = 500
V2_TEMPERATURE = 0.8
V2_TOP_P = 1.0
PAIRS_TOTAL = 5
HISTORY_PAIRS = 5
CONDITIONS = ["baseline", "human", "ai"]

SYSTEM_PROMPT = "You are engaging in a real-time spoken conversation."

# GPT judge model
JUDGE_MODEL = "gpt-4o-mini"


# ========================== VECTOR LOADING ========================== #

def load_concept_vectors_from_probes(probe_dir, from_layer, to_layer):
    """Load concept probe weights as steering vectors."""
    vectors = {}
    for layer_idx in range(from_layer + 1, to_layer + 1):
        probe_path = os.path.join(probe_dir, f"concept_probe_layer_{layer_idx}.pth")
        if not os.path.exists(probe_path):
            print(f"  [WARN] No concept probe for layer {layer_idx}")
            continue
        probe = LinearProbeClassification(
            device="cpu", probe_class=1, input_dim=INPUT_DIM, logistic=True,
        )
        state = torch.load(probe_path, map_location="cpu")
        probe.load_state_dict(state)
        probe.eval()
        vectors[layer_idx] = probe
    print(f"Loaded {len(vectors)} concept probe vectors (layers {from_layer+1}–{to_layer})")
    print(f"  Layer indices: {sorted(vectors.keys())}")
    return vectors


def load_concept_vectors_from_mean(vec_path, from_layer, to_layer):
    """
    Load mean-difference concept vectors and wrap them in probe objects
    so the intervention code works identically.
    """
    data = np.load(vec_path)
    concept_direction = torch.from_numpy(data["concept_direction"]).float()

    vectors = {}
    for layer_idx in range(from_layer + 1, to_layer + 1):
        if layer_idx >= concept_direction.shape[0]:
            continue
        probe = LinearProbeClassification(
            device="cpu", probe_class=1, input_dim=INPUT_DIM, logistic=True,
        )
        with torch.no_grad():
            direction = concept_direction[layer_idx].unsqueeze(0)
            direction = direction / (direction.norm() + 1e-8)
            probe.proj[0].weight.copy_(direction)
        probe.eval()
        vectors[layer_idx] = probe
    print(f"Loaded {len(vectors)} mean-difference concept vectors (layers {from_layer+1}–{to_layer})")
    print(f"  Layer indices: {sorted(vectors.keys())}")
    return vectors


# ========================== INTERVENTION UTILS ========================== #
# Matched to Exp 2b methodology (Viegas/TalkTuner)

def optimize_one_inter_rep(inter_rep, cf_target, probe, N):
    """
    One-step activation steering: h' = h + N * (target @ w)
    Follows TalkTuner (Chen et al., 2024) methodology.
    """
    tensor = inter_rep.to(TORCH_DEVICE).to(torch.float32)
    w = probe.proj[0].weight.to(TORCH_DEVICE).to(torch.float32)
    target = cf_target.to(TORCH_DEVICE).view(1, -1).to(torch.float32)
    direction = target @ w
    return tensor + N * direction


def make_edit_fn(probes_by_layer, cf_target, N):
    """
    Return an edit_output function for TraceDict that shifts the last-token
    residual along the probe/concept direction at each intervened layer.
    """
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
    """Collect residual-stream layer names for intervention."""
    layers = []
    for name, _ in model.named_modules():
        if name.startswith("model.layers.") and name.split(".")[-1].isdigit():
            n = int(name.split("model.layers.")[-1])
            if from_idx <= n < to_idx:
                layers.append(name)
    return layers


# ========================== BATCHED GENERATION ========================== #

def collect_responses_batched(
    model, tokenizer, prompts, modified_layer_names, edit_function,
    batch_size=5, max_new_tokens=MAX_NEW_TOKENS,
    temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, do_sample=GEN_DO_SAMPLE,
):
    """Generate responses with optional interventions, batched for efficiency."""
    print(f"  Intervening on layers: {modified_layer_names}")
    responses = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        message_lists = [[{"role": "user", "content": p}] for p in batch_prompts]
        formatted = [llama_v2_prompt(msgs) for msgs in message_lists]

        use_trace = len(modified_layer_names) > 0
        ctx = TraceDict(model, modified_layer_names, edit_output=edit_function) if use_trace else nullcontext()

        with ctx:
            with torch.no_grad():
                inputs = tokenizer(
                    formatted, return_tensors="pt",
                    padding=True, truncation=True, max_length=2048,
                ).to(TORCH_DEVICE)

                tokens = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                )

        for seq in tokens:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            if "[/INST]" in text:
                text = text.split("[/INST]", 1)[1]
            responses.append(text.strip())

    return responses


# ========================== GPT JUDGE ========================== #

def judge_pairwise(questions, responses_human, responses_ai):
    """
    GPT-based causal evaluation: for each question, show human-steered vs
    AI-steered responses and ask which better matches a specified partner type.
    Randomizes target type and response order to control for bias.
    Matched to Exp 2b judge_pairwise.

    Returns: (success_rate, judge_details_list)
    """
    assert len(questions) == len(responses_human) == len(responses_ai)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    template = (
        "Please answer which one of the following responses is more consistent with "
        "a conversation where the *user/partner* is a {partner_type}. "
        "Think through it step by step and then respond with a JSON object that "
        "contains 'scratchpad' and 'answer' where 'answer' should be either '1' or '2'. "
        "Note that the answer field _must_ be '1' or '2'. You must give an answer. "
        "Do not output anything else.\n\n"
    )

    def formatter(prompt, r1, r2):
        return (
            f"Prompt: {prompt}\n\n"
            f"Response 1: {r1}\n\n"
            f"Response 2: {r2}\n\n"
            "Answer:"
        )

    where_correct = []
    judge_details = []

    for idx, (q, r_h, r_a) in enumerate(tqdm(
        list(zip(questions, responses_human, responses_ai)), desc="GPT judging"
    )):
        d = "human" if np.random.randint(2) == 0 else "ai"
        instruction = template.format(partner_type=d)

        human_first = np.random.randint(2) == 0
        if human_first:
            query = formatter(q, r_h, r_a)
            correct_answer = 1 if d == "human" else 2
        else:
            query = formatter(q, r_a, r_h)
            correct_answer = 2 if d == "human" else 1
        where_correct.append(correct_answer)

        try:
            response = openai.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": instruction + query},
                ],
                temperature=0.0,
                top_p=0.0,
            )
            msg = response.choices[0].message
            content = msg.content if isinstance(msg.content, str) else "".join(
                part["text"] for part in msg.content if isinstance(part, dict) and "text" in part
            )
        except Exception as e:
            print(f"  [WARN] GPT judge failed on question {idx}: {e}")
            content = ""

        content = content.strip()

        try:
            cleaned = content.removeprefix("```json").removesuffix("```").strip()
            obj = json.loads(cleaned)
            judge_answer = int(obj.get("answer", 0))
        except (json.JSONDecodeError, ValueError):
            judge_answer = 0
            for ch in reversed(content):
                if ch in ("1", "2"):
                    judge_answer = int(ch)
                    break

        is_correct = judge_answer == correct_answer
        judge_details.append({
            "question_idx": idx,
            "question": q,
            "target_type": d,
            "response_order": "human_first" if human_first else "ai_first",
            "correct_answer": correct_answer,
            "judge_answer": judge_answer,
            "is_correct": is_correct,
            "judge_raw": content,
        })

    where_correct = np.array(where_correct)
    processed = np.array([d["judge_answer"] for d in judge_details])
    success = float((processed == where_correct).mean())
    print(f"GPT judge success rate: {success:.3f}")
    return success, judge_details


# ========================== V1 MODE ========================== #

def run_v1(model, tokenizer, vectors, result_dir, N):
    """Single-turn test questions with concept steering + GPT judge."""
    os.makedirs(result_dir, exist_ok=True)

    with open(CAUSAL_QUESTION_PATH, "r") as f:
        questions = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(questions)} test questions.")

    modified_layer_names = get_modified_layer_names(model, FROM_LAYER, TO_LAYER)

    def null_edit(output, layer_name):
        return output

    # --- Baseline ---
    print("\n=== Generating baseline (no intervention) responses ===")
    baseline_responses = collect_responses_batched(
        model, tokenizer, questions,
        modified_layer_names=[],
        edit_function=null_edit,
        batch_size=5,
    )

    # --- Human-steered ---
    print("\n=== Generating human-concept-steered responses ===")
    human_edit_fn = make_edit_fn(vectors, torch.tensor([1.0]), N)
    human_responses = collect_responses_batched(
        model, tokenizer, questions,
        modified_layer_names=modified_layer_names,
        edit_function=human_edit_fn,
        batch_size=5,
    )

    # --- AI-steered ---
    print("\n=== Generating AI-concept-steered responses ===")
    ai_edit_fn = make_edit_fn(vectors, torch.tensor([-1.0]), N)
    ai_responses = collect_responses_batched(
        model, tokenizer, questions,
        modified_layer_names=modified_layer_names,
        edit_function=ai_edit_fn,
        batch_size=5,
    )

    # === Save structured CSV ===
    csv_path = os.path.join(result_dir, "intervention_responses.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_idx", "question", "condition", "response",
        ])
        writer.writeheader()
        for idx, q in enumerate(questions):
            for cond, resp_list in [
                ("baseline", baseline_responses),
                ("human", human_responses),
                ("ai", ai_responses),
            ]:
                writer.writerow({
                    "question_idx": idx,
                    "question": q,
                    "condition": cond,
                    "response": resp_list[idx],
                })
    print(f"\nSaved responses to {csv_path}")

    # === Save readable examples ===
    txt_path = os.path.join(result_dir, "human_ai_causal_examples.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i in range(len(questions)):
            f.write(f"USER: {questions[i]}\n\n")
            f.write("-" * 50 + "\n")
            f.write("Baseline (no intervention):\n")
            f.write(f"ASSISTANT: {baseline_responses[i]}\n\n")
            f.write("-" * 50 + "\n")
            f.write("Intervened: steer toward HUMAN concept\n")
            f.write(f"ASSISTANT: {human_responses[i]}\n\n")
            f.write("-" * 50 + "\n")
            f.write("Intervened: steer toward AI concept\n")
            f.write(f"ASSISTANT: {ai_responses[i]}\n\n")
            f.write("=" * 80 + "\n\n")
    print(f"Saved examples to {txt_path}")

    # === GPT judge evaluation ===
    print("\n=== GPT-based causal evaluation ===")
    success_rate, judge_details = judge_pairwise(
        questions, human_responses, ai_responses,
    )

    # === Save full results JSON ===
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp3_concept_injection",
        "config": {
            "model": "LLaMA-2-Chat-13B",
            "model_path": MODEL_NAME,
            "vector_source": "concept",
            "from_layer": FROM_LAYER,
            "to_layer": TO_LAYER,
            "intervention_strength": N,
            "probe_layers_used": sorted(vectors.keys()),
            "max_new_tokens": MAX_NEW_TOKENS,
            "gen_temperature": GEN_TEMPERATURE,
            "gen_top_p": GEN_TOP_P,
            "gen_do_sample": GEN_DO_SAMPLE,
            "n_questions": len(questions),
            "judge_model": JUDGE_MODEL,
        },
        "judge_success_rate": success_rate,
        "judge_details": judge_details,
    }
    json_path = os.path.join(result_dir, "intervention_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {json_path}")
    print(f"\nFinal GPT-judge success rate: {success_rate:.3f}")


# ========================== V2 MODE ========================== #

def generate_single_turn_v2(model, tokenizer, messages, layer_names, edit_fn):
    """Generate a single response for V2 multi-turn mode."""
    formatted = llama_v2_prompt(messages)

    use_trace = len(layer_names) > 0
    ctx = TraceDict(model, layer_names, edit_output=edit_fn) if use_trace else nullcontext()

    with ctx:
        with torch.no_grad():
            inputs = tokenizer(
                formatted, return_tensors="pt",
                truncation=True, max_length=2048,
            ).to(TORCH_DEVICE)
            tokens = model.generate(
                **inputs, max_new_tokens=V2_MAX_TOKENS,
                do_sample=(V2_TEMPERATURE > 0),
                temperature=V2_TEMPERATURE if V2_TEMPERATURE > 0 else None,
                top_p=V2_TOP_P,
            )
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    if "[/INST]" in text:
        text = text.rsplit("[/INST]", 1)[-1]
    return text.strip()


def truncate_history(history, keep_pairs):
    systems = [m for m in history if m["role"] == "system"]
    others = [m for m in history if m["role"] != "system"]
    return systems + (others[-2*keep_pairs:] if keep_pairs > 0 else others[-2:])


def load_prompt_text(prompts_dir, topic):
    p = prompts_dir / f"{topic}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Prompt not found: {p}")
    return p.read_text(encoding="utf-8").strip(), p.name


def run_v2_subject(model, tokenizer, vectors, result_dir, subject_idx, N):
    """Multi-turn Exp 1 recreation for one subject with concept steering."""
    import pandas as pd

    subject_id = f"s{subject_idx + 1:03d}"
    config_path = CONFIG_DIR / f"conds_{subject_id}.csv"
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return

    df_config = pd.read_csv(config_path).reset_index(drop=True)
    df_config["trial"] = df_config.index + 1

    layer_names = get_modified_layer_names(model, FROM_LAYER, TO_LAYER)

    def null_edit(output, layer_name):
        return output

    edit_fns = {
        "baseline": null_edit,
        "human": make_edit_fn(vectors, torch.tensor([1.0]), N),
        "ai": make_edit_fn(vectors, torch.tensor([-1.0]), N),
    }
    layer_per_cond = {"baseline": [], "human": layer_names, "ai": layer_names}

    out_dir = Path(result_dir) / "per_subject"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{subject_id}.csv"

    fieldnames = [
        "subject", "run", "order", "trial", "condition",
        "topic", "topic_file", "pair_index",
        "transcript_sub", "transcript_llm",
    ]

    total = len(df_config) * len(CONDITIONS)
    count = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for _, row in df_config.iterrows():
            topic = str(row["topic"]).strip()
            topic_text, topic_file = load_prompt_text(PROMPTS_DIR, topic)

            topic_intro = (
                f"The conversation topic is: '{topic_text}'.\n\n"
                f"Please begin by producing only your first message to start the conversation.\n"
                f"Do not simulate both sides of the dialogue."
            )

            for condition in CONDITIONS:
                count += 1
                if count % 20 == 0:
                    print(f"  [{subject_id}] {count}/{total}")

                try:
                    sub_hist = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": topic_intro},
                    ]
                    llm_hist = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": topic_intro},
                    ]

                    sub_input = truncate_history(sub_hist, HISTORY_PAIRS)
                    sub_msg = generate_single_turn_v2(
                        model, tokenizer, sub_input,
                        layer_per_cond[condition], edit_fns[condition],
                    )
                    sub_hist.append({"role": "assistant", "content": sub_msg})
                    llm_hist.append({"role": "user", "content": sub_msg})

                    pair_index = 1
                    while pair_index <= PAIRS_TOTAL:
                        llm_input = truncate_history(llm_hist, HISTORY_PAIRS)
                        llm_msg = generate_single_turn_v2(
                            model, tokenizer, llm_input, [], null_edit,
                        )
                        llm_hist.append({"role": "assistant", "content": llm_msg})

                        writer.writerow({
                            "subject": subject_id, "run": int(row["run"]),
                            "order": int(row["order"]), "trial": int(row["trial"]),
                            "condition": condition, "topic": topic,
                            "topic_file": topic_file, "pair_index": pair_index,
                            "transcript_sub": sub_msg, "transcript_llm": llm_msg,
                        })

                        if pair_index == PAIRS_TOTAL:
                            break

                        sub_hist.append({"role": "user", "content": f"Partner: {llm_msg}"})
                        sub_input = truncate_history(sub_hist, HISTORY_PAIRS)
                        sub_msg = generate_single_turn_v2(
                            model, tokenizer, sub_input,
                            layer_per_cond[condition], edit_fns[condition],
                        )
                        sub_hist.append({"role": "assistant", "content": sub_msg})
                        llm_hist.append({"role": "user", "content": sub_msg})
                        pair_index += 1

                except Exception as e:
                    print(f"  [ERROR] {subject_id} | {topic} [{condition}]: {e}")

    print(f"Saved {subject_id} to {out_csv}")


# ========================== MAIN ========================== #

def main():
    args = parse_args()

    N = args.intervention_strength if args.intervention_strength else INTERVENTION_STRENGTH

    # Load model
    print("Loading LLaMA-2-Chat-13B...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=True, padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(TORCH_DEVICE).eval()

    if "<pad>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load concept vectors
    if args.vector_source == "probe":
        vectors = load_concept_vectors_from_probes(CONCEPT_PROBE_DIR, FROM_LAYER, TO_LAYER)
    else:
        vectors = load_concept_vectors_from_mean(CONCEPT_VEC_PATH, FROM_LAYER, TO_LAYER)

    result_dir = f"data/intervention_results/concept_{args.vector_source}_{args.mode}"

    # Log config
    print(f"\nIntervention config:")
    print(f"  Model: LLaMA-2-Chat-13B")
    print(f"  Vector source: {args.vector_source}")
    print(f"  Layer range: model.layers.{FROM_LAYER} to model.layers.{TO_LAYER-1}")
    print(f"  Probe indices used: {sorted(vectors.keys())}")
    print(f"  N = {N}")
    print(f"  Generation: temp={GEN_TEMPERATURE}, top_p={GEN_TOP_P}, sample={GEN_DO_SAMPLE}")

    if args.mode == "v1":
        run_v1(model, tokenizer, vectors, result_dir, N)
    else:
        idx = args.subject_idx
        if idx is None:
            idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        run_v2_subject(model, tokenizer, vectors, result_dir, idx, N)

    print(f"\n✅ Concept injection ({args.mode}, {args.vector_source}, N={N}) complete.")


if __name__ == "__main__":
    main()