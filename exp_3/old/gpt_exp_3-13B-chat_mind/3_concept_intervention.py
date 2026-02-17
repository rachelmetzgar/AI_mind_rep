#!/usr/bin/env python3
"""
Experiment 3, Phase 3: Concept Injection into Conversations

Injects the model's general "human" vs "AI" concept vector into conversational
generation. Sweeps over multiple intervention strengths (N) to produce a
dose-response curve for the concept steering effect.

All vectors (from both probe and mean-difference sources) are unit-normalized
before steering, so N directly controls step size in activation-space units.

Dose-response sweep:
    Default N values: [1, 2, 4, 8]
    Baseline generated once and shared across all N values.
    Each N gets its own output subfolder.
    Summary CSV comparing judge accuracy across N values.

Supports two steering vector sources:
    --vector_source probe   : concept probe weight vectors (normalized)
    --vector_source mean    : mean-difference concept vectors (normalized)

Supports two generation modes:
    --mode v1   : single-turn test questions + GPT judge (with N sweep)
    --mode v2   : multi-turn Exp 1 recreation (single N, use --n_values)

Output (V1):
    data/intervention_results/concept_{probe|mean}_v1/
        N1/   N2/   N4/   N8/      — per-N results
        dose_response_summary.csv   — judge accuracy vs N

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

class TraceDict:
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
        description="Exp 3: Concept injection with dose-response sweep."
    )
    parser.add_argument(
        "--mode", required=True, choices=["v1", "v2"],
        help="V1 = single-turn + GPT judge (sweeps N); V2 = multi-turn Exp 1 recreation.",
    )
    parser.add_argument(
        "--vector_source", default="mean", choices=["probe", "mean"],
        help="Concept probe weights or mean-difference vectors (default: mean).",
    )
    parser.add_argument(
        "--n_values", type=float, nargs="+", default=None,
        help="Override default N sweep values (e.g. --n_values 0.5 1 2 4 8).",
    )
    parser.add_argument(
        "--subject_idx", type=int, default=None,
        help="Subject index for V2 mode (0-49).",
    )
    parser.add_argument(
        "--judge", default="local", choices=["local", "gpt"],
        help="Judge model: 'local' reuses the loaded LLaMA (free), "
             "'gpt' uses GPT-4o-mini API (default: local).",
    )
    return parser.parse_args()


# ========================== CONFIG ========================== #

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

CONCEPT_PROBE_DIR = "data/concept_probes"
CONCEPT_VEC_PATH = "data/concept_activations/concept_vector_per_layer.npz"
CAUSAL_QUESTION_PATH = "data/causality_test_questions/human_ai.txt"

EXP1_ROOT = Path("/jukebox/graziano/rachel/ai_mind_rep/exp_1/code/data_gen")
PROMPTS_DIR = EXP1_ROOT / "utils" / "prompts"
CONFIG_DIR = EXP1_ROOT / "utils" / "config"

INPUT_DIM = 5120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = DEVICE

# Layer range for intervention
FROM_LAYER = 27
TO_LAYER = 39

# Default N sweep: all vectors are unit-normalized, so N = step size in
# activation-space units. Typical hidden state norm is ~50-200, so:
#   N=1 ≈ 0.5-2% nudge, N=2 ≈ 1-4%, N=4 ≈ 2-8%, N=8 ≈ 4-16%
DEFAULT_N_VALUES = [1, 2, 4, 8]

# V1 generation (matched to Exp 2b)
MAX_NEW_TOKENS = 768
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_DO_SAMPLE = True

# V2 generation (matched to Experiment 1)
V2_MAX_TOKENS = 500
V2_TEMPERATURE = 0.8
V2_TOP_P = 1.0
PAIRS_TOTAL = 5
HISTORY_PAIRS = 5
CONDITIONS = ["baseline", "human", "ai"]
SYSTEM_PROMPT = "You are engaging in a real-time spoken conversation."

JUDGE_MODEL = "gpt-4o-mini"


# ========================== VECTOR LOADING ========================== #

def load_concept_vectors_from_probes(probe_dir, from_layer, to_layer):
    """Load concept probe weights, normalized to unit length."""
    vectors = {}
    for layer_idx in range(from_layer + 1, to_layer + 1):
        probe_path = os.path.join(probe_dir, f"concept_probe_layer_{layer_idx}.pth")
        if not os.path.exists(probe_path):
            continue
        probe = LinearProbeClassification(
            device="cpu", probe_class=1, input_dim=INPUT_DIM, logistic=True,
        )
        state = torch.load(probe_path, map_location="cpu")
        probe.load_state_dict(state)
        with torch.no_grad():
            w = probe.proj[0].weight
            probe.proj[0].weight.copy_(w / (w.norm() + 1e-8))
        probe.eval()
        vectors[layer_idx] = probe
    print(f"Loaded {len(vectors)} concept probe vectors (unit-normalized)")
    print(f"  Layer indices: {sorted(vectors.keys())}")
    return vectors


def load_concept_vectors_from_mean(vec_path, from_layer, to_layer):
    """Load mean-difference concept vectors, normalized to unit length."""
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
    print(f"Loaded {len(vectors)} mean-difference concept vectors (unit-normalized)")
    print(f"  Layer indices: {sorted(vectors.keys())}")
    return vectors


# ========================== INTERVENTION UTILS ========================== #

def optimize_one_inter_rep(inter_rep, cf_target, probe, N):
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
    layers = []
    for name, _ in model.named_modules():
        if name.startswith("model.layers.") and name.split(".")[-1].isdigit():
            n = int(name.split("model.layers.")[-1])
            if from_idx <= n < to_idx:
                layers.append(name)
    return layers


# ========================== GENERATION ========================== #

def collect_responses_batched(
    model, tokenizer, prompts, modified_layer_names, edit_function,
    batch_size=5, max_new_tokens=MAX_NEW_TOKENS,
    temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P, do_sample=GEN_DO_SAMPLE,
):
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i+batch_size]
        msg_lists = [[{"role": "user", "content": p}] for p in batch]
        formatted = [llama_v2_prompt(msgs) for msgs in msg_lists]

        use_trace = len(modified_layer_names) > 0
        ctx = TraceDict(model, modified_layer_names, edit_output=edit_function) if use_trace else nullcontext()

        with ctx:
            with torch.no_grad():
                inputs = tokenizer(
                    formatted, return_tensors="pt",
                    padding=True, truncation=True, max_length=2048,
                ).to(TORCH_DEVICE)
                tokens = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
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
        return f"Prompt: {prompt}\n\nResponse 1: {r1}\n\nResponse 2: {r2}\n\nAnswer:"

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
                temperature=0.0, top_p=0.0,
            )
            content = response.choices[0].message.content or ""
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

        judge_details.append({
            "question_idx": idx, "question": q, "target_type": d,
            "response_order": "human_first" if human_first else "ai_first",
            "correct_answer": correct_answer, "judge_answer": judge_answer,
            "is_correct": judge_answer == correct_answer, "judge_raw": content,
        })

    where_correct = np.array(where_correct)
    processed = np.array([d["judge_answer"] for d in judge_details])
    success = float((processed == where_correct).mean())
    print(f"GPT judge success rate: {success:.3f}")
    return success, judge_details


# ========================== LOCAL LLAMA JUDGE ========================== #

def judge_pairwise_local(questions, responses_human, responses_ai, model, tokenizer):
    """
    Use the already-loaded LLaMA-2-Chat-13B as judge. Zero API cost.
    Same protocol as GPT judge: randomize target type and response order,
    ask the model to pick which response matches the target.

    Uses a simpler prompt (no JSON) since LLaMA-2-13B is less reliable
    at structured output. Just asks for "1" or "2".
    """
    assert len(questions) == len(responses_human) == len(responses_ai)

    template = (
        "You are evaluating two AI assistant responses. "
        "Which response is more consistent with a conversation where the "
        "user/partner is a {partner_type}? "
        "You MUST answer with just the number 1 or 2. "
        "Do not explain your reasoning.\n\n"
        "Prompt: {prompt}\n\n"
        "Response 1: {r1}\n\n"
        "Response 2: {r2}\n\n"
        "Answer (1 or 2):"
    )

    where_correct = []
    judge_details = []

    for idx, (q, r_h, r_a) in enumerate(tqdm(
        list(zip(questions, responses_human, responses_ai)), desc="Local LLaMA judging"
    )):
        d = "human" if np.random.randint(2) == 0 else "ai"
        human_first = np.random.randint(2) == 0

        if human_first:
            r1, r2 = r_h, r_a
            correct_answer = 1 if d == "human" else 2
        else:
            r1, r2 = r_a, r_h
            correct_answer = 2 if d == "human" else 1
        where_correct.append(correct_answer)

        # Truncate responses to avoid context length issues
        r1_trunc = r1[:1500] if len(r1) > 1500 else r1
        r2_trunc = r2[:1500] if len(r2) > 1500 else r2

        judge_prompt = template.format(
            partner_type=d, prompt=q, r1=r1_trunc, r2=r2_trunc,
        )

        messages = [{"role": "user", "content": judge_prompt}]
        formatted = llama_v2_prompt(messages)

        with torch.no_grad():
            inputs = tokenizer(
                formatted, return_tensors="pt",
                truncation=True, max_length=3500,
            ).to(TORCH_DEVICE)
            tokens = model.generate(
                **inputs, max_new_tokens=10,
                do_sample=False,  # greedy for consistency
            )

        output_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        if "[/INST]" in output_text:
            output_text = output_text.rsplit("[/INST]", 1)[-1]
        output_text = output_text.strip()

        # Extract answer: look for first "1" or "2"
        judge_answer = 0
        for ch in output_text:
            if ch in ("1", "2"):
                judge_answer = int(ch)
                break

        is_correct = judge_answer == correct_answer
        judge_details.append({
            "question_idx": idx, "question": q, "target_type": d,
            "response_order": "human_first" if human_first else "ai_first",
            "correct_answer": correct_answer, "judge_answer": judge_answer,
            "is_correct": is_correct, "judge_raw": output_text,
        })

    where_correct = np.array(where_correct)
    processed = np.array([d["judge_answer"] for d in judge_details])
    success = float((processed == where_correct).mean())
    print(f"Local LLaMA judge success rate: {success:.3f}")
    return success, judge_details

def run_v1_sweep(model, tokenizer, vectors, result_dir_base, n_values, judge_mode="local"):
    """
    Run single-turn steering at multiple N values.
    Baseline generated once; human/AI steered per N; GPT judge per N.
    Saves per-N results + summary CSV.
    """
    os.makedirs(result_dir_base, exist_ok=True)

    with open(CAUSAL_QUESTION_PATH, "r") as f:
        questions = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(questions)} test questions.")

    modified_layer_names = get_modified_layer_names(model, FROM_LAYER, TO_LAYER)

    def null_edit(output, layer_name):
        return output

    # --- Baseline (once) ---
    print("\n=== Generating baseline (no intervention) ===")
    baseline_responses = collect_responses_batched(
        model, tokenizer, questions,
        modified_layer_names=[], edit_function=null_edit, batch_size=5,
    )

    # --- Sweep over N values ---
    dose_response = []

    for N in n_values:
        n_label = f"N{N}" if N == int(N) else f"N{N:.1f}"
        n_dir = os.path.join(result_dir_base, n_label)
        os.makedirs(n_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  N = {N}")
        print(f"{'='*60}")

        # Human-steered
        print(f"  Generating human-steered (N={N})...")
        human_edit_fn = make_edit_fn(vectors, torch.tensor([1.0]), N)
        human_responses = collect_responses_batched(
            model, tokenizer, questions,
            modified_layer_names=modified_layer_names,
            edit_function=human_edit_fn, batch_size=5,
        )

        # AI-steered
        print(f"  Generating AI-steered (N={N})...")
        ai_edit_fn = make_edit_fn(vectors, torch.tensor([-1.0]), N)
        ai_responses = collect_responses_batched(
            model, tokenizer, questions,
            modified_layer_names=modified_layer_names,
            edit_function=ai_edit_fn, batch_size=5,
        )

        # Save CSV
        csv_path = os.path.join(n_dir, "intervention_responses.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question_idx", "question", "condition", "response",
            ])
            writer.writeheader()
            for idx, q in enumerate(questions):
                for cond, resps in [
                    ("baseline", baseline_responses),
                    ("human", human_responses),
                    ("ai", ai_responses),
                ]:
                    writer.writerow({
                        "question_idx": idx, "question": q,
                        "condition": cond, "response": resps[idx],
                    })

        # Save examples
        txt_path = os.path.join(n_dir, "human_ai_causal_examples.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Concept injection (N={N}, unit-normalized vectors)\n")
            f.write("=" * 80 + "\n\n")
            for i in range(len(questions)):
                f.write(f"USER: {questions[i]}\n\n")
                f.write("-" * 50 + "\nBaseline:\n")
                f.write(f"ASSISTANT: {baseline_responses[i]}\n\n")
                f.write("-" * 50 + f"\nHuman-steered (N={N}):\n")
                f.write(f"ASSISTANT: {human_responses[i]}\n\n")
                f.write("-" * 50 + f"\nAI-steered (N={N}):\n")
                f.write(f"ASSISTANT: {ai_responses[i]}\n\n")
                f.write("=" * 80 + "\n\n")

        # Judge
        print(f"  Judging (N={N}, mode={judge_mode})...")
        if judge_mode == "local":
            success_rate, judge_details = judge_pairwise_local(
                questions, human_responses, ai_responses, model, tokenizer,
            )
            judge_model_name = "LLaMA-2-Chat-13B (local)"
        else:
            success_rate, judge_details = judge_pairwise(
                questions, human_responses, ai_responses,
            )
            judge_model_name = JUDGE_MODEL

        # Save JSON
        results = {
            "timestamp": datetime.now().isoformat(),
            "experiment": "exp3_concept_injection",
            "config": {
                "model": "LLaMA-2-Chat-13B",
                "vectors_unit_normalized": True,
                "from_layer": FROM_LAYER, "to_layer": TO_LAYER,
                "intervention_strength": N,
                "probe_layers_used": sorted(vectors.keys()),
                "max_new_tokens": MAX_NEW_TOKENS,
                "gen_temperature": GEN_TEMPERATURE,
                "gen_do_sample": GEN_DO_SAMPLE,
                "n_questions": len(questions),
                "judge_model": judge_model_name,
            },
            "judge_success_rate": success_rate,
            "judge_details": judge_details,
        }
        with open(os.path.join(n_dir, "intervention_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        dose_response.append({
            "N": N,
            "judge_success_rate": success_rate,
            "n_questions": len(questions),
        })

        print(f"  N={N}: judge accuracy = {success_rate:.3f}")

    # === Save dose-response summary ===
    summary_path = os.path.join(result_dir_base, "dose_response_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["N", "judge_success_rate", "n_questions"])
        writer.writeheader()
        for row in dose_response:
            writer.writerow(row)

    # Print summary
    print(f"\n{'='*60}")
    print("  DOSE-RESPONSE SUMMARY")
    print(f"{'='*60}")
    for row in dose_response:
        print(f"  N={row['N']:5.1f}  →  judge accuracy = {row['judge_success_rate']:.3f}")
    print(f"\nSaved to {summary_path}")


# ========================== V2 MODE ========================== #

def generate_single_turn_v2(model, tokenizer, messages, layer_names, edit_fn):
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
    """V2 multi-turn for one subject at a single N value."""
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
                f"Please begin by producing only your first message.\n"
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

    n_values = args.n_values if args.n_values else DEFAULT_N_VALUES

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

    # Load concept vectors (both sources unit-normalized)
    if args.vector_source == "probe":
        vectors = load_concept_vectors_from_probes(CONCEPT_PROBE_DIR, FROM_LAYER, TO_LAYER)
    else:
        vectors = load_concept_vectors_from_mean(CONCEPT_VEC_PATH, FROM_LAYER, TO_LAYER)

    result_dir_base = f"data/intervention_results/concept_{args.vector_source}_{args.mode}"

    # Log config
    print(f"\nIntervention config:")
    print(f"  Vector source: {args.vector_source} (unit-normalized)")
    print(f"  Layer range: model.layers.{FROM_LAYER}–{TO_LAYER-1}")
    print(f"  Probe layers: {sorted(vectors.keys())}")
    print(f"  N values: {n_values}")
    print(f"  Generation: temp={GEN_TEMPERATURE}, top_p={GEN_TOP_P}, sample={GEN_DO_SAMPLE}")
    print(f"  Judge: {args.judge}")

    if args.mode == "v1":
        run_v1_sweep(model, tokenizer, vectors, result_dir_base, n_values, judge_mode=args.judge)
    else:
        # V2: use first N value (or specify single via --n_values 2)
        N = n_values[0]
        idx = args.subject_idx
        if idx is None:
            idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        n_label = f"N{N}" if N == int(N) else f"N{N:.1f}"
        v2_result_dir = os.path.join(result_dir_base, n_label)
        run_v2_subject(model, tokenizer, vectors, v2_result_dir, idx, N)

    print(f"\n✅ Complete.")


if __name__ == "__main__":
    main()