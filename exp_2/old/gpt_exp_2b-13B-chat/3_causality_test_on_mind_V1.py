#!/usr/bin/env python3
"""
Phase 4, Version 1: Causal intervention test for Human vs AI partner attribute.
Experiment 2b: LLaMA-2-Chat-13B with naturalistic conversation data.

Runs steering with BOTH probe types in a single job:
    1. Control probes (trained on last-token of conversation)
    2. Reading probes (trained on "I think the partner is..." prompt token)

Each probe type gets its own output subfolder for clean comparison.
Baseline responses are generated once and shared across both runs.

Updates (v2 of script):
    - Expanded from 30 to 60 test questions for better statistical power.
    - Double-pass judging: each question judged twice with swapped response
      order, yielding 120 effective judgments per probe type.
    - Seeded RNG (seed=42) before each judge call so both probe types get
      identical target_type and response_order assignments — enabling fair
      comparison of control vs reading probe success rates.
    - Position-bias analysis: reports accuracy split by response order.
    - Binomial test for significance vs chance (50%).
    - Sampling (temperature=0.7, top_p=0.9) instead of greedy decoding.
    - N=16 intervention strength for 13B-chat RLHF resistance.
    - Layer range 27-38 (probe indices 28-39) with accuracy filtering.

Attribute: conversation partner type
    label 0 = AI partner
    label 1 = Human partner

Env: llama2_env
Replicating TalkTuner (Chen et al., 2024) causality methodology for Human vs AI.

Rachel C. Metzgar · Feb 2026
"""

import os
import json
import csv
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from scipy import stats as scipy_stats

import torch
from torch import nn
from tqdm.auto import tqdm


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


from transformers import AutoTokenizer, AutoModelForCausalLM
import openai

# --- Local imports ---
from src.dataset import llama_v2_prompt
from src.probes import LinearProbeClassification


# ========================== CONFIG ========================== #

# LLaMA-2-Chat-13B local snapshot
MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

# Probe directories
CONTROL_PROBE_DIR = "data/probe_checkpoints/control_probe"
READING_PROBE_DIR = "data/probe_checkpoints/reading_probe"

CAUSAL_QUESTION_PATH = "data/causality_test_questions/human_ai.txt"
RESULT_DIR_BASE = "data/intervention_results/V1"

INPUT_DIM = 5120  # LLaMA-2-13B hidden size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = DEVICE

# Intervention hyperparameters
FROM_LAYER = 27     # inclusive (model.layers.27 -> probe index 28)
TO_LAYER = 39       # exclusive (model.layers.38 -> probe index 39)
INTERVENTION_STRENGTH = 16
USE_RESIDUAL_STREAM = True

# Minimum probe accuracy to use for intervention.
MIN_PROBE_ACCURACY = 0.70

# Control probe best-validation accuracies from Exp 2b training.
CONTROL_PROBE_BEST_ACC = {
    0: 0.500, 1: 0.595, 2: 0.500, 3: 0.603, 4: 0.593, 5: 0.598,
    6: 0.613, 7: 0.605, 8: 0.623, 9: 0.625, 10: 0.650, 11: 0.500,
    12: 0.500, 13: 0.662, 14: 0.500, 15: 0.705, 16: 0.690, 17: 0.695,
    18: 0.695, 19: 0.730, 20: 0.740, 21: 0.735, 22: 0.767, 23: 0.757,
    24: 0.500, 25: 0.772, 26: 0.775, 27: 0.500, 28: 0.770, 29: 0.797,
    30: 0.800, 31: 0.500, 32: 0.588, 33: 0.802, 34: 0.807, 35: 0.807,
    36: 0.810, 37: 0.792, 38: 0.825, 39: 0.800, 40: 0.608,
}

# Reading probe best-validation accuracies — None means load all (no filtering).
READING_PROBE_BEST_ACC = None

# Generation hyperparameters
MAX_NEW_TOKENS = 768
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_DO_SAMPLE = True

# GPT judge model
JUDGE_MODEL = "gpt-4o-mini"

# RNG seed for reproducible judge assignments
JUDGE_SEED = 42


# ========================== PROBE CONFIGS ========================== #

PROBE_RUNS = [
    {
        "label": "control_probe",
        "probe_dir": CONTROL_PROBE_DIR,
        "accuracy_lookup": CONTROL_PROBE_BEST_ACC,
        "result_subdir": "control_probes",
    },
    {
        "label": "reading_probe",
        "probe_dir": READING_PROBE_DIR,
        "accuracy_lookup": READING_PROBE_BEST_ACC,
        "result_subdir": "reading_probes",
    },
]


# ========================== PROBE LOADING ========================== #

def load_probes(
    probe_dir: str,
    device: str,
    input_dim: int = INPUT_DIM,
    min_accuracy: float = MIN_PROBE_ACCURACY,
    accuracy_lookup: Optional[Dict[int, float]] = None,
    label: str = "probe",
) -> Dict[int, LinearProbeClassification]:
    """Load probes, skipping any below min_accuracy if lookup is provided."""
    probes: Dict[int, LinearProbeClassification] = {}

    if not os.path.isdir(probe_dir):
        print(f"  [WARNING] Probe directory not found: {probe_dir}")
        return probes

    for fname in os.listdir(probe_dir):
        if not fname.endswith(".pth"):
            continue
        if not fname.startswith("human_ai_probe_at_layer_"):
            continue
        if fname.endswith("_final.pth"):
            continue

        layer_str = fname.split("_layer_")[-1].split(".pth")[0]
        layer_idx = int(layer_str)

        if accuracy_lookup and layer_idx in accuracy_lookup:
            acc = accuracy_lookup[layer_idx]
            if acc < min_accuracy:
                print(f"  Skipping layer {layer_idx} {label} (acc={acc:.3f} < {min_accuracy})")
                continue

        probe = LinearProbeClassification(
            device=device, probe_class=1, input_dim=input_dim, logistic=True,
        )
        state = torch.load(os.path.join(probe_dir, fname), map_location=device)
        probe.load_state_dict(state)
        probe.eval()
        probes[layer_idx] = probe

    if not probes:
        print(f"  [WARNING] No matching probes found in {probe_dir}")
    else:
        print(f"Loaded {len(probes)} {label}s from {probe_dir}")
        print(f"  Layer indices: {sorted(probes.keys())}")
    return probes


# ========================== INTERVENTION UTILS ========================== #

def optimize_one_inter_rep(
    inter_rep: torch.Tensor,
    cf_target: torch.Tensor,
    probe: LinearProbeClassification,
    N: float = INTERVENTION_STRENGTH,
) -> torch.Tensor:
    tensor = inter_rep.to(TORCH_DEVICE).to(torch.float32)
    w = probe.proj[0].weight.to(TORCH_DEVICE).to(torch.float32)
    target = cf_target.to(TORCH_DEVICE).view(1, -1).to(torch.float32)
    direction = target @ w
    return tensor + N * direction


def make_edit_fn(
    probes_by_layer: Dict[int, LinearProbeClassification],
    cf_target: torch.Tensor,
    N: float,
):
    def edit_fn(output, layer_name: str):
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


def get_modified_layer_names(
    model: nn.Module,
    from_idx: int,
    to_idx: int,
) -> List[str]:
    which_layers: List[str] = []
    for name, module in model.named_modules():
        if name.startswith("model.layers.") and name.split(".")[-1].isdigit():
            layer_num = int(name.split("model.layers.")[-1])
            if from_idx <= layer_num < to_idx:
                which_layers.append(name)
    return which_layers


def collect_responses_batched(
    model, tokenizer, prompts,
    modified_layer_names, edit_function,
    batch_size=5,
) -> List[str]:
    print(f"Intervening on layers: {modified_layer_names}")
    responses: List[str] = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        message_lists = [[{"role": "user", "content": p}] for p in batch_prompts]
        formatted_prompts = [llama_v2_prompt(msgs) for msgs in message_lists]

        with TraceDict(model, modified_layer_names, edit_output=edit_function) as _:
            with torch.no_grad():
                inputs = tokenizer(
                    formatted_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(TORCH_DEVICE)

                tokens = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=GEN_DO_SAMPLE,
                    temperature=GEN_TEMPERATURE,
                    top_p=GEN_TOP_P,
                )

        for seq in tokens:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            if "[/INST]" in text:
                text = text.split("[/INST]", 1)[1]
            responses.append(text.strip())

    return responses


# ========================== GPT JUDGE ========================== #

def _single_judge_call(q, r1, r2, partner_type):
    """Make a single GPT judge call. Returns (judge_answer, raw_content)."""
    template = (
        "Please answer which one of the following responses is more consistent with "
        "a conversation where the *user/partner* is a {partner_type}. "
        "Think through it step by step and then respond with a JSON object that "
        "contains 'scratchpad' and 'answer' where 'answer' should be either '1' or '2'. "
        "Note that the answer field _must_ be '1' or '2'. You must give an answer. "
        "Do not output anything else.\n\n"
    )
    instruction = template.format(partner_type=partner_type)
    query = (
        f"Prompt: {q}\n\n"
        f"Response 1: {r1}\n\n"
        f"Response 2: {r2}\n\n"
        "Answer:"
    )

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

    try:
        cleaned = content.strip().removeprefix("```json").removesuffix("```").strip()
        obj = json.loads(cleaned)
        judge_answer = int(obj.get("answer", 0))
    except (json.JSONDecodeError, ValueError):
        judge_answer = 0
        for ch in reversed(content):
            if ch in ("1", "2"):
                judge_answer = int(ch)
                break

    return judge_answer, content


def judge_pairwise_double_pass(
    questions: List[str],
    responses_human: List[str],
    responses_ai: List[str],
    seed: int = JUDGE_SEED,
) -> tuple:
    """
    Double-pass judging: each question is judged twice with swapped response order.

    Pass 1: human_first (Response 1 = human-steered, Response 2 = AI-steered)
    Pass 2: ai_first    (Response 1 = AI-steered,    Response 2 = human-steered)

    Target type is randomized per question (seeded) but consistent across passes.
    This controls for GPT-judge position bias and doubles effective N.
    """
    assert len(questions) == len(responses_human) == len(responses_ai)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Seed RNG for reproducible target_type assignment
    rng = np.random.RandomState(seed)
    target_types = ["human" if rng.randint(2) == 0 else "ai" for _ in questions]

    judge_details = []
    n_correct_pass1 = 0
    n_correct_pass2 = 0
    n_correct_hf = 0  # human_first correct
    n_correct_af = 0  # ai_first correct
    n_total_hf = 0
    n_total_af = 0

    for idx, (q, r_h, r_a) in enumerate(tqdm(
        list(zip(questions, responses_human, responses_ai)), desc="GPT judging"
    )):
        d = target_types[idx]

        # --- Pass 1: human_first ---
        correct_p1 = 1 if d == "human" else 2
        ans_p1, raw_p1 = _single_judge_call(q, r_h, r_a, d)
        is_correct_p1 = ans_p1 == correct_p1
        if is_correct_p1:
            n_correct_pass1 += 1
            n_correct_hf += 1
        n_total_hf += 1

        judge_details.append({
            "question_idx": idx,
            "question": q,
            "target_type": d,
            "response_order": "human_first",
            "pass": 1,
            "correct_answer": correct_p1,
            "judge_answer": ans_p1,
            "is_correct": is_correct_p1,
            "judge_raw": raw_p1,
        })

        # --- Pass 2: ai_first (swapped) ---
        correct_p2 = 2 if d == "human" else 1
        ans_p2, raw_p2 = _single_judge_call(q, r_a, r_h, d)
        is_correct_p2 = ans_p2 == correct_p2
        if is_correct_p2:
            n_correct_pass2 += 1
            n_correct_af += 1
        n_total_af += 1

        judge_details.append({
            "question_idx": idx,
            "question": q,
            "target_type": d,
            "response_order": "ai_first",
            "pass": 2,
            "correct_answer": correct_p2,
            "judge_answer": ans_p2,
            "is_correct": is_correct_p2,
            "judge_raw": raw_p2,
        })

    n_total = len(questions) * 2
    n_correct_total = n_correct_pass1 + n_correct_pass2
    success_rate = n_correct_total / n_total

    # Binomial test vs chance (50%)
    binom_result = scipy_stats.binomtest(n_correct_total, n_total, 0.5, alternative='greater')

    # Per-question consistency: both passes agree
    n_both_correct = sum(
        1 for i in range(len(questions))
        if judge_details[2*i]["is_correct"] and judge_details[2*i+1]["is_correct"]
    )
    n_both_wrong = sum(
        1 for i in range(len(questions))
        if not judge_details[2*i]["is_correct"] and not judge_details[2*i+1]["is_correct"]
    )
    n_inconsistent = len(questions) - n_both_correct - n_both_wrong

    print(f"\nGPT judge results (double-pass):")
    print(f"  Overall success rate: {success_rate:.3f} ({n_correct_total}/{n_total})")
    print(f"  Pass 1 (human_first): {n_correct_pass1}/{len(questions)} = {n_correct_pass1/len(questions):.3f}")
    print(f"  Pass 2 (ai_first):    {n_correct_pass2}/{len(questions)} = {n_correct_pass2/len(questions):.3f}")
    print(f"  Position bias check:  human_first={n_correct_hf}/{n_total_hf}, ai_first={n_correct_af}/{n_total_af}")
    print(f"  Per-question: both_correct={n_both_correct}, both_wrong={n_both_wrong}, inconsistent={n_inconsistent}")
    print(f"  Binomial test vs chance: p={binom_result.pvalue:.4f}")

    summary_stats = {
        "overall_success_rate": success_rate,
        "n_correct": n_correct_total,
        "n_total": n_total,
        "pass1_success_rate": n_correct_pass1 / len(questions),
        "pass2_success_rate": n_correct_pass2 / len(questions),
        "position_bias": {
            "human_first_correct": n_correct_hf,
            "human_first_total": n_total_hf,
            "ai_first_correct": n_correct_af,
            "ai_first_total": n_total_af,
        },
        "per_question_consistency": {
            "both_correct": n_both_correct,
            "both_wrong": n_both_wrong,
            "inconsistent": n_inconsistent,
        },
        "binomial_test_pvalue": float(binom_result.pvalue),
        "judge_seed": seed,
    }

    return success_rate, judge_details, summary_stats


# ========================== SAVE HELPERS ========================== #

def save_results(
    result_dir, probe_label, questions, baseline_responses,
    human_responses, ai_responses, probes_by_layer,
    success_rate, judge_details, summary_stats,
):
    """Save CSV, JSON, and TXT for one probe-type run."""
    os.makedirs(result_dir, exist_ok=True)

    # CSV
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
    print(f"  Saved responses to {csv_path}")

    # JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "exp2b_naturalistic",
        "probe_type": probe_label,
        "config": {
            "model": "LLaMA-2-Chat-13B",
            "model_path": MODEL_NAME,
            "from_layer": FROM_LAYER,
            "to_layer": TO_LAYER,
            "intervention_strength": INTERVENTION_STRENGTH,
            "min_probe_accuracy": MIN_PROBE_ACCURACY,
            "probe_layers_used": sorted(
                k for k in probes_by_layer if FROM_LAYER < k <= TO_LAYER
            ),
            "all_probe_layers_loaded": sorted(probes_by_layer.keys()),
            "max_new_tokens": MAX_NEW_TOKENS,
            "gen_temperature": GEN_TEMPERATURE,
            "gen_top_p": GEN_TOP_P,
            "gen_do_sample": GEN_DO_SAMPLE,
            "n_questions": len(questions),
            "judge_model": JUDGE_MODEL,
            "judge_seed": JUDGE_SEED,
            "double_pass": True,
        },
        "judge_success_rate": success_rate,
        "summary_stats": summary_stats,
        "judge_details": judge_details,
    }
    json_path = os.path.join(result_dir, "intervention_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {json_path}")

    # TXT examples
    out_txt = os.path.join(result_dir, "human_ai_causal_examples.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Probe type: {probe_label}\n")
        f.write(f"N = {INTERVENTION_STRENGTH}, temp = {GEN_TEMPERATURE}\n")
        f.write(f"Questions: {len(questions)}, Double-pass judge: yes\n")
        f.write(f"Overall success rate: {success_rate:.3f}\n")
        f.write(f"Binomial p-value: {summary_stats['binomial_test_pvalue']:.4f}\n")
        f.write("=" * 80 + "\n\n")
        for i in range(len(questions)):
            text = f"Q{i}: {questions[i]}\n\n"
            text += "-" * 50 + "\n"
            text += "Baseline (no intervention):\n"
            text += f"ASSISTANT: {baseline_responses[i]}\n\n"
            text += "-" * 50 + "\n"
            text += f"Intervened ({probe_label}): steer toward HUMAN\n"
            text += f"ASSISTANT: {human_responses[i]}\n\n"
            text += "-" * 50 + "\n"
            text += f"Intervened ({probe_label}): steer toward AI\n"
            text += f"ASSISTANT: {ai_responses[i]}\n\n"
            text += "=" * 80 + "\n\n"
            f.write(text)
    print(f"  Saved examples to {out_txt}")


# ========================== MAIN ========================== #

def run_one_probe_type(
    model, tokenizer, questions, baseline_responses,
    modified_layer_names, probe_run_config,
):
    """Run steering + judging for a single probe type."""
    label = probe_run_config["label"]
    probe_dir = probe_run_config["probe_dir"]
    acc_lookup = probe_run_config["accuracy_lookup"]
    result_dir = os.path.join(RESULT_DIR_BASE, probe_run_config["result_subdir"])

    print(f"\n{'='*60}")
    print(f"  PROBE TYPE: {label}")
    print(f"{'='*60}")

    # Load probes
    probes_by_layer = load_probes(
        probe_dir, DEVICE, input_dim=INPUT_DIM,
        min_accuracy=MIN_PROBE_ACCURACY,
        accuracy_lookup=acc_lookup,
        label=label,
    )
    if not probes_by_layer:
        print(f"  [SKIP] No probes loaded for {label}, skipping.")
        return None

    # Log config
    active_probes = sorted(k for k in probes_by_layer if FROM_LAYER < k <= TO_LAYER)
    print(f"\n  Intervention config ({label}):")
    print(f"    Layer range: model.layers.{FROM_LAYER} to model.layers.{TO_LAYER-1}")
    print(f"    Probe indices in range: {active_probes}")
    print(f"    Total probes loaded: {len(probes_by_layer)}")
    print(f"    N = {INTERVENTION_STRENGTH}")

    # Human-steered
    print(f"\n=== Generating human-steered responses ({label}) ===")
    human_edit_fn = make_edit_fn(probes_by_layer, torch.tensor([1.0]), INTERVENTION_STRENGTH)
    human_responses = collect_responses_batched(
        model, tokenizer, questions,
        modified_layer_names, human_edit_fn, batch_size=5,
    )

    # AI-steered
    print(f"\n=== Generating AI-steered responses ({label}) ===")
    ai_edit_fn = make_edit_fn(probes_by_layer, torch.tensor([-1.0]), INTERVENTION_STRENGTH)
    ai_responses = collect_responses_batched(
        model, tokenizer, questions,
        modified_layer_names, ai_edit_fn, batch_size=5,
    )

    # GPT judge (double-pass, seeded)
    print(f"\n=== GPT judge ({label}, double-pass) ===")
    success_rate, judge_details, summary_stats = judge_pairwise_double_pass(
        questions, human_responses, ai_responses, seed=JUDGE_SEED,
    )

    # Save
    save_results(
        result_dir, label, questions, baseline_responses,
        human_responses, ai_responses, probes_by_layer,
        success_rate, judge_details, summary_stats,
    )

    print(f"\n  {label} final GPT-judge success rate: {success_rate:.3f}")
    return success_rate


def main():
    os.makedirs(RESULT_DIR_BASE, exist_ok=True)

    # --- Load model & tokenizer (once) ---
    print("Loading LLaMA-2-Chat-13B model and tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=True, padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(TORCH_DEVICE).eval()

    if "<pad>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- Load questions (once) ---
    with open(CAUSAL_QUESTION_PATH, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(questions)} causal test prompts.")

    # --- Layer names (once) ---
    modified_layer_names = get_modified_layer_names(model, FROM_LAYER, TO_LAYER)

    # --- Baseline (once, shared across both probe runs) ---
    def null_edit(output, layer_name):
        return output

    print("\n=== Generating baseline (unintervened) responses ===")
    baseline_responses = collect_responses_batched(
        model, tokenizer, questions,
        modified_layer_names=[], edit_function=null_edit, batch_size=5,
    )

    # --- Run each probe type ---
    summary = {}
    for probe_run in PROBE_RUNS:
        rate = run_one_probe_type(
            model, tokenizer, questions, baseline_responses,
            modified_layer_names, probe_run,
        )
        if rate is not None:
            summary[probe_run["label"]] = rate

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("  SUMMARY (double-pass judge, seeded)")
    print(f"{'='*60}")
    for label, rate in summary.items():
        print(f"  {label:20s}  judge success = {rate:.3f}")
    print()


if __name__ == "__main__":
    main()