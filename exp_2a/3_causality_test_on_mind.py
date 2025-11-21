#!/usr/bin/env python3
"""
Phase 5: Causal intervention test for Human vs AI partner attribute.

This script:
- Loads LLaMA-2-7B (local snapshot)
- Loads pre-trained *control probes* for partner type (AI vs Human)
- Uses BAUKIT's TraceDict to shift residual activations along the probe direction
- Generates:
    - baseline (no intervention)
    - "human-partner" steered responses
    - "ai-partner" steered responses
- Uses GPT-4(o)-style judging to evaluate which response better matches
  a target partner type, reproducing the TalkTuner-style causality check.

Attribute: conversation partner type
    label 0 = AI partner
    label 1 = Human partner

Env: llama2_env
Replicating TalkTuner (Chen et al., 2024) causality methodology for Human vs AI.

Rachel C. Metzgar · Nov 2025
"""

import os
import json
import numpy as np
from typing import List, Dict

import torch
from torch import nn
from tqdm.auto import tqdm
from baukit import TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai

# --- Local imports ---
from src.dataset import llama_v2_prompt
from src.probes import LinearProbeClassification


# ========================== CONFIG ========================== #

# LLaMA-2-7B local snapshot (same as training script)
MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
)

CONTROL_PROBE_DIR = "data/probe_checkpoints/control_probe"
CAUSAL_QUESTION_PATH = "data/causality_test_questions/human_ai.txt"
RESULT_DIR = "data/intervention_results/partner"

INPUT_DIM = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = DEVICE

# Intervention hyperparameters
FROM_LAYER = 20     # inclusive
TO_LAYER = 30       # exclusive
INTERVENTION_STRENGTH = 8  # N
USE_RESIDUAL_STREAM = True

# Generation hyperparameters
MAX_NEW_TOKENS = 768
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 1.0

# GPT judge model
JUDGE_MODEL = "gpt-4o-mini"  # or "gpt-4o", "gpt-4-turbo" etc.


# ========================== PROBE LOADING ========================== #

def load_partner_control_probes(
    probe_dir: str,
    device: str,
    input_dim: int = INPUT_DIM,
) -> Dict[int, LinearProbeClassification]:
    """
    Load control probes trained for Human vs AI partner classification.

    Expects files like:
        human_ai_probe_at_layer_{L}.pth
    in probe_dir.

    Returns:
        dict: layer_index -> LinearProbeClassification (probe_class=1, logistic=True)
    """
    probes: Dict[int, LinearProbeClassification] = {}

    for fname in os.listdir(probe_dir):
        if not fname.endswith(".pth"):
            continue
        if not fname.startswith("human_ai_probe_at_layer_"):
            continue
        # skip "final" checkpoints for now; we want best-val versions
        if fname.endswith("_final.pth"):
            continue

        layer_str = fname.split("_layer_")[-1].split(".pth")[0]
        layer_idx = int(layer_str)

        probe = LinearProbeClassification(
            device=device,
            probe_class=1,          # binary logistic
            input_dim=input_dim,
            logistic=True,
        )
        state = torch.load(os.path.join(probe_dir, fname), map_location=device)
        probe.load_state_dict(state)
        probe.eval()
        probes[layer_idx] = probe

    if not probes:
        raise RuntimeError(f"No matching probes found in {probe_dir}")

    print(f"Loaded {len(probes)} control probes from {probe_dir}")
    return probes


# ========================== INTERVENTION UTILS ========================== #

def optimize_one_inter_rep(
    inter_rep: torch.Tensor,
    cf_target: torch.Tensor,
    probe: LinearProbeClassification,
    N: float = INTERVENTION_STRENGTH,
    normalized: bool = False,
) -> torch.Tensor:
    """
    One-step update of a single hidden representation.

    inter_rep : [1, hidden_dim]  (float16 or float32)
    cf_target : scalar tensor, shape [1] or [1,1], values +1 (human) or -1 (AI)
    probe     : linear probe with proj[0].weight of shape [1, hidden_dim]

    Returns:
        updated_rep : [1, hidden_dim] float32
    """
    tensor = inter_rep.to(TORCH_DEVICE).to(torch.float32)  # avoid fp16 overflow
    w = probe.proj[0].weight.to(TORCH_DEVICE).to(torch.float32)  # [1, dim]
    target = cf_target.to(TORCH_DEVICE).view(1, -1).to(torch.float32)  # [1,1]

    direction = target @ w  # [1, dim]
    if normalized:
        direction = direction * (100.0 / (tensor.norm() + 1e-8))
    updated = tensor + N * direction
    return updated


def make_edit_fn(
    probes_by_layer: Dict[int, LinearProbeClassification],
    cf_target: torch.Tensor,
    N: float,
    residual: bool = True,
):
    """
    Return an edit_output function suitable for TraceDict, which:
    - Finds the layer index from the module name
    - Looks up the corresponding control probe
    - Shifts the last-token residual along the probe direction
    """

    def edit_inter_rep_multi_layers(output, layer_name: str):
        # Expect names like "model.layers.20"
        if "model.layers." not in layer_name:
            return output

        layer_str = layer_name.split("model.layers.")[-1].split(".")[0]
        try:
            layer_num = int(layer_str)
        except ValueError:
            return output

        # Probe index is +1 because hidden_states[0] is embedding
        probe_idx = layer_num + 1
        if probe_idx not in probes_by_layer:
            return output

        probe = probes_by_layer[probe_idx]

        # output[0] is the hidden_states from this layer: [batch, seq, dim]
        hidden = output[0]
        if hidden.ndim != 3:
            return output

        last_tok = hidden[:, -1, :]  # [B, dim]
        updated_batch = []

        for i in range(last_tok.size(0)):
            cur = last_tok[i : i + 1, :]  # [1, dim]
            updated = optimize_one_inter_rep(cur, cf_target, probe, N=N, normalized=False)
            updated_batch.append(updated)

        updated_batch = torch.cat(updated_batch, dim=0).to(hidden.dtype)
        hidden[:, -1, :] = updated_batch
        output = list(output)
        output[0] = hidden
        return tuple(output)

    return edit_inter_rep_multi_layers


def get_modified_layer_names(
    model: nn.Module,
    from_idx: int,
    to_idx: int,
    residual: bool = True,
) -> List[str]:
    """
    Collect names of modules to be intervened on. For LLaMA-2, intervening on
    'model.layers.{i}' gives you residual-stream-like activations per block.
    """
    which_layers: List[str] = []
    for name, module in model.named_modules():
        if residual and name.startswith("model.layers.") and name.split(".")[-1].isdigit():
            layer_num = int(name.split("model.layers.")[-1])
            if from_idx <= layer_num < to_idx:
                which_layers.append(name)
    return which_layers


def collect_responses_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    modified_layer_names: List[str],
    edit_function,
    batch_size: int = 5,
) -> List[str]:
    """
    Generate responses with optional interventions on a set of prompts.
    """
    print(f"Intervening on layers: {modified_layer_names}")
    responses: List[str] = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i : i + batch_size]
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
                    do_sample=False,
                    temperature=GEN_TEMPERATURE,
                    top_p=GEN_TOP_P,
                )

        for seq in tokens:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            # Split off the assistant part after [/INST]
            if "[/INST]" in text:
                text = text.split("[/INST]", 1)[1]
            responses.append(text.strip())

    return responses


# ========================== GPT JUDGE HELPERS ========================== #

def judge_pairwise(
    questions: List[str],
    responses_human: List[str],
    responses_ai: List[str],
) -> float:
    """
    Use GPT as a judge: for each question, we show two responses
    (human-steered vs ai-steered) and ask which better matches a specified
    partner type. Inspired by TalkTuner's GPT-based causal evaluation.
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

    def formatter(prompt: str, r1: str, r2: str) -> str:
        return (
            f"Prompt: {prompt}\n\n"
            f"Response 1: {r1}\n\n"
            f"Response 2: {r2}\n\n"
            "Answer:"
        )

    results = []
    where_correct = []

    for q, r_h, r_a in tqdm(list(zip(questions, responses_human, responses_ai)), desc="GPT judging"):
        # Present human-steered as response1, ai-steered as response2
        # Randomly choose which partner type GPT should look for
        d = "human" if np.random.randint(2) == 0 else "ai"
        instruction = template.format(partner_type=d)
        query = formatter(q, r_h, r_a)
        correct_answer = 1 if d == "human" else 2
        where_correct.append(correct_answer)

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
        results.append(content.strip())

    where_correct = np.array(where_correct)

    def process(entry: str) -> dict:
        entry_cleaned = entry.strip()
        if entry_cleaned.startswith("```json"):
            entry_cleaned = entry_cleaned.removeprefix("```json").removeprefix("\n")
        if entry_cleaned.endswith("```"):
            entry_cleaned = entry_cleaned.removesuffix("```").rstrip("\n")
        try:
            obj = json.loads(entry_cleaned)
        except json.JSONDecodeError:
            # Very crude fallback: look for the last digit 1 or 2
            for ch in reversed(entry_cleaned):
                if ch in ("1", "2"):
                    return {"answer": ch}
            raise
        return obj

    processed = np.array([int(process(entry).get("answer", 0)) for entry in results])
    success = (processed == where_correct).mean()
    print(f"GPT judge success rate (0–1): {success:.3f}")
    return float(success)


# ========================== MAIN ========================== #

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # --- Load model & tokenizer ---
    print("Loading LLaMA-2-7B model and tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(TORCH_DEVICE).eval()

    # Ensure pad token is set
    if "<pad>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- Load control probes ---
    probes_by_layer = load_partner_control_probes(CONTROL_PROBE_DIR, DEVICE, input_dim=INPUT_DIM)

    # --- Load questions ---
    with open(CAUSAL_QUESTION_PATH, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(questions)} causal test prompts.")

    # --- Which layers to intervene on ---
    modified_layer_names = get_modified_layer_names(
        model,
        from_idx=FROM_LAYER,
        to_idx=TO_LAYER,
        residual=USE_RESIDUAL_STREAM,
    )

    # --- Baseline (no intervention) ---
    def null_edit(output, layer_name):
        return output

    print("\n=== Generating baseline (unintervened) responses ===")
    baseline_responses = collect_responses_batched(
        model,
        tokenizer,
        questions,
        modified_layer_names=[],
        edit_function=null_edit,
        batch_size=10,
    )

    # --- Human-steered responses (cf_target = +1) ---
    print("\n=== Generating human-steered responses ===")
    cf_target_human = torch.tensor([1.0])  # steer along +w (label=1=human)
    human_edit_fn = make_edit_fn(
        probes_by_layer,
        cf_target=cf_target_human,
        N=INTERVENTION_STRENGTH,
        residual=USE_RESIDUAL_STREAM,
    )
    human_responses = collect_responses_batched(
        model,
        tokenizer,
        questions,
        modified_layer_names=modified_layer_names,
        edit_function=human_edit_fn,
        batch_size=10,
    )

    # --- AI-steered responses (cf_target = -1) ---
    print("\n=== Generating AI-steered responses ===")
    cf_target_ai = torch.tensor([-1.0])  # steer along -w (label=0=AI)
    ai_edit_fn = make_edit_fn(
        probes_by_layer,
        cf_target=cf_target_ai,
        N=INTERVENTION_STRENGTH,
        residual=USE_RESIDUAL_STREAM,
    )
    ai_responses = collect_responses_batched(
        model,
        tokenizer,
        questions,
        modified_layer_names=modified_layer_names,
        edit_function=ai_edit_fn,
        batch_size=10,
    )

    # --- Evaluate with GPT judge ---
    print("\n=== GPT-based causal evaluation (Human vs AI partner) ===")
    success_rate = judge_pairwise(questions, human_responses, ai_responses)

    # --- Save a subset of examples ---
    out_txt = os.path.join(RESULT_DIR, "human_ai_causal_examples.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for i in range(min(30, len(questions))):
            text = f"USER: {questions[i]}\n\n"
            text += "-" * 50 + "\n"
            text += "Baseline (no intervention):\n"
            text += f"ASSISTANT: {baseline_responses[i]}\n\n"
            text += "-" * 50 + "\n"
            text += "Intervened: steer toward HUMAN partner representation\n"
            text += f"ASSISTANT: {human_responses[i]}\n\n"
            text += "-" * 50 + "\n"
            text += "Intervened: steer toward AI partner representation\n"
            text += f"ASSISTANT: {ai_responses[i]}\n\n"
            text += "=" * 80 + "\n\n"
            f.write(text)

    print(f"\nSaved example interventions to {out_txt}")
    print(f"Final GPT-judge success rate: {success_rate:.3f}")


if __name__ == "__main__":
    main()
