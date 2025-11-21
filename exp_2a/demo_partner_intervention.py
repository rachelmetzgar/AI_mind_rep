#!/usr/bin/env python3
"""
Minimal demo: intervene on LLaMA-2-7B's internal partner representation.

- Loads LLaMA-2-7B
- Loads Human vs AI *control probes* (partner type)
- Intervenes on a chosen layer range
- Prints baseline vs HUMAN-steered vs AI-steered responses for a small list of prompts.

Env: llama2_env

Rachel C. Metzgar · Nov 2025
"""

import os
from typing import List, Dict

import torch
from torch import nn
from tqdm.auto import tqdm
from baukit import TraceDict
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.dataset import llama_v2_prompt
from src.probes import LinearProbeClassification


# ========================== CONFIG ========================== #

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
)

CONTROL_PROBE_DIR = "data/probe_checkpoints/control_probe"
INPUT_DIM = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DEVICE = DEVICE

# Intervention hyperparameters
FROM_LAYER = 20
TO_LAYER = 30
INTERVENTION_STRENGTH = 8
USE_RESIDUAL_STREAM = True

MAX_NEW_TOKENS = 256
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 1.0


# ========================== HELPERS ========================== #

def load_partner_control_probes(
    probe_dir: str,
    device: str,
    input_dim: int = INPUT_DIM,
) -> Dict[int, LinearProbeClassification]:
    probes: Dict[int, LinearProbeClassification] = {}

    for fname in os.listdir(probe_dir):
        if not fname.endswith(".pth"):
            continue
        if not fname.startswith("human_ai_probe_at_layer_"):
            continue
        if fname.endswith("_final.pth"):
            continue

        layer_str = fname.split("_layer_")[-1].split(".pth")[0]
        layer_idx = int(layer_str)

        probe = LinearProbeClassification(
            device=device,
            probe_class=1,
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


def optimize_one_inter_rep(
    inter_rep: torch.Tensor,
    cf_target: torch.Tensor,
    probe: LinearProbeClassification,
    N: float = INTERVENTION_STRENGTH,
    normalized: bool = False,
) -> torch.Tensor:
    tensor = inter_rep.to(TORCH_DEVICE).to(torch.float32)
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
    def edit_inter_rep_multi_layers(output, layer_name: str):
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
            cur = last_tok[i : i + 1, :]
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
    batch_size: int = 2,
) -> List[str]:
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
            if "[/INST]" in text:
                text = text.split("[/INST]", 1)[1]
            responses.append(text.strip())

    return responses


# ========================== MAIN DEMO ========================== #

def main():
    print("Loading LLaMA-2-7B model and tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(TORCH_DEVICE).eval()

    if "<pad>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    probes_by_layer = load_partner_control_probes(CONTROL_PROBE_DIR, DEVICE, input_dim=INPUT_DIM)
    modified_layer_names = get_modified_layer_names(
        model,
        from_idx=FROM_LAYER,
        to_idx=TO_LAYER,
        residual=USE_RESIDUAL_STREAM,
    )

    # A few toy prompts – replace with whatever you care about
    questions = [
        "How should I phrase my message to the assistant so it understands I'm a human user?",
        "What would an AI user ask an AI assistant in a debugging scenario?",
        "I'm chatting with a bot that might be a human or an AI. What should I say next?",
    ]

    # Baseline
    def null_edit(output, layer_name: str):
        return output

    print("\n=== Baseline (no intervention) ===")
    baseline = collect_responses_batched(
        model,
        tokenizer,
        questions,
        modified_layer_names=[],
        edit_function=null_edit,
        batch_size=2,
    )

    # Human-steered
    print("\n=== Steer toward HUMAN partner representation ===")
    cf_target_human = torch.tensor([1.0])
    human_edit_fn = make_edit_fn(
        probes_by_layer,
        cf_target=cf_target_human,
        N=INTERVENTION_STRENGTH,
        residual=USE_RESIDUAL_STREAM,
    )
    human = collect_responses_batched(
        model,
        tokenizer,
        questions,
        modified_layer_names=modified_layer_names,
        edit_function=human_edit_fn,
        batch_size=2,
    )

    # AI-steered
    print("\n=== Steer toward AI partner representation ===")
    cf_target_ai = torch.tensor([-1.0])
    ai = collect_responses_batched(
        model,
        tokenizer,
        questions,
        modified_layer_names=modified_layer_names,
        edit_function=human_edit_fn.__class__(  # just to keep structure similar; we override below
        ),
        batch_size=2,
    )
    # Oops: easier to just re-create edit_fn for AI:
    ai_edit_fn = make_edit_fn(
        probes_by_layer,
        cf_target=cf_target_ai,
        N=INTERVENTION_STRENGTH,
        residual=USE_RESIDUAL_STREAM,
    )
    ai = collect_responses_batched(
        model,
        tokenizer,
        questions,
        modified_layer_names=modified_layer_names,
        edit_function=ai_edit_fn,
        batch_size=2,
    )

    # Print nicely
    print("\n\n=== SAMPLE COMPARISONS ===\n")
    for q, b, h, a in zip(questions, baseline, human, ai):
        print("USER:", q)
        print("\n--- Baseline ---")
        print("ASSISTANT:", b)
        print("\n--- Steered: HUMAN partner ---")
        print("ASSISTANT:", h)
        print("\n--- Steered: AI partner ---")
        print("ASSISTANT:", a)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
