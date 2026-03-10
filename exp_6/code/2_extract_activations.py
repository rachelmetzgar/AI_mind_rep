"""Extract hidden-state activations at agent name token positions from LLaMA-2-13B-Chat.

For each narrative, tokenizes the text within a LLaMA-2 chat template, runs a
single forward pass with hooks on all 40 transformer layers, and saves the
hidden state at each agent name's last token position.

Output: one .pt file per narrative in config.ACTIVATIONS_DIR.
"""

import os, sys, json
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.utils import load_stimuli, find_agent_token_positions
from transformers import AutoTokenizer, AutoModelForCausalLM


def wrap_in_chat_template(narrative_text: str) -> str:
    """Wrap narrative in LLaMA-2 chat template."""
    return (
        f"[INST] <<SYS>>\n"
        f"You are a helpful assistant.\n"
        f"<</SYS>>\n\n"
        f"{narrative_text} [/INST]"
    )


def extract_activations(narrative, model, tokenizer):
    """Extract hidden states at agent name positions for a single narrative.

    Returns:
        dict with narrative_id, agent_activations, and token_positions
    """
    # Prepare prompt
    prompt = wrap_in_chat_template(narrative["narrative_text"])
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    input_ids_list = input_ids[0].tolist()

    # Find token positions for each agent name
    names = narrative["names"]
    extraction_sentence = narrative["extraction_sentence"]
    token_positions = find_agent_token_positions(
        tokenizer, input_ids_list, names, extraction_sentence
    )

    # Set up hooks on all transformer layers
    activations = {}
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        def hook_fn(module, input, output, idx=layer_idx):
            activations[idx] = output[0].detach().cpu()
        hooks.append(layer.register_forward_hook(hook_fn))

    # Single forward pass
    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Extract hidden states at agent name positions
    agent_activations = {}
    for name in names:
        pos = token_positions[name]
        agent_activations[name] = {
            layer_idx: activations[layer_idx][0, pos, :]
            for layer_idx in range(config.NUM_LAYERS)
        }

    return {
        "narrative_id": narrative["narrative_id"],
        "agent_activations": agent_activations,
        "token_positions": token_positions,
    }


def verify_tokenization(narrative, tokenizer, n_to_show=None):
    """Print tokenization details for verification."""
    prompt = wrap_in_chat_template(narrative["narrative_text"])
    input_ids = tokenizer.encode(prompt)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    names = narrative["names"]
    extraction_sentence = narrative["extraction_sentence"]

    positions = find_agent_token_positions(
        tokenizer, input_ids, names, extraction_sentence
    )

    print(f"\n--- Verification: {narrative['narrative_id']} ---")
    print(f"  Total tokens: {len(input_ids)}")
    print(f"  Agents: {names}")
    print(f"  Extraction sentence: {extraction_sentence[:80]}...")
    for name, pos in positions.items():
        # Show a few tokens around the position for context
        start = max(0, pos - 2)
        end = min(len(tokens), pos + 3)
        context_tokens = tokens[start:end]
        print(f"  {name}: position {pos}, context tokens: {context_tokens}")
    print()


def main():
    # Load stimuli
    print(f"Loading stimuli from {config.STIMULI_PATH}")
    stimuli = load_stimuli(config.STIMULI_PATH)
    print(f"Loaded {len(stimuli)} narratives")

    # Check which narratives already have saved activations
    os.makedirs(config.ACTIVATIONS_DIR, exist_ok=True)
    to_process = []
    skipped = 0
    for narrative in stimuli:
        out_path = os.path.join(config.ACTIVATIONS_DIR, f"{narrative['narrative_id']}.pt")
        if os.path.exists(out_path):
            skipped += 1
        else:
            to_process.append(narrative)

    if skipped > 0:
        print(f"Skipping {skipped} narratives with existing .pt files")

    if len(to_process) == 0:
        print("All narratives already processed. Nothing to do.")
        return

    print(f"Will process {len(to_process)} narratives")

    # Load tokenizer
    print(f"Loading tokenizer from {config.MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)

    # Tokenization verification on first 3 narratives
    print("\n=== Tokenization Verification ===")
    for narrative in stimuli[:3]:
        verify_tokenization(narrative, tokenizer)

    # Load model
    print(f"Loading model from {config.MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

    # Extract activations
    saved_count = 0
    for narrative in tqdm(to_process, desc="Extracting activations"):
        out_path = os.path.join(config.ACTIVATIONS_DIR, f"{narrative['narrative_id']}.pt")

        try:
            result = extract_activations(narrative, model, tokenizer)
            torch.save(result, out_path)
            saved_count += 1
        except Exception as e:
            print(f"\nERROR processing {narrative['narrative_id']}: {e}")
            continue

    print(f"\n=== Done ===")
    print(f"Saved {saved_count} activation files to {config.ACTIVATIONS_DIR}")
    print(f"Total files in directory: {len([f for f in os.listdir(config.ACTIVATIONS_DIR) if f.endswith('.pt')])}")

    # Spot-check tensor shapes
    print("\n=== Spot-check tensor shapes ===")
    pt_files = sorted([f for f in os.listdir(config.ACTIVATIONS_DIR) if f.endswith(".pt")])
    for fname in pt_files[:3]:
        fpath = os.path.join(config.ACTIVATIONS_DIR, fname)
        data = torch.load(fpath, map_location="cpu")
        print(f"\n  {fname}:")
        print(f"    narrative_id: {data['narrative_id']}")
        print(f"    agents: {list(data['agent_activations'].keys())}")
        print(f"    token_positions: {data['token_positions']}")
        first_agent = list(data["agent_activations"].keys())[0]
        print(f"    layers for '{first_agent}': {len(data['agent_activations'][first_agent])}")
        sample_tensor = data["agent_activations"][first_agent][0]
        print(f"    layer 0 shape: {sample_tensor.shape} (expected [{config.HIDDEN_DIM}])")
        assert sample_tensor.shape == (config.HIDDEN_DIM,), (
            f"Unexpected shape {sample_tensor.shape}, expected ({config.HIDDEN_DIM},)"
        )
    print("\nAll spot-checks passed.")


if __name__ == "__main__":
    main()
