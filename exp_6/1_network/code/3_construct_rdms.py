"""
Construct RDMs from extracted activations.
==========================================
For each narrative and each layer:
1. Load activation file from config.ACTIVATIONS_DIR/{narrative_id}.pt
2. Get 4 agent activation vectors (shape [5120])
3. Compute pairwise cosine distance: 1 - cosine_similarity(vi, vj)
4. Store as upper triangle (6 values)
5. Construct epistemic, communication, and position candidate RDMs from stimulus metadata
6. Save everything to config.RDMS_DIR/model_rdms.pkl
"""

import os, sys, json, pickle
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.utils import (load_stimuli, cosine_distance, epistemic_rdm_from_dict,
                         compute_communication_rdm, compute_position_rdm, extract_upper_triangle)


def build_model_rdm(activations: dict, names: list, layer_idx: int) -> np.ndarray:
    """Build a 4x4 model RDM from agent activations at a given layer.

    Args:
        activations: Dict mapping agent_name -> {layer_idx: np.array of shape [5120]}
        names: Ordered list of 4 agent names
        layer_idx: Which layer to use

    Returns:
        Upper triangle (6 values) of cosine distance matrix.
    """
    n = len(names)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            vi = activations[names[i]][layer_idx]
            vj = activations[names[j]][layer_idx]
            dist = cosine_distance(vi, vj)
            rdm[i, j] = dist
            rdm[j, i] = dist
    return extract_upper_triangle(rdm)


def main():
    # Load stimuli
    stimuli = load_stimuli(config.STIMULI_PATH)
    print(f"Loaded {len(stimuli)} stimuli")

    all_rdms = {}
    missing = []

    for stim in tqdm(stimuli, desc="Constructing RDMs"):
        narrative_id = stim["narrative_id"]
        names = stim["names"]
        topology = stim["topology"]
        condition = stim["condition"]

        # Load activations for this narrative
        act_path = os.path.join(config.ACTIVATIONS_DIR, f"{narrative_id}.pt")
        if not os.path.exists(act_path):
            missing.append(narrative_id)
            continue

        act_data = torch.load(act_path, map_location="cpu")
        # act_data structure: {"narrative_id": ..., "agent_activations": {name: {layer: tensor}}, ...}
        agent_acts = act_data["agent_activations"]
        # Convert tensors to numpy
        activations = {}
        for name in names:
            activations[name] = {}
            for layer_idx in range(config.NUM_LAYERS):
                vec = agent_acts[name][layer_idx]
                if isinstance(vec, torch.Tensor):
                    vec = vec.numpy()
                activations[name][layer_idx] = vec.astype(np.float64)

        # Build model RDMs for each layer
        model_rdm_by_layer = {}
        for layer_idx in range(config.NUM_LAYERS):
            model_rdm_by_layer[layer_idx] = build_model_rdm(activations, names, layer_idx)

        # Build candidate RDMs from stimulus metadata
        epistemic_rdm = epistemic_rdm_from_dict(stim["epistemic_rdm"], names)
        communication_rdm = compute_communication_rdm(topology)
        position_rdm = compute_position_rdm()

        all_rdms[narrative_id] = {
            "model_rdm": model_rdm_by_layer,
            "epistemic_rdm": epistemic_rdm,
            "communication_rdm": communication_rdm,
            "position_rdm": position_rdm,
            "condition": condition,
            "topology": topology,
            "names": names,
        }

    if missing:
        print(f"WARNING: Missing activation files for {len(missing)} narratives:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # Save
    out_path = os.path.join(config.RDMS_DIR, "model_rdms.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(all_rdms, f, protocol=4)

    print(f"\nSaved RDMs for {len(all_rdms)} narratives to {out_path}")
    print(f"Each narrative has model RDMs for {config.NUM_LAYERS} layers")

    # Quick sanity check on one narrative
    if all_rdms:
        sample_id = next(iter(all_rdms))
        sample = all_rdms[sample_id]
        print(f"\nSanity check ({sample_id}):")
        print(f"  Model RDM layer 20: {sample['model_rdm'][20]}")
        print(f"  Epistemic RDM:      {sample['epistemic_rdm']}")
        print(f"  Communication RDM:  {sample['communication_rdm']}")
        print(f"  Position RDM:       {sample['position_rdm']}")


if __name__ == "__main__":
    main()
