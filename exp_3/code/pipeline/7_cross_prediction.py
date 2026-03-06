#!/usr/bin/env python3
"""
Experiment 3, Phase 5: Cross-Prediction & Representational Alignment

Tests whether concept-of-mind representations (Exp 3) and conversational
partner-identity representations (Exp 2b) share structure, using three
complementary analyses:

    1. Cosine alignment: cosine similarity between concept and conversational
       probe weight vectors at each layer.

    2. Cross-prediction (Concept → Conversation): Load Exp 3 concept probes,
       evaluate on Exp 2b conversational activations. If a probe trained on
       "what the model thinks human/AI minds are" can classify "is the model
       talking to a human/AI?" from conversation data, that's strong evidence
       of shared representation.

    3. Cross-prediction (Conversation → Concept): Load Exp 2b control/reading
       probes, evaluate on Exp 3 concept activations. The reverse direction.

Inputs:
    Exp 3 concept probes:     data/concept_probes/concept_probe_layer_*.pth
    Exp 3 concept activations: data/concept_activations/concept_activations.npz
    Exp 3 concept vectors:     data/concept_activations/concept_vector_per_layer.npz
    Exp 2b probes:            {EXP2B_ROOT}/probe_checkpoints/{control,reading}_probe/
    Exp 2b conversations:     {EXP2B_ROOT}/data/human_ai_conversations/*.txt

Outputs:
    data/cross_prediction/
        cross_prediction_results.json
        cross_prediction_plot.png
        cosine_alignment.json
        combined_analysis_plot.png

Requires GPU (for extracting Exp 2b conversational activations).
Env: llama2_env

Rachel C. Metzgar · Feb 2026
"""

import os
import json
import pickle
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Local imports ---
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.probes import LinearProbeClassification
from src.dataset import TextDataset, llama_v2_prompt
from config import config, set_version, add_version_argument, get_version_results_dir


# ========================== CONFIG ========================== #

MODEL_NAME = config.MODEL_NAME

# Exp 3 paths (version-independent)
CONCEPT_PROBE_DIR = str(config.PATHS.concept_probes)
CONCEPT_ACT_PATH = str(config.PATHS.concept_activations / "concept_activations.npz")
CONCEPT_VEC_PATH = str(config.PATHS.concept_activations / "concept_vector_per_layer.npz")

INPUT_DIM = config.INPUT_DIM
DEVICE = config.get_device()

torch.manual_seed(config.ANALYSIS.seed)

# Version-dependent paths — set by _init_paths() after set_version()
EXP2_CONV_DIR = None
EXP2_PROBE_DIRS = None
OUTPUT_DIR = None


def _init_paths():
    """Initialize version-dependent paths after set_version() has been called."""
    global EXP2_CONV_DIR, EXP2_PROBE_DIRS, OUTPUT_DIR

    EXP2_CONV_DIR = str(config.PATHS.exp2_conversations)
    EXP2_PROBE_DIRS = {
        "operational": str(config.PATHS.exp2_operational),
        "metacognitive": str(config.PATHS.exp2_metacognitive),
    }
    OUTPUT_DIR = str(get_version_results_dir(config.RESULTS.cross_prediction))


# ========================== LOADING UTILITIES ========================== #

def load_concept_activations(npz_path):
    """Load pre-extracted Exp 3 concept activations."""
    data = np.load(npz_path, allow_pickle=True)
    activations = torch.from_numpy(data["activations"]).float()  # (N, n_layers, dim)
    labels = torch.from_numpy(data["labels"]).long()              # (N,)
    print(f"Loaded concept activations: {activations.shape}")
    print(f"  Human: {(labels == 1).sum().item()}, AI: {(labels == 0).sum().item()}")
    return activations, labels


def load_probes_from_dir(probe_dir, filename_pattern, n_layers=41):
    """
    Load probe state dicts from a directory.
    Returns dict: {layer_idx: probe_object}
    """
    probes = {}
    if not os.path.isdir(probe_dir):
        print(f"[WARN] Probe directory not found: {probe_dir}")
        return probes

    for fname in sorted(os.listdir(probe_dir)):
        if not fname.endswith(".pth") or "_final.pth" in fname:
            continue
        if filename_pattern not in fname:
            continue

        # Extract layer number
        # Handles both "concept_probe_layer_38.pth" and "human_ai_probe_at_layer_38.pth"
        parts = fname.replace(".pth", "").split("_")
        layer_idx = int(parts[-1])

        probe = LinearProbeClassification(
            device="cpu", probe_class=1, input_dim=INPUT_DIM, logistic=True,
        )
        state = torch.load(os.path.join(probe_dir, fname), map_location="cpu")
        probe.load_state_dict(state)
        probe.eval()
        probes[layer_idx] = probe

    print(f"Loaded {len(probes)} probes from {probe_dir}")
    return probes


def extract_conv_activations(model, tokenizer, conv_dir, control_probe=True):
    """
    Extract conversational activations from Exp 2b using TextDataset.
    Returns activations tensor (N, n_layers, dim) and labels tensor (N,).
    """
    print(f"\nExtracting conversational activations (control_probe={control_probe})...")
    print(f"  Directory: {conv_dir}")

    dataset = TextDataset(
        directory=conv_dir,
        tokenizer=tokenizer,
        model=model,
        label_idf="_partner_",
        label_to_id={"human": 1, "ai": 0},
        convert_to_llama2_format=True,
        control_probe=control_probe,
        residual_stream=True,  # use hidden_states (residual stream)
        new_format=True,
        include_inst=True,
        if_remove_last_ai_response=True,
    )

    # Stack into tensors
    all_acts = []
    all_labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        all_acts.append(item["hidden_states"])
        all_labels.append(item["age"])  # "age" key is used for label

    activations = torch.stack(all_acts).float()  # (N, n_layers*dim) or (N, n_layers, dim)
    labels = torch.tensor(all_labels).long()

    # Reshape if needed: TextDataset stores as (n_layers * dim,) flat vector
    n_samples = activations.shape[0]
    n_layers = 41  # LLaMA-2-13B: embedding + 40 transformer layers
    if activations.ndim == 2 and activations.shape[1] == n_layers * INPUT_DIM:
        activations = activations.view(n_samples, n_layers, INPUT_DIM)
    elif activations.ndim == 2 and activations.shape[1] != n_layers * INPUT_DIM:
        # Try to infer n_layers
        inferred_layers = activations.shape[1] // INPUT_DIM
        print(f"  Inferred {inferred_layers} layers from shape {activations.shape}")
        activations = activations.view(n_samples, inferred_layers, INPUT_DIM)

    print(f"  Extracted: {activations.shape}")
    print(f"  Human: {(labels == 1).sum().item()}, AI: {(labels == 0).sum().item()}")
    return activations, labels


# ========================== EVALUATION ========================== #

def evaluate_probe_on_activations(probe, activations_at_layer, labels):
    """
    Evaluate a single probe on a set of activations.
    Returns accuracy.
    """
    probe.eval()
    with torch.no_grad():
        # Get probe weight and bias
        w = probe.proj[0].weight  # (1, dim)
        b = probe.proj[0].bias    # (1,)

        # Compute predictions
        logits = activations_at_layer @ w.T + b  # (N, 1)
        preds = (torch.sigmoid(logits) > 0.5).squeeze().long()

        acc = (preds == labels).float().mean().item()
    return acc


def run_cross_prediction(
    source_probes, target_activations, target_labels,
    source_name, target_name,
):
    """
    Evaluate probes from one domain on activations from another.
    Returns dict of {layer: accuracy}.
    """
    results = {}
    n_layers = target_activations.shape[1]

    print(f"\n  {source_name} probes → {target_name} activations:")
    for layer_idx in sorted(source_probes.keys()):
        if layer_idx >= n_layers:
            continue
        acts = target_activations[:, layer_idx, :]
        acc = evaluate_probe_on_activations(source_probes[layer_idx], acts, target_labels)
        results[layer_idx] = acc

    # Print summary
    if results:
        layers = sorted(results.keys())
        accs = [results[l] for l in layers]
        for l in layers:
            if l % 5 == 0 or l == layers[-1]:
                print(f"    Layer {l:2d}: {results[l]:.3f}")
        print(f"    Mean: {np.mean(accs):.3f}, Max: {max(accs):.3f} (layer {layers[np.argmax(accs)]})")

    return results


# ========================== COSINE ALIGNMENT ========================== #

def compute_cosine_alignment(concept_probes, exp2b_probes, concept_vec_path, probe_type_label):
    """
    Cosine similarity between concept and conversational probe weight vectors.
    Three comparisons per layer:
        1. Concept probe ↔ Exp 2b probe
        2. Concept mean-diff ↔ Exp 2b probe
        3. Concept probe ↔ Concept mean-diff
    """
    vec_data = np.load(concept_vec_path)
    concept_direction = torch.from_numpy(vec_data["concept_direction"]).float()

    results = {"layers": [], "concept_probe_to_2b": [], "concept_mean_to_2b": [], "concept_probe_to_mean": []}

    common_layers = sorted(set(concept_probes.keys()) & set(exp2b_probes.keys()))

    print(f"\n  Cosine alignment: Concept ↔ 2b {probe_type_label}")
    print(f"  Common layers: {len(common_layers)}")

    for layer in common_layers:
        cp_w = concept_probes[layer].proj[0].weight.detach()
        e2b_w = exp2b_probes[layer].proj[0].weight.detach()

        cos_pp = torch.nn.functional.cosine_similarity(cp_w, e2b_w).item()

        if layer < concept_direction.shape[0]:
            mean_dir = concept_direction[layer].unsqueeze(0)
            cos_m2b = torch.nn.functional.cosine_similarity(mean_dir, e2b_w).item()
            cos_pm = torch.nn.functional.cosine_similarity(cp_w, mean_dir).item()
        else:
            cos_m2b = None
            cos_pm = None

        results["layers"].append(layer)
        results["concept_probe_to_2b"].append(cos_pp)
        results["concept_mean_to_2b"].append(cos_m2b)
        results["concept_probe_to_mean"].append(cos_pm)

    # Print
    print(f"  {'Layer':<8} {'Probe↔2b':<14} {'Mean↔2b':<14} {'Probe↔Mean':<14}")
    print(f"  {'-'*48}")
    for i, layer in enumerate(results["layers"]):
        pp = results["concept_probe_to_2b"][i]
        m2b = results["concept_mean_to_2b"][i]
        pm = results["concept_probe_to_mean"][i]
        m2b_s = f"{m2b:.4f}" if m2b is not None else "N/A"
        pm_s = f"{pm:.4f}" if pm is not None else "N/A"
        print(f"  {layer:<8} {pp:.4f}        {m2b_s:<14} {pm_s:<14}")

    if results["concept_probe_to_2b"]:
        vals = results["concept_probe_to_2b"]
        best = max(vals, key=abs)
        print(f"\n  Max |cos| (probe↔2b): {best:.4f}")

    return results


# ========================== PLOTTING ========================== #

def make_cross_prediction_plot(all_cross_results, output_path):
    """Plot cross-prediction accuracy curves."""
    try:
        n_panels = len(all_cross_results)
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
        axes = axes[0]

        for ax, (label, data) in zip(axes, all_cross_results.items()):
            for direction_label, results in data.items():
                if not results:
                    continue
                layers = sorted(results.keys())
                accs = [results[l] for l in layers]
                ax.plot(layers, accs, 'o-', label=direction_label, markersize=3)

            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.7, label="Chance")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Cross-Prediction: {label}")
            ax.legend(fontsize=8)
            ax.set_ylim(0.3, 1.0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved cross-prediction plot to {output_path}")
    except Exception as e:
        print(f"[WARN] Plot failed: {e}")


def make_combined_plot(cross_results, cosine_results, output_path):
    """Combined plot: cross-prediction + cosine alignment."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top row: cross-prediction
        for ax, (probe_type, data) in zip(axes[0], cross_results.items()):
            for direction, results in data.items():
                if not results:
                    continue
                layers = sorted(results.keys())
                accs = [results[l] for l in layers]
                ax.plot(layers, accs, 'o-', label=direction, markersize=3)
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.7)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Cross-Prediction ({probe_type})")
            ax.legend(fontsize=7)
            ax.set_ylim(0.3, 1.0)

        # Bottom row: cosine alignment
        for ax, (probe_type, results) in zip(axes[1], cosine_results.items()):
            layers = results["layers"]
            ax.plot(layers, results["concept_probe_to_2b"],
                    'o-', label="Concept probe ↔ 2b", color="tab:blue", markersize=3)
            valid_m = [(l, v) for l, v in zip(layers, results["concept_mean_to_2b"]) if v is not None]
            if valid_m:
                ax.plot([x[0] for x in valid_m], [x[1] for x in valid_m],
                        's--', label="Mean-diff ↔ 2b", color="tab:orange", markersize=3)
            ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(f"Cosine Alignment ({probe_type})")
            ax.legend(fontsize=7)
            ax.set_ylim(-0.3, 0.3)

        plt.suptitle(
            "Exp 3: Concept-of-Mind ↔ Partner-Identity Representational Overlap",
            fontsize=13, y=1.02,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved combined plot to {output_path}")
    except Exception as e:
        print(f"[WARN] Combined plot failed: {e}")


# ========================== MAIN ========================== #

def _has_conversations(conv_dir):
    """Check if the Exp 2 conversations directory exists and is non-empty."""
    if not os.path.isdir(conv_dir):
        return False
    # Check for at least one .txt file
    for f in os.listdir(conv_dir):
        if f.endswith(".txt"):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Exp 3 Phase 5: Cross-prediction & representational alignment"
    )
    add_version_argument(parser)
    args = parser.parse_args()

    # Set version and initialize paths
    set_version(args.version)
    _init_paths()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Version: {args.version}")
    print(f"Output: {OUTPUT_DIR}")

    has_convs = _has_conversations(EXP2_CONV_DIR)
    if not has_convs:
        print(f"\n⚠️  WARNING: No conversations found for version '{args.version}'")
        print(f"   Directory: {EXP2_CONV_DIR}")
        print(f"   Skipping 'Concept → Conversation' cross-prediction.")
        print(f"   Will run: 'Conversation → Concept' + cosine alignment only.\n")

    # === Load Exp 3 concept activations (pre-extracted) ===
    concept_acts, concept_labels = load_concept_activations(CONCEPT_ACT_PATH)
    n_concept_layers = concept_acts.shape[1]

    # === Load probes ===
    concept_probes = load_probes_from_dir(
        CONCEPT_PROBE_DIR, "concept_probe_layer",
    )
    exp2b_probes = {}
    for probe_type, probe_dir in EXP2_PROBE_DIRS.items():
        exp2b_probes[probe_type] = load_probes_from_dir(
            probe_dir, "human_ai_probe_at_layer",
        )

    # === Extract Exp 2b conversational activations (only if conversations exist) ===
    conv_acts_control = conv_labels_control = None
    conv_acts_reading = conv_labels_reading = None

    if has_convs:
        print("Loading LLaMA-2-Chat-13B...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
        model.half().to(DEVICE).eval()
        print("Model loaded.")

        # Control probe position (last user-message token)
        conv_acts_control, conv_labels_control = extract_conv_activations(
            model, tokenizer, EXP2_CONV_DIR, control_probe=True,
        )
        # Reading probe position ("I think the partner..." token)
        conv_acts_reading, conv_labels_reading = extract_conv_activations(
            model, tokenizer, EXP2_CONV_DIR, control_probe=False,
        )

        # Save extracted activations for future use
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, "exp2b_conv_activations_control.npz"),
            activations=conv_acts_control.numpy(),
            labels=conv_labels_control.numpy(),
        )
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, "exp2b_conv_activations_reading.npz"),
            activations=conv_acts_reading.numpy(),
            labels=conv_labels_reading.numpy(),
        )
        print("Saved extracted conversational activations for future use.")

    # === Cross-prediction ===
    all_cross_results = {}

    for probe_type in ["operational", "metacognitive"]:
        print(f"\n{'#'*70}")
        print(f"# Cross-prediction: {probe_type}")
        print(f"{'#'*70}")

        direction_results = {}

        # Direction 1: Concept probes → Conversational activations (requires conversations)
        if has_convs:
            conv_acts = conv_acts_control if probe_type == "operational" else conv_acts_reading
            conv_labels = conv_labels_control if probe_type == "operational" else conv_labels_reading

            d1 = run_cross_prediction(
                concept_probes, conv_acts, conv_labels,
                source_name="Exp3 concept", target_name=f"Exp2b {probe_type}",
            )
            direction_results["Concept → Conversation"] = d1
        else:
            print(f"  [SKIP] Concept → Conversation (no conversations for version '{args.version}')")
            direction_results["Concept → Conversation"] = {}

        # Direction 2: Exp 2b probes → Concept activations (always available)
        d2 = run_cross_prediction(
            exp2b_probes[probe_type], concept_acts, concept_labels,
            source_name=f"Exp2b {probe_type}", target_name="Exp3 concept",
        )
        direction_results["Conversation → Concept"] = d2

        all_cross_results[probe_type] = direction_results

    # === Cosine alignment (always available — only needs probe weights) ===
    all_cosine_results = {}
    for probe_type in ["operational", "metacognitive"]:
        print(f"\n{'#'*70}")
        print(f"# Cosine alignment: {probe_type}")
        print(f"{'#'*70}")
        cosine = compute_cosine_alignment(
            concept_probes, exp2b_probes[probe_type],
            CONCEPT_VEC_PATH, probe_type,
        )
        all_cosine_results[probe_type] = cosine

    # === Save all results ===
    # Convert cross-prediction results to serializable format
    serializable_cross = {}
    for probe_type, directions in all_cross_results.items():
        serializable_cross[probe_type] = {}
        for direction, layer_accs in directions.items():
            serializable_cross[probe_type][direction] = {
                str(k): v for k, v in layer_accs.items()
            }

    with open(os.path.join(OUTPUT_DIR, "cross_prediction_results.json"), "w") as f:
        json.dump(serializable_cross, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "cosine_alignment.json"), "w") as f:
        json.dump(all_cosine_results, f, indent=2)

    # === Plots ===
    make_cross_prediction_plot(
        all_cross_results,
        os.path.join(OUTPUT_DIR, "cross_prediction_plot.png"),
    )
    if has_convs:
        make_combined_plot(
            all_cross_results, all_cosine_results,
            os.path.join(OUTPUT_DIR, "combined_analysis_plot.png"),
        )

    # === Summary ===
    print(f"\n{'='*70}")
    print(f"SUMMARY (version: {args.version})")
    print(f"{'='*70}")

    for probe_type in ["operational", "metacognitive"]:
        print(f"\n  {probe_type}:")
        for direction, results in all_cross_results[probe_type].items():
            if results:
                layers = sorted(results.keys())
                accs = [results[l] for l in layers]
                best_layer = layers[np.argmax(accs)]
                print(f"    {direction}: max acc = {max(accs):.3f} (layer {best_layer}), "
                      f"mean = {np.mean(accs):.3f}")
            else:
                print(f"    {direction}: SKIPPED (no conversations)")

        cosine = all_cosine_results[probe_type]
        if cosine["concept_probe_to_2b"]:
            best_cos = max(cosine["concept_probe_to_2b"], key=abs)
            print(f"    Cosine (probe↔2b): max |cos| = {best_cos:.4f}")

    print(f"\n✅ Cross-prediction analysis complete. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()