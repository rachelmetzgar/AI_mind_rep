#!/usr/bin/env python3
"""
Experiment 2c, Phase 2-3: Train Concept Probes and Representational Alignment

Trains linear probes on the concept elicitation activations (from Script 1)
to obtain a refined concept steering vector, then computes representational
alignment between:
    (a) The concept vector (general human/AI semantics)
    (b) The conversational steering vectors from Experiment 2b:
        - Control probes (last user-message token)
        - Reading probes ("I think the partner…" token)

This alignment analysis is the key test of Experiment 2c: do the model's
general concepts of "human" and "AI" share representational structure with
the conversational partner-identity signal?

Inputs:
    data/concept_activations/concept_activations.npz
    data/concept_activations/concept_vector_per_layer.npz
    Experiment 2b probes:
        ../exp_2b-13B-chat/data/probe_checkpoints/control_probe/
        ../exp_2b-13B-chat/data/probe_checkpoints/reading_probe/

Outputs:
    data/concept_probes/
        concept_probe_layer_{N}.pth
        accuracy_summary.pkl
        cm_layer_{N}.png
    data/alignment/
        control_probe/
            alignment_results.json
            alignment_plot.png
        reading_probe/
            alignment_results.json
            alignment_plot.png
        combined_alignment_plot.png

Env: llama2_env

Rachel C. Metzgar · Feb 2026
"""

import os
import json
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Local imports ---
from src.probes import LinearProbeClassification, TrainerConfig
from src.train_test_utils import train, test
from src.losses import edl_mse_loss


# ========================== CONFIG ========================== #

CONCEPT_ACT_PATH = "data/concept_activations/concept_activations.npz"
CONCEPT_VEC_PATH = "data/concept_activations/concept_vector_per_layer.npz"

CONCEPT_PROBE_DIR = "data/concept_probes"
ALIGNMENT_DIR = "data/alignment"

# Experiment 2b probes — both types
EXP2B_ROOT = (
    "/jukebox/graziano/rachel/ai_mind_rep/exp_2b-13B-chat/"
    "data/probe_checkpoints"
)
EXP2B_PROBE_DIRS = {
    "control_probe": os.path.join(EXP2B_ROOT, "control_probe"),
    "reading_probe": os.path.join(EXP2B_ROOT, "reading_probe"),
}

INPUT_DIM = 5120  # LLaMA-2-13B hidden size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 50
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 16
LOGISTIC = True
torch.manual_seed(12345)


# ========================== DATASET ========================== #

class ConceptActivationDataset(Dataset):
    """Simple dataset wrapping pre-extracted concept activations."""

    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.activations = torch.from_numpy(data["activations"]).float()
        self.labels = torch.from_numpy(data["labels"]).long()
        self.n_samples = len(self.labels)
        self.n_layers = self.activations.shape[1]
        self.hidden_dim = self.activations.shape[2]
        print(f"Loaded {self.n_samples} concept activations, "
              f"{self.n_layers} layers, dim={self.hidden_dim}")
        print(f"  Human: {(self.labels == 1).sum().item()}, "
              f"AI: {(self.labels == 0).sum().item()}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "hidden_states": self.activations[idx],
            "age": self.labels[idx],
        }


# ========================== PROBE TRAINING ========================== #

def train_concept_probes(dataset):
    """Train a linear probe per layer on concept activations."""
    os.makedirs(CONCEPT_PROBE_DIR, exist_ok=True)

    n_layers = dataset.n_layers
    loss_func = nn.BCELoss()

    idx_train, idx_test = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        random_state=12345,
        stratify=dataset.labels.numpy(),
    )
    train_ds = Subset(dataset, idx_train)
    test_ds = Subset(dataset, idx_test)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE_TRAIN, drop_last=False)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE_TEST, drop_last=False)

    acc_summary = {"acc": [], "final": [], "train": []}
    probe_weights = {}

    print(f"\n=== Training concept probes ({n_layers} layers) ===")
    print(f"  Train: {len(idx_train)}, Test: {len(idx_test)}")

    for layer_num in range(n_layers):
        trainer_cfg = TrainerConfig()
        probe = LinearProbeClassification(
            device=DEVICE,
            probe_class=1,
            input_dim=INPUT_DIM,
            logistic=LOGISTIC,
        )
        optimizer, scheduler = probe.configure_optimizers(trainer_cfg)
        best_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            train_results = train(
                probe, DEVICE, train_loader, optimizer, epoch,
                loss_func=loss_func, verbose=False, layer_num=layer_num,
                return_raw_outputs=True, one_hot=False,
                uncertainty=False, num_classes=2,
            )
            tr_loss, tr_acc = train_results[0], train_results[1]

            test_results = test(
                probe, DEVICE, test_loader,
                loss_func=loss_func, verbose=False, layer_num=layer_num,
                scheduler=scheduler, return_raw_outputs=True,
                one_hot=False, uncertainty=False, num_classes=2,
            )
            te_loss, te_acc = test_results[0], test_results[1]
            te_preds, te_truths = test_results[2], test_results[3]

            if te_acc > best_acc:
                best_acc = te_acc
                torch.save(
                    probe.state_dict(),
                    os.path.join(CONCEPT_PROBE_DIR, f"concept_probe_layer_{layer_num}.pth"),
                )

        torch.save(
            probe.state_dict(),
            os.path.join(CONCEPT_PROBE_DIR, f"concept_probe_layer_{layer_num}_final.pth"),
        )

        probe_weights[layer_num] = probe.proj[0].weight.detach().cpu().clone()

        cm = confusion_matrix(te_truths, te_preds)
        ConfusionMatrixDisplay(cm, display_labels=["AI", "Human"]).plot()
        plt.title(f"Concept Probe Layer {layer_num} Acc {best_acc:.3f}")
        plt.savefig(
            os.path.join(CONCEPT_PROBE_DIR, f"cm_layer_{layer_num}.png"), dpi=200
        )
        plt.close()

        acc_summary["acc"].append(best_acc)
        acc_summary["final"].append(te_acc)
        acc_summary["train"].append(tr_acc)
        torch.cuda.empty_cache()

        if layer_num % 5 == 0 or layer_num == n_layers - 1:
            print(f"  Layer {layer_num:2d}: best_acc = {best_acc:.3f}")

    with open(os.path.join(CONCEPT_PROBE_DIR, "accuracy_summary.pkl"), "wb") as f:
        pickle.dump(acc_summary, f)

    print(f"\n✅ Concept probe training complete.")
    print(f"  Mean best accuracy: {np.mean(acc_summary['acc']):.3f}")
    print(f"  Max accuracy: {np.max(acc_summary['acc']):.3f} "
          f"(layer {np.argmax(acc_summary['acc'])})")

    return probe_weights, acc_summary


# ========================== ALIGNMENT ANALYSIS ========================== #

def load_exp2b_probe_weights(probe_dir, probe_type_label="2b"):
    """Load Experiment 2b probe weight vectors from a directory."""
    weights = {}
    if not os.path.isdir(probe_dir):
        print(f"[WARN] Exp 2b {probe_type_label} dir not found: {probe_dir}")
        return weights

    for fname in sorted(os.listdir(probe_dir)):
        if not fname.startswith("human_ai_probe_at_layer_") or not fname.endswith(".pth"):
            continue
        if fname.endswith("_final.pth"):
            continue
        layer_str = fname.split("_layer_")[-1].split(".pth")[0]
        layer_idx = int(layer_str)

        probe = LinearProbeClassification(
            device="cpu", probe_class=1, input_dim=INPUT_DIM, logistic=LOGISTIC,
        )
        state = torch.load(os.path.join(probe_dir, fname), map_location="cpu")
        probe.load_state_dict(state)
        weights[layer_idx] = probe.proj[0].weight.detach().clone()

    print(f"Loaded {len(weights)} Exp 2b {probe_type_label} weights from {probe_dir}")
    return weights


def compute_alignment(
    concept_probe_weights, concept_mean_direction, exp2b_weights,
    probe_type_label, out_subdir,
):
    """
    Compute cosine similarity between concept vectors and one set of
    Exp 2b conversational vectors at each layer.

    Three comparisons:
        1. Concept probe weight vs Exp 2b probe weight (probe-to-probe)
        2. Mean-difference concept vector vs Exp 2b probe weight
        3. Concept probe weight vs mean-difference concept vector (internal)
    """
    out_dir = os.path.join(ALIGNMENT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    results = {"layers": [], "probe_to_2b": [], "mean_to_2b": [], "probe_to_mean": []}

    common_layers = sorted(set(concept_probe_weights.keys()) & set(exp2b_weights.keys()))
    if not common_layers:
        print(f"[WARN] No overlapping layers for {probe_type_label}.")
        for layer in sorted(concept_probe_weights.keys()):
            if layer >= concept_mean_direction.shape[0]:
                continue
            cos_pm = torch.nn.functional.cosine_similarity(
                concept_probe_weights[layer],
                concept_mean_direction[layer].unsqueeze(0),
            ).item()
            results["layers"].append(layer)
            results["probe_to_2b"].append(None)
            results["mean_to_2b"].append(None)
            results["probe_to_mean"].append(cos_pm)
    else:
        for layer in common_layers:
            cp_w = concept_probe_weights[layer]
            e2b_w = exp2b_weights[layer]
            mean_dir = concept_mean_direction[layer].unsqueeze(0)

            cos_pp = torch.nn.functional.cosine_similarity(cp_w, e2b_w).item()
            cos_m2b = torch.nn.functional.cosine_similarity(mean_dir, e2b_w).item()
            cos_pm = torch.nn.functional.cosine_similarity(cp_w, mean_dir).item()

            results["layers"].append(layer)
            results["probe_to_2b"].append(cos_pp)
            results["mean_to_2b"].append(cos_m2b)
            results["probe_to_mean"].append(cos_pm)

    # Save
    with open(os.path.join(out_dir, "alignment_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print
    print(f"\n{'='*70}")
    print(f"ALIGNMENT: Concept Vector ↔ Exp 2b {probe_type_label}")
    print(f"{'='*70}")
    print(f"\n{'Layer':<8} {'Probe↔2b':<14} {'Mean↔2b':<14} {'Probe↔Mean':<14}")
    print("-" * 50)
    for i, layer in enumerate(results["layers"]):
        pp = results["probe_to_2b"][i]
        m2b = results["mean_to_2b"][i]
        pm = results["probe_to_mean"][i]
        pp_str = f"{pp:.4f}" if pp is not None else "N/A"
        m2b_str = f"{m2b:.4f}" if m2b is not None else "N/A"
        pm_str = f"{pm:.4f}" if pm is not None else "N/A"
        print(f"  {layer:<6} {pp_str:<14} {m2b_str:<14} {pm_str:<14}")

    if any(v is not None for v in results["probe_to_2b"]):
        valid = [v for v in results["probe_to_2b"] if v is not None]
        best_idx = int(np.argmax(np.abs(valid)))
        print(f"\nProbe↔2b: mean={np.mean(valid):.4f}, "
              f"max|cos|={valid[best_idx]:.4f} (layer {results['layers'][best_idx]})")

    # Per-probe-type plot
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if any(v is not None for v in results["probe_to_2b"]):
            ax = axes[0]
            valid_idx = [i for i, v in enumerate(results["probe_to_2b"]) if v is not None]
            layers_v = [results["layers"][i] for i in valid_idx]
            ax.plot(layers_v, [results["probe_to_2b"][i] for i in valid_idx],
                    'o-', label=f"Concept probe ↔ 2b {probe_type_label}", color="tab:blue")
            ax.plot(layers_v, [results["mean_to_2b"][i] for i in valid_idx],
                    's--', label=f"Mean-diff ↔ 2b {probe_type_label}", color="tab:orange")
            ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(f"Alignment: Concept ↔ 2b {probe_type_label}")
            ax.legend()
            ax.set_ylim(-1, 1)

        ax2 = axes[1]
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Cosine Similarity")
        ax2.set_title("Internal: Concept Probe ↔ Mean-Diff")
        valid_pm = [(results["layers"][i], results["probe_to_mean"][i])
                    for i in range(len(results["layers"]))
                    if results["probe_to_mean"][i] is not None]
        if valid_pm:
            ax2.plot([x[0] for x in valid_pm], [x[1] for x in valid_pm],
                     'o-', color="tab:green")
            ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
            ax2.set_ylim(-1, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "alignment_plot.png"), dpi=200)
        plt.close()
        print(f"Saved plot to {out_dir}/alignment_plot.png")
    except Exception as e:
        print(f"[WARN] Plot failed: {e}")

    return results


def make_combined_plot(all_results):
    """
    Combined plot showing alignment with both control and reading probes
    side by side for easy comparison.
    """
    os.makedirs(ALIGNMENT_DIR, exist_ok=True)

    try:
        n_types = len(all_results)
        fig, axes = plt.subplots(1, n_types, figsize=(7 * n_types, 5))
        if n_types == 1:
            axes = [axes]

        for ax, (probe_type, results) in zip(axes, all_results.items()):
            valid_idx = [i for i, v in enumerate(results["probe_to_2b"]) if v is not None]
            if not valid_idx:
                ax.set_title(f"{probe_type}: no data")
                continue
            layers_v = [results["layers"][i] for i in valid_idx]
            ax.plot(layers_v, [results["probe_to_2b"][i] for i in valid_idx],
                    'o-', label="Concept probe ↔ 2b", color="tab:blue", markersize=3)
            ax.plot(layers_v, [results["mean_to_2b"][i] for i in valid_idx],
                    's--', label="Mean-diff ↔ 2b", color="tab:orange", markersize=3)
            ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(f"Concept ↔ 2b {probe_type}")
            ax.legend(fontsize=8)
            ax.set_ylim(-0.15, 0.15)

        plt.suptitle(
            "Representational Alignment: General Concept ↔ Conversational Partner-Identity",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(ALIGNMENT_DIR, "combined_alignment_plot.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nSaved combined plot to {ALIGNMENT_DIR}/combined_alignment_plot.png")
    except Exception as e:
        print(f"[WARN] Combined plot failed: {e}")


# ========================== MAIN ========================== #

def main():
    # Load concept activations
    dataset = ConceptActivationDataset(CONCEPT_ACT_PATH)

    # Load mean-difference concept vector
    vec_data = np.load(CONCEPT_VEC_PATH)
    concept_mean_direction = torch.from_numpy(vec_data["concept_direction"]).float()
    print(f"Loaded concept mean-difference vector: shape {concept_mean_direction.shape}")

    # Train concept probes
    concept_probe_weights, acc_summary = train_concept_probes(dataset)

    # Run alignment against BOTH 2b probe types
    all_alignment_results = {}

    for probe_type, probe_dir in EXP2B_PROBE_DIRS.items():
        print(f"\n{'#'*70}")
        print(f"# Alignment analysis: Concept ↔ 2b {probe_type}")
        print(f"{'#'*70}")

        exp2b_weights = load_exp2b_probe_weights(probe_dir, probe_type)

        results = compute_alignment(
            concept_probe_weights,
            concept_mean_direction,
            exp2b_weights,
            probe_type_label=probe_type,
            out_subdir=probe_type,
        )
        all_alignment_results[probe_type] = results

    # Combined comparison plot
    if len(all_alignment_results) > 1:
        make_combined_plot(all_alignment_results)

    # Print summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: Max |cosine similarity| with Exp 2b probes")
    print(f"{'='*70}")
    for probe_type, results in all_alignment_results.items():
        valid_pp = [v for v in results["probe_to_2b"] if v is not None]
        valid_m = [v for v in results["mean_to_2b"] if v is not None]
        if valid_pp:
            best_pp = max(valid_pp, key=abs)
            best_m = max(valid_m, key=abs)
            print(f"  {probe_type}:")
            print(f"    Concept probe ↔ 2b: max |cos| = {best_pp:.4f}")
            print(f"    Mean-diff     ↔ 2b: max |cos| = {best_m:.4f}")

    print(f"\n✅ Phase 2-3 complete.")


if __name__ == "__main__":
    main()