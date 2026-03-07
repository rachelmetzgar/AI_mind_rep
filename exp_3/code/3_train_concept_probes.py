#!/usr/bin/env python3
"""
Experiment 3, Phase 3: Train Concept Probes

For a SINGLE concept dimension (specified by --dim_id):
    1. Trains linear probes on contrast concept activations (from Script 1)
    2. Optionally computes quick alignment with Exp 2b conversational probes
       (the authoritative alignment analysis is in 2a_alignment_analysis.py)

Designed to run in parallel via SLURM array jobs (one job per dimension).

Inputs:
    results/{model}/concept_activations/contrasts/{dim_name}/concept_activations.npz
    results/{model}/concept_activations/contrasts/{dim_name}/concept_vector_per_layer.npz
    Exp 2b probes (control + reading)

Outputs:
    results/{model}/concept_probes/{dim_name}/
        concept_probe_layer_{N}.pth
        accuracy_summary.pkl

Usage:
    python 3_train_concept_probes.py --dim_id 1
    python 3_train_concept_probes.py --dim_id 7

SLURM:
    sbatch slurm/train_concept_probes.sh

Env: llama2_env
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import pickle
import argparse
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
from utils.probes import LinearProbeClassification, TrainerConfig
from utils.train_test_utils import train, test
from config import config, set_version, add_version_argument, add_turn_argument


# ========================== CONFIG ========================== #

# Version-independent paths
CONTRAST_ACT_DIR = str(config.RESULTS.concept_activations_contrasts)
OUTPUT_ROOT_PROBES = str(config.RESULTS.concept_probes_data)  # results/{model}/concept_probes/
OUTPUT_ROOT_ALIGN = str(config.RESULTS.alignment_versions)  # results/{model}/{version}/alignment/ (final outputs)

INPUT_DIM = config.INPUT_DIM
DEVICE = config.get_device()
EPOCHS = config.TRAINING.epochs
BATCH_SIZE_TRAIN = config.TRAINING.batch_size_train
BATCH_SIZE_TEST = config.TRAINING.batch_size_test
LOGISTIC = config.TRAINING.logistic
torch.manual_seed(config.ANALYSIS.seed)

# Version-dependent paths — set by _init_paths() after set_version()
EXP2_PROBE_DIRS = None


def _init_paths():
    """Initialize version-dependent paths after set_version() has been called."""
    global EXP2_PROBE_DIRS
    EXP2_PROBE_DIRS = {
        "operational": str(config.PATHS.exp2_operational),
        "metacognitive": str(config.PATHS.exp2_metacognitive),
    }


# ========================== DIMENSION DISCOVERY ========================== #

def discover_contrast_dimensions():
    """
    Scan data/concept_activations/contrasts/ for available dimensions.
    Returns dict: {dim_id: dim_name}.
    """
    dims = {}
    if not os.path.isdir(CONTRAST_ACT_DIR):
        return dims
    for name in sorted(os.listdir(CONTRAST_ACT_DIR)):
        full = os.path.join(CONTRAST_ACT_DIR, name)
        if not os.path.isdir(full):
            continue
        parts = name.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            dim_id = int(parts[0])
        except ValueError:
            continue
        dims[dim_id] = name
    return dims


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
            "age": self.labels[idx],  # kept for compatibility with train/test utils
        }


# ========================== PROBE TRAINING ========================== #

def train_concept_probes(dataset, probe_dir):
    """Train a linear probe per layer on concept activations."""
    os.makedirs(probe_dir, exist_ok=True)

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
    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=BATCH_SIZE_TRAIN, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, batch_size=BATCH_SIZE_TEST, drop_last=False
    )

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
                    os.path.join(probe_dir, f"concept_probe_layer_{layer_num}.pth"),
                )

        torch.save(
            probe.state_dict(),
            os.path.join(probe_dir, f"concept_probe_layer_{layer_num}_final.pth"),
        )

        probe_weights[layer_num] = probe.proj[0].weight.detach().cpu().clone()

        # Confusion matrix
        cm = confusion_matrix(te_truths, te_preds)
        ConfusionMatrixDisplay(cm, display_labels=["AI", "Human"]).plot()
        plt.title(f"Concept Probe Layer {layer_num} Acc {best_acc:.3f}")
        plt.savefig(
            os.path.join(probe_dir, f"cm_layer_{layer_num}.png"), dpi=200
        )
        plt.close()

        acc_summary["acc"].append(best_acc)
        acc_summary["final"].append(te_acc)
        acc_summary["train"].append(tr_acc)
        torch.cuda.empty_cache()

        if layer_num % 5 == 0 or layer_num == n_layers - 1:
            print(f"  Layer {layer_num:2d}: best_acc = {best_acc:.3f}")

    with open(os.path.join(probe_dir, "accuracy_summary.pkl"), "wb") as f:
        pickle.dump(acc_summary, f)

    print(f"\n✅ Probe training complete.")
    print(f"  Mean best accuracy: {np.mean(acc_summary['acc']):.3f}")
    print(f"  Max accuracy: {np.max(acc_summary['acc']):.3f} "
          f"(layer {np.argmax(acc_summary['acc'])})")

    return probe_weights, acc_summary


# ========================== QUICK ALIGNMENT (sanity check) ========================== #

def load_exp2b_probe_weights(probe_dir, probe_type_label="2b"):
    """Load Experiment 2b probe weight vectors."""
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

    print(f"Loaded {len(weights)} Exp 2b {probe_type_label} weights")
    return weights


def quick_alignment(
    concept_probe_weights, concept_mean_direction, exp2b_weights,
    probe_type_label, out_dir,
):
    """
    Quick cosine similarity check between concept vectors and Exp 2b vectors.
    For the full analysis with bootstrap CIs, see 2b_alignment_analysis.py.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = {"layers": [], "probe_to_2b": [], "mean_to_2b": [], "probe_to_mean": []}

    common_layers = sorted(
        set(concept_probe_weights.keys()) & set(exp2b_weights.keys())
    )

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

    with open(os.path.join(out_dir, "alignment_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n  Quick alignment: Concept ↔ 2b {probe_type_label}")
    valid_pp = [v for v in results["probe_to_2b"] if v is not None]
    valid_m = [v for v in results["mean_to_2b"] if v is not None]
    if valid_pp:
        print(f"    Probe↔2b: mean={np.mean(valid_pp):.4f}, "
              f"max|cos|={max(valid_pp, key=abs):.4f}")
        print(f"    Mean↔2b:  mean={np.mean(valid_m):.4f}, "
              f"max|cos|={max(valid_m, key=abs):.4f}")

    return results


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 3: Train concept probes for one dimension"
    )
    add_version_argument(parser)
    add_turn_argument(parser)
    parser.add_argument("--dim_id", type=int, required=True,
                        help="Dimension ID")
    parser.add_argument("--skip_alignment", action="store_true",
                        help="Skip quick alignment check (use 2b_alignment_analysis.py instead)")
    args = parser.parse_args()

    # Set version and initialize version-dependent paths
    set_version(args.version, turn=args.turn)
    _init_paths()
    print(f"Version: {args.version}")
    print(f"Turn: {args.turn}")

    # Discover available dimensions from extracted activations
    available_dims = discover_contrast_dimensions()
    if args.dim_id not in available_dims:
        raise ValueError(
            f"dim_id={args.dim_id} not found in {CONTRAST_ACT_DIR}. "
            f"Available: {list(available_dims.keys())}. "
            f"Run 1_elicit_concept_vectors.py --mode contrasts --dim_id {args.dim_id} first."
        )

    dim_name = available_dims[args.dim_id]

    # Paths
    act_dir = os.path.join(CONTRAST_ACT_DIR, dim_name)
    act_path = os.path.join(act_dir, "concept_activations.npz")
    vec_path = os.path.join(act_dir, "concept_vector_per_layer.npz")
    probe_dir = os.path.join(OUTPUT_ROOT_PROBES, dim_name)

    if not os.path.isfile(act_path):
        raise FileNotFoundError(
            f"Activations not found: {act_path}\n"
            f"Run 1_elicit_concept_vectors.py --mode contrasts --dim_id {args.dim_id} first."
        )

    # Load
    dataset = ConceptActivationDataset(act_path)
    vec_data = np.load(vec_path)
    concept_mean_direction = torch.from_numpy(vec_data["concept_direction"]).float()
    print(f"Loaded concept mean-diff vector: shape {concept_mean_direction.shape}")

    # Train probes
    concept_probe_weights, acc_summary = train_concept_probes(dataset, probe_dir)

    # Quick alignment sanity check (optional; full analysis in 2b_alignment_analysis.py)
    if not args.skip_alignment:
        align_dir = os.path.join(OUTPUT_ROOT_ALIGN, dim_name)
        for probe_type, probe_dir_2b in EXP2_PROBE_DIRS.items():
            exp2b_weights = load_exp2b_probe_weights(probe_dir_2b, probe_type)
            align_sub_dir = os.path.join(align_dir, probe_type)
            quick_alignment(
                concept_probe_weights, concept_mean_direction,
                exp2b_weights, probe_type, align_sub_dir,
            )

    # Final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {dim_name} (dim_id={args.dim_id})")
    print(f"{'='*60}")
    print(f"  Probe accuracy: mean={np.mean(acc_summary['acc']):.3f}, "
          f"max={np.max(acc_summary['acc']):.3f} "
          f"(layer {np.argmax(acc_summary['acc'])})")
    print(f"\n✅ Phase 2a complete for dimension {args.dim_id} ({dim_name}).")
    if not args.skip_alignment:
        print(f"  Note: For full alignment analysis with bootstrap CIs,")
        print(f"  run 2b_alignment_analysis.py after all dimensions are trained.")


if __name__ == "__main__":
    main()