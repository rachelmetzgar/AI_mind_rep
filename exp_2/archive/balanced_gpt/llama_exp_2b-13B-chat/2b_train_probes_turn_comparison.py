#!/usr/bin/env python3
"""
Train probes at different conversation turns to test whether degradation
in later turns hurts probe accuracy.

Compares:
  - turn_index=2  (through turn 3, before worst degradation)
  - turn_index=-1 (turn 5, full conversation — current default)

Saves to separate output directories so existing probes are untouched.

Usage:
    python 2b_train_probes_turn_comparison.py --turn_index 2
    python 2b_train_probes_turn_comparison.py --turn_index -1

Env: llama2_env (needs GPU)
Rachel C. Metzgar · Feb 2026
"""

import os, argparse, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.dataset import TextDatasetCSV
from src.train_test_utils import train, test
from src.probes import LinearProbeClassification, TrainerConfig
from src.losses import edl_mse_loss

# ========================== CONFIG ========================== #
CSV_DIR = (
    "/jukebox/graziano/rachel/ai_mind_rep/exp_1/balanced_gpt/"
    "data/meta-llama-Llama-2-13b-chat-hf/0.8"
)

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

INPUT_DIM = 5120
EPOCHS = 50
BATCH_SIZE_TRAIN = 200
BATCH_SIZE_TEST = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(12345)

LOGISTIC = True
UNCERTAINTY = False
ONE_HOT = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train probes at a specific turn index.")
    parser.add_argument("--turn_index", type=int, required=True,
                        help="Turn index (0-based). 0=turn1, 2=turn3, -1=turn5/last.")
    return parser.parse_args()


def run_probe_train(model, tokenizer, control_probe, out_subdir, turn_index, probe_dir):
    """Train probes across all layers."""
    os.makedirs(f"{probe_dir}/{out_subdir}", exist_ok=True)

    dataset = TextDatasetCSV(
        csv_dir=CSV_DIR,
        tokenizer=tokenizer,
        model=model,
        control_probe=control_probe,
        label_idf="_partner_",
        label_to_id={"ai": 0, "human": 1},
        residual_stream=True,
        turn_index=turn_index,
        max_length=2048,
        one_hot=ONE_HOT,
    )

    n_layers = dataset.acts[0].shape[0] if len(dataset.acts) > 0 else 41
    print(f"\nDataset size: {len(dataset)}, n_layers: {n_layers}")

    labels_for_split = [
        l if isinstance(l, int) else l.item() if isinstance(l, torch.Tensor) and l.dim() == 0 else torch.argmax(l).item()
        for l in dataset.labels
    ]
    idx_train, idx_test = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        random_state=12345,
        stratify=labels_for_split,
    )
    train_ds, test_ds = Subset(dataset, idx_train), Subset(dataset, idx_test)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE_TRAIN,
                              num_workers=1, drop_last=True)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE_TEST,
                             num_workers=1, drop_last=False)

    loss_func = edl_mse_loss if UNCERTAINTY else nn.BCELoss()
    acc_summary = {"acc": [], "final": [], "train": []}
    probe_label = "control" if control_probe else "reading"
    print(f"\n=== Training {probe_label} probe (turn_index={turn_index}) ===")

    for layer_num in range(n_layers):
        print(f"\n{'-'*40}\nLayer {layer_num}\n{'-'*40}")
        trainer_cfg = TrainerConfig()
        probe = LinearProbeClassification(
            device=DEVICE, probe_class=1, input_dim=INPUT_DIM, logistic=LOGISTIC,
        )
        optimizer, scheduler = probe.configure_optimizers(trainer_cfg)
        best_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            train_results = train(
                probe, DEVICE, train_loader, optimizer, epoch,
                loss_func=loss_func, verbose=False, layer_num=layer_num,
                return_raw_outputs=True, one_hot=ONE_HOT,
                uncertainty=UNCERTAINTY, num_classes=2,
            )
            tr_loss, tr_acc = train_results[0], train_results[1]

            test_results = test(
                probe, DEVICE, test_loader, loss_func=loss_func,
                verbose=False, layer_num=layer_num, scheduler=scheduler,
                return_raw_outputs=True, one_hot=ONE_HOT,
                uncertainty=UNCERTAINTY, num_classes=2,
            )
            te_loss, te_acc = test_results[0], test_results[1]

            if te_acc > best_acc:
                best_acc = te_acc
                torch.save(
                    probe.state_dict(),
                    f"{probe_dir}/{out_subdir}/human_ai_probe_at_layer_{layer_num}.pth",
                )

            print(f"Epoch {epoch:02d}/{EPOCHS} | Train {tr_acc:.3f} | Test {te_acc:.3f}")

        torch.save(
            probe.state_dict(),
            f"{probe_dir}/{out_subdir}/human_ai_probe_at_layer_{layer_num}_final.pth",
        )

        te_preds, te_truths = test_results[2], test_results[3]
        cm = confusion_matrix(te_truths, te_preds)
        ConfusionMatrixDisplay(cm, display_labels=["AI", "Human"]).plot()
        plt.title(f"{probe_label.title()} Layer {layer_num} Acc {best_acc:.3f}")
        plt.savefig(f"{probe_dir}/{out_subdir}/cm_layer_{layer_num}.png", dpi=200)
        plt.close()

        acc_summary["acc"].append(best_acc)
        acc_summary["final"].append(te_acc)
        acc_summary["train"].append(tr_acc)
        torch.cuda.empty_cache()

    with open(f"{probe_dir}/{out_subdir}/accuracy_summary.pkl", "wb") as f:
        pickle.dump(acc_summary, f)

    print(f"\nFinished {out_subdir} probes — Mean Acc = {np.mean(acc_summary['acc']):.3f}")
    return acc_summary


if __name__ == "__main__":
    args = parse_args()
    ti = args.turn_index

    # Name the output directory by turn
    turn_label = f"turn_{ti+1}" if ti >= 0 else "turn_5"
    probe_dir = f"data/probe_checkpoints/{turn_label}"
    print(f"Turn index: {ti} → output dir: {probe_dir}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Loading LLaMA-2-13B-Chat model and tokenizer...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().cuda().eval()

    reading_summary = run_probe_train(
        model, tokenizer,
        control_probe=False,
        out_subdir="reading_probe",
        turn_index=ti,
        probe_dir=probe_dir,
    )

    control_summary = run_probe_train(
        model, tokenizer,
        control_probe=True,
        out_subdir="control_probe",
        turn_index=ti,
        probe_dir=probe_dir,
    )

    # Print comparison summary
    print("\n" + "=" * 60)
    print(f"PROBE TRAINING SUMMARY — {turn_label}")
    print("=" * 60)
    print(f"  Reading probe: Mean best acc = {np.mean(reading_summary['acc']):.3f}")
    print(f"                 Peak acc = {np.max(reading_summary['acc']):.3f} (layer {np.argmax(reading_summary['acc'])})")
    print(f"  Control probe: Mean best acc = {np.mean(control_summary['acc']):.3f}")
    print(f"                 Peak acc = {np.max(control_summary['acc']):.3f} (layer {np.argmax(control_summary['acc'])})")
