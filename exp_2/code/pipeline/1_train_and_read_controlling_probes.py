#!/usr/bin/env python3
"""
Train linear probes for Human vs AI partner classification.
Updated to read directly from per-subject CSV files (sub_input column),
which contain the exact message lists fed to the participant LLM.

Probe types:
  - Metacognitive probe: appends "I think my partner is"
    after the LLaMA-2-formatted conversation. Probes at last token.
  - Operational probe: no suffix. The conversation ends with [/INST] after the
    partner's last message, so the model at the last token is about to
    generate the participant's next response. This is a pre-generation
    probe position.

Usage:
    python code/pipeline/1_train_and_read_controlling_probes.py --version labels

Rachel C. Metzgar · Feb 2026
"""

import os, sys, pickle, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Config and local imports ===
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config, set_version, add_version_argument
from src.dataset import TextDatasetCSV
from src.train_test_utils import train, test
from src.probes import LinearProbeClassification, TrainerConfig
from src.losses import edl_mse_loss


# ========================== MAIN TRAIN LOOP ========================== #
def run_probe_train(
    model, tokenizer,
    control_probe=False,
    out_subdir="metacognitive",
    turn_index=-1,
):
    """Train probes across all layers for a given probe type.

    Parameters
    ----------
    control_probe : bool
        False = reading probe, True = control probe.
    out_subdir : str
        Subdirectory name under PROBE_DIR for saving checkpoints.
    turn_index : int
        Which conversation turn to use. -1 = last turn (full conversation).
    """
    PROBE_DIR = config.PATHS.probe_checkpoints / "turn_5"
    os.makedirs(f"{PROBE_DIR}/{out_subdir}", exist_ok=True)

    dataset = TextDatasetCSV(
        csv_dir=str(config.PATHS.csv_dir),
        tokenizer=tokenizer,
        model=model,
        control_probe=control_probe,
        label_idf="_partner_",
        label_to_id={"ai": 0, "human": 1},
        residual_stream=True,
        turn_index=turn_index,
        max_length=2048,
        one_hot=config.TRAINING.one_hot,
    )

    n_layers = dataset.acts[0].shape[0] if len(dataset.acts) > 0 else config.N_LAYERS
    print(f"\nDataset size: {len(dataset)}, n_layers: {n_layers}")

    # Train/test split, stratified by label
    labels_for_split = [
        l if isinstance(l, int) else l.item() if isinstance(l, torch.Tensor) and l.dim() == 0 else torch.argmax(l).item()
        for l in dataset.labels
    ]
    idx_train, idx_test = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        random_state=config.TRAINING.seed,
        stratify=labels_for_split,
    )
    train_ds, test_ds = Subset(dataset, idx_train), Subset(dataset, idx_test)
    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=config.TRAINING.batch_size_train,
        num_workers=1, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, batch_size=config.TRAINING.batch_size_test,
        num_workers=1, drop_last=False,
    )

    # Loss
    loss_func = edl_mse_loss if config.TRAINING.uncertainty else nn.BCELoss()

    acc_summary = {"acc": [], "final": [], "train": []}
    print(f"\n=== Training {'operational' if control_probe else 'metacognitive'} probe ===")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    for layer_num in range(n_layers):
        print(f"\n{'-'*40}\nLayer {layer_num}\n{'-'*40}")
        trainer_cfg = TrainerConfig()
        probe = LinearProbeClassification(
            device=DEVICE,
            probe_class=1,  # binary
            input_dim=config.INPUT_DIM,
            logistic=config.TRAINING.logistic,
        )
        optimizer, scheduler = probe.configure_optimizers(trainer_cfg)
        best_acc = 0.0

        for epoch in range(1, config.TRAINING.epochs + 1):
            train_results = train(
                probe, DEVICE, train_loader, optimizer, epoch,
                loss_func=loss_func, verbose=False, layer_num=layer_num,
                return_raw_outputs=True, one_hot=config.TRAINING.one_hot,
                uncertainty=config.TRAINING.uncertainty, num_classes=2,
            )
            tr_loss, tr_acc = train_results[0], train_results[1]

            test_results = test(
                probe, DEVICE, test_loader, loss_func=loss_func,
                verbose=False, layer_num=layer_num, scheduler=scheduler,
                return_raw_outputs=True, one_hot=config.TRAINING.one_hot,
                uncertainty=config.TRAINING.uncertainty, num_classes=2,
            )
            te_loss, te_acc = test_results[0], test_results[1]
            te_preds, te_truths = test_results[2], test_results[3]

            if te_acc > best_acc:
                best_acc = te_acc
                torch.save(
                    probe.state_dict(),
                    f"{PROBE_DIR}/{out_subdir}/human_ai_probe_at_layer_{layer_num}.pth",
                )

            print(f"Epoch {epoch:02d}/{config.TRAINING.epochs} | Train Acc {tr_acc:.3f} | Test Acc {te_acc:.3f}")

        # Save final weights
        torch.save(
            probe.state_dict(),
            f"{PROBE_DIR}/{out_subdir}/human_ai_probe_at_layer_{layer_num}_final.pth",
        )

        # Confusion matrix
        cm = confusion_matrix(te_truths, te_preds)
        ConfusionMatrixDisplay(cm, display_labels=["AI", "Human"]).plot()
        plt.title(
            f"{'Operational' if control_probe else 'Metacognitive'} Layer {layer_num} "
            f"Acc {best_acc:.3f}"
        )
        plt.savefig(f"{PROBE_DIR}/{out_subdir}/cm_layer_{layer_num}.png", dpi=200)
        plt.close()

        acc_summary["acc"].append(best_acc)
        acc_summary["final"].append(te_acc)
        acc_summary["train"].append(tr_acc)
        torch.cuda.empty_cache()

    # Save summary
    with open(f"{PROBE_DIR}/{out_subdir}/accuracy_summary.pkl", "wb") as f:
        pickle.dump(acc_summary, f)

    print(f"\nFinished {out_subdir} probes – Mean Acc = {np.mean(acc_summary['acc']):.3f}")


# ========================== EXECUTION ========================== #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train probes for Human vs AI classification.")
    add_version_argument(parser)
    args = parser.parse_args()
    set_version(args.version)

    torch.manual_seed(config.TRAINING.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"Loading LLaMA-2-13B-Chat model and tokenizer ...")
    print(f"Version: {args.version}")
    print(f"CSV dir: {config.PATHS.csv_dir}")
    print(f"Probe dir: {config.PATHS.probe_checkpoints}")

    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, local_files_only=True)
    model.half().cuda().eval()

    # -- Metacognitive probe --
    run_probe_train(
        model, tokenizer,
        control_probe=False,
        out_subdir="metacognitive",
        turn_index=-1,
    )

    # -- Operational probe (pre-generation) --
    run_probe_train(
        model, tokenizer,
        control_probe=True,
        out_subdir="operational",
        turn_index=-1,
    )
