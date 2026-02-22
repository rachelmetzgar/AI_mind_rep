#!/usr/bin/env python3
"""
Train linear probes for Human vs AI partner classification.
Updated to read directly from per-subject CSV files (sub_input column),
which contain the exact message lists fed to the participant LLM.

Probe types:
  - Reading probe: appends "I think the conversation partner of this user is"
    after the LLaMA-2-formatted conversation. Probes at last token.
  - Control probe: no suffix. The conversation ends with [/INST] after the
    partner's last message, so the model at the last token is about to
    generate the participant's next response. This is a pre-generation
    probe position.

Rachel C. Metzgar · Feb 2026
"""

import os, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Local imports ===
from src.dataset import TextDatasetCSV
from src.train_test_utils import train, test
from src.probes import LinearProbeClassification, TrainerConfig
from src.losses import edl_mse_loss

# ========================== CONFIG ========================== #
# Path to directory containing sXXX.csv files
# Points directly to exp_1/labels/ data (no local copy needed)
CSV_DIR = (
    "/jukebox/graziano/rachel/ai_mind_rep/exp_1/labels/"
    "data/meta-llama-Llama-2-13b-chat-hf/0.8"
)

DATA_DIR = "data"
PROBE_DIR = f"{DATA_DIR}/probe_checkpoints/turn_5"
os.makedirs(f"{PROBE_DIR}/reading_probe", exist_ok=True)
os.makedirs(f"{PROBE_DIR}/control_probe", exist_ok=True)

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    #"models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

INPUT_DIM = 5120      # LLaMA-2-13B hidden size
EPOCHS = 50
BATCH_SIZE_TRAIN = 200
BATCH_SIZE_TEST = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(12345)

LOGISTIC = True
UNCERTAINTY = False
ONE_HOT = False


# ========================== MAIN TRAIN LOOP ========================== #
def run_probe_train(
    model, tokenizer,
    control_probe=False,
    out_subdir="reading_probe",
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

    # Train/test split, stratified by label
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
    train_loader = DataLoader(
        train_ds, shuffle=True, batch_size=BATCH_SIZE_TRAIN,
        num_workers=1, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, batch_size=BATCH_SIZE_TEST,
        num_workers=1, drop_last=False,
    )

    # Loss
    loss_func = edl_mse_loss if UNCERTAINTY else nn.BCELoss()

    acc_summary = {"acc": [], "final": [], "train": []}
    print(f"\n=== Training {'control' if control_probe else 'reading'} probe ===")

    for layer_num in range(n_layers):
        print(f"\n{'-'*40}\nLayer {layer_num}\n{'-'*40}")
        trainer_cfg = TrainerConfig()
        probe = LinearProbeClassification(
            device=DEVICE,
            probe_class=1,  # binary
            input_dim=INPUT_DIM,
            logistic=LOGISTIC,
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
            te_preds, te_truths = test_results[2], test_results[3]

            if te_acc > best_acc:
                best_acc = te_acc
                torch.save(
                    probe.state_dict(),
                    f"{PROBE_DIR}/{out_subdir}/human_ai_probe_at_layer_{layer_num}.pth",
                )

            print(f"Epoch {epoch:02d}/{EPOCHS} | Train Acc {tr_acc:.3f} | Test Acc {te_acc:.3f}")

        # Save final weights
        torch.save(
            probe.state_dict(),
            f"{PROBE_DIR}/{out_subdir}/human_ai_probe_at_layer_{layer_num}_final.pth",
        )

        # Confusion matrix
        cm = confusion_matrix(te_truths, te_preds)
        ConfusionMatrixDisplay(cm, display_labels=["AI", "Human"]).plot()
        plt.title(
            f"{'Control' if control_probe else 'Reading'} Layer {layer_num} "
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

    print(f"\n✅ Finished {out_subdir} probes – Mean Acc = {np.mean(acc_summary['acc']):.3f}")


# ========================== EXECUTION ========================== #
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Loading LLaMA-2-13B-Chat model and tokenizer …")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().cuda().eval()

    # ── Reading probe ──
    # Appends "I think the conversation partner of this user is" to the
    # LLaMA-2 formatted conversation. Probes at last token of this suffix.
    run_probe_train(
        model, tokenizer,
        control_probe=False,
        out_subdir="reading_probe",
        turn_index=-1,   # full conversation (last turn)
    )

    # ── Control probe (pre-generation) ──
    # No suffix. Text ends with [/INST] after partner's last message.
    # Model at last token is about to generate participant's response.
    run_probe_train(
        model, tokenizer,
        control_probe=True,
        out_subdir="control_probe",
        turn_index=-1,   # full conversation (last turn)
    )