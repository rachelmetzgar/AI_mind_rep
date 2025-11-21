#!/usr/bin/env python3
"""
Train linear probes for Human vs AI partner classification
Replicating TalkTuner (Chen et al., 2024) architecture, optimization,
and data handling — limited to binary Human vs AI classification.

Rachel C. Metzgar · Oct 2025
"""

import os, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Local imports ===
from src.dataset import TextDataset
from src.train_test_utils import train, test
from src.probes import LinearProbeClassification, TrainerConfig
from src.losses import edl_mse_loss

# ========================== CONFIG ========================== #
DATA_DIR = "data"
CONVO_DIR = f"{DATA_DIR}/human_ai_conversations"
PROBE_DIR = f"{DATA_DIR}/probe_checkpoints"
os.makedirs(f"{PROBE_DIR}/reading_probe", exist_ok=True)
os.makedirs(f"{PROBE_DIR}/control_probe", exist_ok=True)

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
)

INPUT_DIM = 4096  # LLaMA-2-7B hidden size
EPOCHS = 5        # TODO: increase to 50 for full run
BATCH_SIZE_TRAIN = 200
BATCH_SIZE_TEST = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(12345)

LOGISTIC = True      # Sigmoid output (binary)
UNCERTAINTY = False  # Switch to True to use edl_mse_loss
ONE_HOT = False

# ========================== MAIN TRAIN LOOP ========================== #
def run_probe_train(model, tokenizer, control_probe=False, out_subdir="reading_probe"):
    dataset = TextDataset(
        CONVO_DIR,
        tokenizer,
        model,
        label_idf="_partner_",
        label_to_id={"ai": 0, "human": 1},
        convert_to_llama2_format=True,
        control_probe=control_probe,
        residual_stream=True,
        new_format=False,
        if_augmented=False,
        remove_last_ai_response=False,
        include_inst=False,
        one_hot=ONE_HOT,
    )

    n_layers = dataset.acts[0].shape[0] if len(dataset.acts) > 0 else 40
    idx_train, idx_test = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        random_state=12345,
        stratify=dataset.labels,
    )
    train_ds, test_ds = Subset(dataset, idx_train), Subset(dataset, idx_test)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE_TRAIN, num_workers=1, drop_last=True)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=BATCH_SIZE_TEST, num_workers=1, drop_last=True)

    # Loss configuration
    if UNCERTAINTY:
        loss_func = edl_mse_loss
    else:
        loss_func = nn.BCELoss()

    acc_summary = {"acc": [], "final": [], "train": []}
    print(f"\n=== Training {'control' if control_probe else 'reading'} probe ===")

    for layer_num in range(2):  # TODO: increase to n_layers for full run
        print(f"\n{'-'*40}\nLayer {layer_num}\n{'-'*40}")
        trainer_cfg = TrainerConfig()
        probe = LinearProbeClassification(
            device=DEVICE,
            probe_class=1,  # binary
            input_dim=INPUT_DIM,
            logistic=LOGISTIC
        )
        optimizer, scheduler = probe.configure_optimizers(trainer_cfg)
        best_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            train_results = train(
                probe,
                DEVICE,
                train_loader,
                optimizer,
                epoch,
                loss_func=loss_func,
                verbose=False,
                layer_num=layer_num,
                return_raw_outputs=True,
                one_hot=ONE_HOT,
                uncertainty=UNCERTAINTY,
                num_classes=2,
            )
            tr_loss, tr_acc = train_results[0], train_results[1]

            test_results = test(
                probe,
                DEVICE,
                test_loader,
                loss_func=loss_func,
                verbose=False,
                layer_num=layer_num,
                scheduler=scheduler,
                return_raw_outputs=True,
                one_hot=ONE_HOT,
                uncertainty=UNCERTAINTY,
                num_classes=2,
            )
            te_loss, te_acc, te_preds, te_truths = test_results[0], test_results[1], test_results[2], test_results[3]

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
        plt.title(f"{'Control' if control_probe else 'Reading'} Layer {layer_num} Acc {best_acc:.3f}")
        plt.savefig(f"{PROBE_DIR}/{out_subdir}/cm_layer_{layer_num}.png", dpi=200)
        plt.close()

        acc_summary["acc"].append(best_acc)
        acc_summary["final"].append(te_acc)
        acc_summary["train"].append(tr_acc)
        torch.cuda.empty_cache()

    with open(f"{PROBE_DIR}/{out_subdir}/accuracy_summary.pkl", "wb") as f:
        pickle.dump(acc_summary, f)

    print(f"\n✅ Finished {out_subdir} probes – Mean Acc = {np.mean(acc_summary['acc']):.3f}")


# ========================== EXECUTION ========================== #
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Loading LLaMA-2 model and tokenizer …")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().cuda().eval()

    # Reading probe (uses “I think the partner …” activations)
    run_probe_train(model, tokenizer, control_probe=False, out_subdir="reading_probe")

    # Control probe (uses last-user-message activations)
    run_probe_train(model, tokenizer, control_probe=True, out_subdir="control_probe")
