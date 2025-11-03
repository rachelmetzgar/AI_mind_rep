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
matplotlib.use("Agg")          # headless-safe
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dataset import TextDataset   # your dataset.py

# ========================== CONFIG ========================== #
DATA_DIR = "data/human_ai_conversations"
PROBE_OUT_DIR = "probe_checkpoints"
os.makedirs(f"{PROBE_OUT_DIR}/reading_probe", exist_ok=True)
os.makedirs(f"{PROBE_OUT_DIR}/control_probe", exist_ok=True)

MODEL_NAME = "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"

INPUT_DIM = 4096            # LLaMA-2-7B hidden size
LR = 1e-3
EPOCHS = 5  # TODO: increase to 50 for full run
BATCH_SIZE_TRAIN = 200
BATCH_SIZE_TEST  = 400
WARMUP_EPOCHS = 5
K_LAST_TOKENS = 20          # mean-pool window
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(12345)

# ========================== PROBE MODEL ========================== #
class LinearProbe(nn.Module):
    """TalkTuner logistic probe."""
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.proj(x).squeeze(-1))

# ========================== OPTIM CONFIG ========================== #
def configure_optim(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR * 0.1
    )
    return optimizer, scheduler

# ========================== TRAIN / TEST ========================== #
def train_one_epoch(model, loader, opt, loss_fn, layer, warmup_steps, step):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        x = batch["hidden_states"].to(DEVICE)   # [B, n_layers, H]
        y = batch["age"].float().to(DEVICE)

        # Select activations from the current layer
        x_layer = x[:, layer, :]  # no token dim
        opt.zero_grad()
        out = model(x_layer)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        preds = (out > 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
        step += 1
        if step < warmup_steps:
            for g in opt.param_groups:
                g["lr"] = LR * step / warmup_steps
    return loss_sum / total, correct / total, step


def eval_epoch(model, loader, loss_fn, layer):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["hidden_states"].to(DEVICE)  # [B, n_layers, H]
            y = batch["age"].float().to(DEVICE)
            x_layer = x[:, layer, :]  # no token dim
            out = model(x_layer)
            loss = loss_fn(out, y)
            preds = (out > 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return loss_sum / total, correct / total, y_true, y_pred

# ========================== MAIN TRAIN LOOP ========================== #
def run_probe_train(model, tokenizer, control_probe=False, out_subdir="reading_probe"):
    dataset = TextDataset(
        DATA_DIR, tokenizer, model,
        label_idf="_partner_",   # <-- FIXED to match filenames
        label_to_id={"ai": 0, "human": 1},
        convert_to_llama2_format=True,
        control_probe=control_probe,
        residual_stream=True,
        new_format=False,
        if_augmented=False,
        remove_last_ai_response=False,
        include_inst=False,
        one_hot=False,
        k=K_LAST_TOKENS,
    )

    n_layers = dataset.acts[0].shape[0] if len(dataset.acts) > 0 else 40
    idx_train, idx_test = train_test_split(
        np.arange(len(dataset)), test_size=0.2,
        random_state=12345, stratify=dataset.labels
    )
    train_ds, test_ds = Subset(dataset, idx_train), Subset(dataset, idx_test)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE_TRAIN, num_workers=1)
    test_loader  = DataLoader(test_ds,  shuffle=False, batch_size=BATCH_SIZE_TEST,  num_workers=1)

    loss_fn = nn.BCELoss()
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    acc_summary = {"acc": [], "final": [], "train": []}

    print(f"\n=== Training {'control' if control_probe else 'reading'} probe ===")

    for layer in range(2):  # TODO: increase to n_layers for full run
        print(f"\n{'-'*40}\nLayer {layer}\n{'-'*40}")
        probe = LinearProbe(INPUT_DIM).to(DEVICE)
        opt, sched = configure_optim(probe)
        best_acc, step = 0.0, 0

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc, step = train_one_epoch(probe, train_loader, opt, loss_fn, layer, warmup_steps, step)
            te_loss, te_acc, y_t, y_p = eval_epoch(probe, test_loader, loss_fn, layer)
            if epoch > WARMUP_EPOCHS:
                sched.step()
            if te_acc > best_acc:
                best_acc = te_acc
                torch.save(probe.state_dict(),
                           f"{PROBE_OUT_DIR}/{out_subdir}/human_ai_probe_at_layer_{layer}.pth")
            if epoch % 10 == 0 or epoch == EPOCHS:
                print(f"Epoch {epoch:02d}/{EPOCHS} | Train Acc {tr_acc:.3f} | Test Acc {te_acc:.3f}")

        torch.save(probe.state_dict(),
                   f"{PROBE_OUT_DIR}/{out_subdir}/human_ai_probe_at_layer_{layer}_final.pth")

        cm = confusion_matrix(y_t, y_p)
        ConfusionMatrixDisplay(cm, display_labels=["AI", "Human"]).plot()
        plt.title(f"{'Control' if control_probe else 'Reading'} Layer {layer} Acc {best_acc:.3f}")
        plt.savefig(f"{PROBE_OUT_DIR}/{out_subdir}/cm_layer_{layer}.png", dpi=200)
        plt.close()

        acc_summary["acc"].append(best_acc)
        acc_summary["final"].append(te_acc)
        acc_summary["train"].append(tr_acc)
        torch.cuda.empty_cache()

    with open(f"{PROBE_OUT_DIR}/{out_subdir}/accuracy_summary.pkl", "wb") as f:
        pickle.dump(acc_summary, f)
    print(f"\n✅ Finished {out_subdir} probes – Mean Acc = {np.mean(acc_summary['acc']):.3f}")

# ========================== EXECUTION ========================== #
if __name__ == "__main__":
    print("Loading LLaMA-2 model and tokenizer …")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().cuda().eval()

    # Reading probe (uses “I think the partner …” activations)
    run_probe_train(model, tokenizer, control_probe=False, out_subdir="reading_probe")

    # Control probe (uses last-user-message activations)
    run_probe_train(model, tokenizer, control_probe=True, out_subdir="control_probe")
