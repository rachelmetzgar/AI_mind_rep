#!/usr/bin/env python3
"""
Train linear reading and control probes for Human vs AI partner classification
EXACTLY following TalkTuner (Chen et al., 2024) implementation style.

Implements logistic probe training per layer with Adam optimizer,
weight decay, cosine LR schedule, BCELoss, 50 epochs, 80/20 split.

Rachel C. Metzgar, Oct 2025
"""

import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm


# ========================== CONFIG ========================== #
ACTIVATION_DIR = "data/activations_human_ai"
PROBE_OUT_DIR = "probe_checkpoints"
os.makedirs(os.path.join(PROBE_OUT_DIR, "reading_probe"), exist_ok=True)
os.makedirs(os.path.join(PROBE_OUT_DIR, "control_probe"), exist_ok=True)

INPUT_DIM = 5120           # LLaMA-2-13B hidden size
LR = 1e-3
BATCH_SIZE_TRAIN = 200
BATCH_SIZE_TEST = 400
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(12345)

# ========================== DATASET ========================== #
class ActivationDataset(Dataset):
    """Loads .npz activation files (saved from Phase 2)."""
    def __init__(self, files, key="read_hidden"):
        self.data = []
        self.labels = []
        for f in tqdm(files, desc=f"Loading {key}"):
            dat = np.load(f)
            arr = dat[key]  # [layers, seq_len, dim]
            label = int(dat["human_label"])  # 1=human, 0=AI
            arr_mean = arr.mean(axis=1)  # average across sequence length
            self.data.append(arr_mean)
            self.labels.append(label)
        self.data = np.stack(self.data)  # [N, layers, dim]
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), self.labels[idx]


# ========================== MODEL ========================== #
class LinearProbeClassification(nn.Module):
    """Logistic linear probe (matches TalkTuner’s implementation)."""
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.proj(x).squeeze(-1)
        return self.sigmoid(logits)

    def configure_optimizers(self):
        """Adam + cosine schedule exactly as in TalkTuner TrainerConfig."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=LR,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=LR * 0.1
        )
        return optimizer, scheduler


# ========================== TRAIN / TEST ========================== #
def train(model, device, loader, optimizer, epoch, loss_func, layer_num):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for X, y in loader:
        x_layer = X[:, layer_num, :].to(device)
        y = y.float().to(device)
        optimizer.zero_grad()
        out = model(x_layer)
        loss = loss_func(out, y)
        loss.backward()
        optimizer.step()
        preds = (out > 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total


def test(model, device, loader, loss_func, layer_num):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            x_layer = X[:, layer_num, :].to(device)
            y = y.float().to(device)
            out = model(x_layer)
            loss = loss_func(out, y)
            preds = (out > 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return loss_sum / total, correct / total, y_true, y_pred


# ========================== MAIN LOOP ========================== #
def run_probe_train(key="read_hidden", out_subdir="reading_probe"):
    files = sorted(glob.glob(os.path.join(ACTIVATION_DIR, "*.npz")))
    dataset = ActivationDataset(files, key)
    n_layers = dataset.data.shape[1]

    idx_train, idx_test = train_test_split(
        np.arange(len(dataset)), test_size=0.2,
        random_state=12345, stratify=dataset.labels
    )
    train_ds = Subset(dataset, idx_train)
    test_ds = Subset(dataset, idx_test)

    train_loader = DataLoader(train_ds, shuffle=True, pin_memory=True,
                              batch_size=BATCH_SIZE_TRAIN, num_workers=1)
    test_loader = DataLoader(test_ds, shuffle=False, pin_memory=True,
                             batch_size=BATCH_SIZE_TEST, num_workers=1)

    loss_func = nn.BCELoss()
    accuracy_dict = {"acc": [], "final": [], "train": []}

    print(f"\n=== Training probes for {key} ===")

    for layer_num in range(n_layers):
        print(f"\n{'-'*40}\nLayer {layer_num}\n{'-'*40}")

        model = LinearProbeClassification(input_dim=INPUT_DIM).to(DEVICE)
        optimizer, scheduler = model.configure_optimizers()

        best_acc = 0.0
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train(model, DEVICE, train_loader, optimizer, epoch, loss_func, layer_num)
            test_loss, test_acc, y_true, y_pred = test(model, DEVICE, test_loader, loss_func, layer_num)
            scheduler.step()

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(),
                           f"{PROBE_OUT_DIR}/{out_subdir}/human_ai_probe_at_layer_{layer_num}.pth")

            if epoch % 10 == 0 or epoch == EPOCHS:
                print(f"Epoch {epoch:02d}/{EPOCHS} | Train Acc={train_acc:.3f} | Test Acc={test_acc:.3f}")

        # Save final probe after all epochs
        torch.save(model.state_dict(),
                   f"{PROBE_OUT_DIR}/{out_subdir}/human_ai_probe_at_layer_{layer_num}_final.pth")

        # Confusion matrix and accuracy tracking
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["AI", "Human"]).plot()
        plt.title(f"{key} | Layer {layer_num} | Acc={best_acc:.3f}")
        plt.show()

        accuracy_dict["acc"].append(best_acc)
        accuracy_dict["final"].append(test_acc)
        accuracy_dict["train"].append(train_acc)

        torch.cuda.empty_cache()

    # Save accuracy summary
    with open(f"{PROBE_OUT_DIR}/{out_subdir}/accuracy_summary.pkl", "wb") as f:
        pickle.dump(accuracy_dict, f)

    print(f"\n✅ Finished training {key} probes. Mean accuracy = {np.mean(accuracy_dict['acc']):.3f}")


# ========================== EXECUTION ========================== #
if __name__ == "__main__":
    # Reading probe (uses "I think the partner..." activations)
    run_probe_train(key="read_hidden", out_subdir="reading_probe")

    # Control probe (uses last user message activations)
    run_probe_train(key="user_hidden", out_subdir="control_probe")
