#!/usr/bin/env python3
"""
Alternative Token Position Probes

Train probes at non-standard token positions and with irrelevant suffixes
to test whether the partner identity representation is:
  (a) localized to the generation point, or broadcast everywhere
  (b) specific to partner-relevant reflection, or triggered by any suffix

Conditions:
  1. control_first:  Control probe at token position 0 (BOS <s>)
  2. control_last:   Control probe at last token [/INST] (BASELINE)
  3. reading_real:   Reading probe with "I think the conversation partner..." (BASELINE)
  4. reading_irrelevant: Reading probe with "I think the weather outside today is"

For each condition, trains probes across all 41 layers at turn 5.

Usage:
    python 1f_alternative_position_probes.py --condition control_first
    python 1f_alternative_position_probes.py --condition reading_irrelevant

Env: llama2_env (needs GPU)
Rachel C. Metzgar · Feb 2026
"""

import os, sys, argparse, csv, json, glob, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config, set_version, add_version_argument

from src.probes import LinearProbeClassification, TrainerConfig
from src.train_test_utils import train, test
from src.losses import edl_mse_loss
from src.dataset import llama_v2_prompt

# ========================== CONFIG ========================== #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(12345)

# Suffix definitions
REAL_SUFFIX = " I think the conversation partner of this user is"
IRRELEVANT_SUFFIX = " I think the weather outside today is"

CONDITIONS = {
    "control_first": {
        "description": "Control probe at BOS token (position 0)",
        "suffix": None,
        "token_position": "first",
    },
    "control_last": {
        "description": "Control probe at last token [/INST] (baseline)",
        "suffix": None,
        "token_position": "last",
    },
    "control_random": {
        "description": "Control probe at random mid-sequence token",
        "suffix": None,
        "token_position": "random",
    },
    "control_eos": {
        "description": "Control probe at </s> token ending first exchange",
        "suffix": None,
        "token_position": "eos_first",
    },
    "reading_real": {
        "description": "Reading probe with partner-relevant suffix (baseline)",
        "suffix": REAL_SUFFIX,
        "token_position": "last",
    },
    "reading_irrelevant": {
        "description": "Reading probe with irrelevant suffix (weather)",
        "suffix": IRRELEVANT_SUFFIX,
        "token_position": "last",
    },
}


# ========================== DATASET ========================== #
class AlternativePositionDataset(Dataset):
    """Dataset that extracts activations at configurable token positions."""

    def __init__(self, csv_dir, tokenizer, model, suffix, token_position,
                 turn_index=-1, max_length=2048):
        self.tokenizer = tokenizer
        self.model = model
        self.suffix = suffix
        self.token_position = token_position
        self.turn_index = turn_index
        self.max_length = max_length

        self.labels = []
        self.acts = []
        self.texts = []
        self.metadata = []

        csv_files = sorted(glob.glob(os.path.join(csv_dir, "s[0-9][0-9][0-9].csv")))
        assert len(csv_files) > 0, f"No sXXX.csv files found in {csv_dir}"
        print(f"Found {len(csv_files)} subject files")

        for csv_path in tqdm(csv_files, desc="Loading subjects"):
            subject_id = os.path.basename(csv_path).replace(".csv", "")
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            trials = {}
            for r in rows:
                t = int(r["trial"])
                trials.setdefault(t, []).append(r)

            for trial_num in sorted(trials.keys()):
                trial_rows = trials[trial_num]
                if turn_index == -1:
                    row = trial_rows[-1]
                elif turn_index < len(trial_rows):
                    row = trial_rows[turn_index]
                else:
                    continue

                partner_type = row["partner_type"]
                if "ai" in partner_type.lower():
                    label = 0
                elif "human" in partner_type.lower():
                    label = 1
                else:
                    continue

                try:
                    messages = json.loads(row["sub_input"])
                except (json.JSONDecodeError, KeyError):
                    continue
                if len(messages) < 2:
                    continue

                try:
                    text = llama_v2_prompt(messages)
                except Exception:
                    continue

                if self.suffix:
                    text += self.suffix

                acts = self._extract(text)
                if acts is None:
                    continue

                self.labels.append(label)
                self.acts.append(acts)
                self.texts.append(text)
                self.metadata.append({
                    "subject": subject_id,
                    "trial": trial_num,
                    "partner_type": partner_type,
                })
                torch.cuda.empty_cache()

        n_ai = sum(1 for l in self.labels if l == 0)
        n_human = sum(1 for l in self.labels if l == 1)
        print(f"Loaded {len(self.labels)} conversations (AI: {n_ai}, Human: {n_human})")

    def _extract(self, text):
        with torch.no_grad():
            encoding = self.tokenizer(
                text, truncation=True, max_length=self.max_length,
                return_attention_mask=True, return_tensors="pt",
            )
            input_ids = encoding["input_ids"]
            output = self.model(
                input_ids=input_ids.to(DEVICE),
                attention_mask=encoding["attention_mask"].to(DEVICE),
                output_hidden_states=True, return_dict=True,
            )

            seq_len = input_ids.shape[1]

            if self.token_position == "last":
                pos = -1
            elif self.token_position == "first":
                pos = 0
            elif self.token_position == "random":
                # Pick a random position in the middle 50% of the sequence
                # to avoid BOS/EOS/structural tokens at boundaries
                quarter = max(1, seq_len // 4)
                pos = np.random.randint(quarter, max(quarter + 1, seq_len - quarter))
            elif self.token_position == "eos_first":
                # Find the first </s> token (end of first exchange)
                # LLaMA EOS token id = 2
                eos_id = self.tokenizer.eos_token_id or 2
                ids = input_ids[0].tolist()
                eos_positions = [i for i, tok in enumerate(ids) if tok == eos_id]
                if eos_positions:
                    pos = eos_positions[0]  # first </s>
                else:
                    # Fallback: if no EOS found (single-turn), use last token
                    pos = -1
                    print(f"  WARNING: No </s> found, falling back to last token")
            else:
                pos = int(self.token_position)

            acts = torch.cat([
                hs[:, pos].detach().cpu().to(torch.float)
                for hs in output["hidden_states"]
            ])
            return acts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "hidden_states": self.acts[idx],
            "age": self.labels[idx],
            "text": self.texts[idx],
            "metadata": self.metadata[idx],
        }


# ========================== TRAINING ========================== #
def train_probes(dataset, out_dir):
    """Train probes across all layers."""
    os.makedirs(out_dir, exist_ok=True)

    n_layers = dataset.acts[0].shape[0] if len(dataset.acts) > 0 else 41
    print(f"Dataset size: {len(dataset)}, n_layers: {n_layers}")

    idx_train, idx_test = train_test_split(
        np.arange(len(dataset)), test_size=0.2,
        random_state=12345, stratify=dataset.labels,
    )
    train_ds, test_ds = Subset(dataset, idx_train), Subset(dataset, idx_test)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=config.TRAINING.batch_size_train,
                              num_workers=1, drop_last=True)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=config.TRAINING.batch_size_test,
                             num_workers=1, drop_last=False)

    loss_func = nn.BCELoss()
    acc_summary = {"acc": [], "final": [], "train": []}

    for layer_num in range(n_layers):
        print(f"\n{'-'*40}\nLayer {layer_num}\n{'-'*40}")
        trainer_cfg = TrainerConfig()
        probe = LinearProbeClassification(
            device=DEVICE, probe_class=1, input_dim=config.INPUT_DIM, logistic=True,
        )
        optimizer, scheduler = probe.configure_optimizers(trainer_cfg)
        best_acc = 0.0

        for epoch in range(1, config.TRAINING.epochs + 1):
            train_results = train(
                probe, DEVICE, train_loader, optimizer, epoch,
                loss_func=loss_func, verbose=False, layer_num=layer_num,
                return_raw_outputs=True, one_hot=False,
                uncertainty=False, num_classes=2,
            )
            tr_loss, tr_acc = train_results[0], train_results[1]

            test_results = test(
                probe, DEVICE, test_loader, loss_func=loss_func,
                verbose=False, layer_num=layer_num, scheduler=scheduler,
                return_raw_outputs=True, one_hot=False,
                uncertainty=False, num_classes=2,
            )
            te_loss, te_acc = test_results[0], test_results[1]

            if te_acc > best_acc:
                best_acc = te_acc
                torch.save(probe.state_dict(),
                          f"{out_dir}/human_ai_probe_at_layer_{layer_num}.pth")

            print(f"Epoch {epoch:02d}/{config.TRAINING.epochs} | Train {tr_acc:.3f} | Test {te_acc:.3f}")

        torch.save(probe.state_dict(),
                  f"{out_dir}/human_ai_probe_at_layer_{layer_num}_final.pth")

        te_preds, te_truths = test_results[2], test_results[3]
        cm = confusion_matrix(te_truths, te_preds)
        ConfusionMatrixDisplay(cm, display_labels=["AI", "Human"]).plot()
        plt.title(f"Layer {layer_num} Acc {best_acc:.3f}")
        plt.savefig(f"{out_dir}/cm_layer_{layer_num}.png", dpi=200)
        plt.close()

        acc_summary["acc"].append(best_acc)
        acc_summary["final"].append(te_acc)
        acc_summary["train"].append(tr_acc)
        torch.cuda.empty_cache()

    with open(f"{out_dir}/accuracy_summary.pkl", "wb") as f:
        pickle.dump(acc_summary, f)

    print(f"\nFinished — Mean Acc = {np.mean(acc_summary['acc']):.3f}, "
          f"Peak = {np.max(acc_summary['acc']):.3f} (layer {np.argmax(acc_summary['acc'])})")
    return acc_summary


# ========================== MAIN ========================== #
def parse_args():
    parser = argparse.ArgumentParser()
    add_version_argument(parser)
    parser.add_argument("--condition", required=True,
                        choices=list(CONDITIONS.keys()),
                        help="Which alternative probe condition to run.")
    parser.add_argument("--turn_index", type=int, default=-1,
                        help="Turn index (default -1 = turn 5).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_version(args.version)

    cond = CONDITIONS[args.condition]
    print(f"Version: {args.version}")
    print(f"Condition: {args.condition}")
    print(f"  Description: {cond['description']}")
    print(f"  Suffix: {cond['suffix']}")
    print(f"  Token position: {cond['token_position']}")
    print(f"  Turn index: {args.turn_index}")

    out_dir = str(config.PATHS.probe_checkpoints / "alternative" / args.condition)
    print(f"  Output: {out_dir}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("\nLoading LLaMA-2-13B-Chat...")
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, local_files_only=True)
    model.half().to(DEVICE).eval()

    dataset = AlternativePositionDataset(
        csv_dir=str(config.PATHS.csv_dir),
        tokenizer=tokenizer,
        model=model,
        suffix=cond["suffix"],
        token_position=cond["token_position"],
        turn_index=args.turn_index,
    )

    summary = train_probes(dataset, out_dir)

    print("\n" + "=" * 60)
    print(f"SUMMARY — {args.condition}")
    print("=" * 60)
    print(f"  Mean acc: {np.mean(summary['acc']):.3f}")
    print(f"  Peak acc: {np.max(summary['acc']):.3f} (layer {np.argmax(summary['acc'])})")
