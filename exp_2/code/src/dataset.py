"""
TextDatasetCSV — Load participant conversations from CSV files.

Reads sub_input (the exact message list fed to the participant LLM)
from per-subject CSV files. For each conversation, extracts the last
interaction (turn 5, containing the full conversation) and converts
to LLaMA-2 chat format for activation extraction.

Probe positions:
  - Reading probe: appends reflective prompt, probes at last token
  - Control probe:  no suffix; probes at last token of LLaMA-2 text,
    which is [/INST] after the partner's last message — the position
    where the model is about to generate the participant's next response.

Rachel C. Metzgar · Feb 2026
"""

from __future__ import annotations

import os, csv, json, glob
from typing import Optional

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from tqdm.auto import tqdm
from collections import OrderedDict


# ── Hooks for capturing activations ─────────────────────────
class ModuleHook:
    def __init__(self, module: torch.nn.Module) -> None:
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = []

    def hook_fn(self, module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.module = module
        self.features.append(output.detach())

    def close(self) -> None:
        self.hook.remove()


# ── LLaMA-2 chat formatting ────────────────────────────────
def llama_v2_prompt(messages: list[dict]) -> str:
    """Format a message list into LLaMA-2-Chat token string.

    Expects messages[0] to be role='system'. System content is
    folded into the first [INST] block following Meta's format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"

    assert messages[0]["role"] == "system", \
        f"First message must be system, got {messages[0]['role']}"

    # Fold system prompt into first user message
    merged = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    # Pair up user/assistant turns
    parts = [
        f"{BOS}{B_INST} {prompt['content'].strip()} {E_INST} {answer['content'].strip()} {EOS}"
        for prompt, answer in zip(merged[::2], merged[1::2])
    ]
    # If the last message is unpaired user, add [INST]...[/INST] without response
    if merged[-1]["role"] == "user":
        parts.append(
            f"{BOS}{B_INST} {merged[-1]['content'].strip()} {E_INST}"
        )

    return "".join(parts)


# ── Reading-probe prompt translator ────────────────────────
prompt_translator = {
    "_partner_": "conversation partner",
    "_age_": "age",
    "_gender_": "gender",
    "_socioeco_": "socioeconomic status",
    "_education_": "education level",
}


# ── Main Dataset ───────────────────────────────────────────
class TextDatasetCSV(Dataset):
    """Load conversations from per-subject CSV files.

    Parameters
    ----------
    csv_dir : str
        Directory containing sXXX.csv files.
    tokenizer, model :
        HuggingFace tokenizer and model (already on GPU).
    control_probe : bool
        If False → reading probe (appends reflective prompt).
        If True  → control probe (no suffix; last token = [/INST]).
    label_idf : str
        Key into prompt_translator for reading-probe suffix.
    label_to_id : dict
        Maps partner type strings to integer labels.
    residual_stream : bool
        If True, extract from output hidden states (residual stream).
        If False, extract from MLP outputs.
    turn_index : int or None
        Which turn to use (0-indexed within each trial).
        None or -1 → last turn (full conversation). Default: -1.
    max_length : int
        Max token length for truncation. Default: 2048.
    """

    def __init__(
        self,
        csv_dir: str,
        tokenizer: "transformers.PreTrainedTokenizer",
        model: torch.nn.Module,
        control_probe: bool = False,
        label_idf: str = "_partner_",
        label_to_id: Optional[dict[str, int]] = None,
        residual_stream: bool = True,
        turn_index: int = -1,
        max_length: int = 2048,
        one_hot: bool = False,
    ) -> None:
        self.csv_dir = csv_dir
        self.tokenizer = tokenizer
        self.model = model
        self.control_probe = control_probe
        self.label_idf = label_idf
        self.label_to_id = label_to_id or {"ai": 0, "human": 1}
        self.residual_stream = residual_stream
        self.turn_index = turn_index
        self.max_length = max_length
        self.one_hot = one_hot

        # Storage
        self.labels = []
        self.acts = []
        self.texts = []
        self.metadata = []  # (subject, trial, agent, partner_type, partner_name)

        # Discover CSV files (non-clean versions)
        self.csv_files = sorted(
            glob.glob(os.path.join(csv_dir, "s[0-9][0-9][0-9].csv"))
        )
        assert len(self.csv_files) > 0, f"No sXXX.csv files found in {csv_dir}"
        print(f"Found {len(self.csv_files)} subject files in {csv_dir}")

        self._load_all()

    def _load_all(self) -> None:
        """Load and process all conversations."""
        for csv_path in tqdm(self.csv_files, desc="Loading subjects"):
            subject_id = os.path.basename(csv_path).replace(".csv", "")

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Group rows by trial
            trials = {}
            for r in rows:
                t = int(r["trial"])
                if t not in trials:
                    trials[t] = []
                trials[t].append(r)

            # Process each trial
            for trial_num in sorted(trials.keys()):
                trial_rows = trials[trial_num]

                # Select the desired turn
                if self.turn_index == -1 or self.turn_index is None:
                    row = trial_rows[-1]  # last turn = full conversation
                else:
                    if self.turn_index >= len(trial_rows):
                        continue
                    row = trial_rows[self.turn_index]

                # Parse label from partner_type
                partner_type = row["partner_type"]
                if "AI" in partner_type or "ai" in partner_type.lower():
                    label_str = "ai"
                elif "Human" in partner_type or "human" in partner_type.lower():
                    label_str = "human"
                else:
                    print(f"  Skipping unknown partner_type: {partner_type}")
                    continue

                if label_str not in self.label_to_id:
                    continue
                label = self.label_to_id[label_str]

                if self.one_hot:
                    label = F.one_hot(
                        torch.tensor([label], dtype=torch.long),
                        len(self.label_to_id),
                    )

                # Parse sub_input → message list → LLaMA-2 format
                try:
                    messages = json.loads(row["sub_input"])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  Skipping {subject_id} trial {trial_num}: {e}")
                    continue

                if len(messages) < 2:
                    print(f"  Skipping {subject_id} trial {trial_num}: too few messages")
                    continue

                try:
                    text = llama_v2_prompt(messages)
                except Exception as e:
                    print(f"  Skipping {subject_id} trial {trial_num}: format error: {e}")
                    continue

                # Append probe-specific suffix
                if not self.control_probe:
                    # Reading probe: reflective prompt
                    text += " I think my partner is"
                # Control probe: no suffix.
                # Text already ends with [/INST] after partner's last message.
                # Model at last token is about to generate participant's response.

                # ── Extract activations ──
                acts = self._extract_activations(text)
                if acts is None:
                    continue

                # Store
                self.texts.append(text)
                self.labels.append(label)
                self.acts.append(acts)
                self.metadata.append({
                    "subject": subject_id,
                    "trial": trial_num,
                    "agent": row["agent"],
                    "partner_type": partner_type,
                    "partner_name": row.get("partner_name", ""),
                    "topic": row.get("topic", ""),
                })

                torch.cuda.empty_cache()

        print(
            f"\nLoaded {len(self.labels)} conversations "
            f"(AI: {sum(1 for l in self.labels if (l == 0 if isinstance(l, int) else l.item() == 0))}, "
            f"Human: {sum(1 for l in self.labels if (l == 1 if isinstance(l, int) else l.item() == 1))})"
        )

    def _extract_activations(self, text: str) -> Optional[torch.Tensor]:
        """Tokenize text, run forward pass, return stacked last-token activations."""
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
                return_tensors="pt",
            )

            if self.residual_stream:
                output = self.model(
                    input_ids=encoding["input_ids"].to("cuda"),
                    attention_mask=encoding["attention_mask"].to("cuda"),
                    output_hidden_states=True,
                    return_dict=True,
                )
                # Stack last-token hidden states from all layers
                # output["hidden_states"] is a tuple of (n_layers+1,) tensors
                last_acts = []
                for layer_hs in output["hidden_states"]:
                    last_acts.append(
                        layer_hs[:, -1].detach().cpu().clone().to(torch.float)
                    )
                return torch.cat(last_acts)  # shape: [n_layers+1, hidden_dim]
            else:
                # MLP-output extraction
                features = OrderedDict()
                for name, module in self.model.named_modules():
                    if name.endswith(".mlp") or name.endswith(".embed_tokens"):
                        features[name] = ModuleHook(module)

                output = self.model(
                    input_ids=encoding["input_ids"].to("cuda"),
                    attention_mask=encoding["attention_mask"].to("cuda"),
                    output_hidden_states=True,
                    return_dict=True,
                )
                for feature in features.values():
                    feature.close()

                n_layers = getattr(
                    self.model.config, "num_hidden_layers",
                    len(output["hidden_states"]) - 1,
                )
                last_acts = [
                    features["model.embed_tokens"].features[0][:, -1]
                    .detach().cpu().clone().to(torch.float)
                ]
                for layer_num in range(1, n_layers + 1):
                    last_acts.append(
                        features[f"model.layers.{layer_num - 1}.mlp"]
                        .features[0][:, -1]
                        .detach().cpu().clone().to(torch.float)
                    )
                return torch.cat(last_acts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {
            "hidden_states": self.acts[idx],
            "age": self.labels[idx],  # kept as "age" for compatibility with train/test code
            "text": self.texts[idx],
            "metadata": self.metadata[idx],
        }