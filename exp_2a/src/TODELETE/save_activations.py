from transformers import AutoTokenizer, AutoModelForCausalLM
from src.dataset_human_ai import TextDataset
import numpy as np, os, torch

# --- Load model ---
model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
model.half().cuda().eval()

# --- Build dataset ---
dataset = TextDataset(
    directory="data/human_ai_conversations",
    tokenizer=tokenizer,
    model=model,
    label_idf="_partner_",
    label_to_id={"human": 1, "ai": 0},
    convert_to_llama2_format=True,
    residual_stream=True,
)

# --- Save activations ---
out_dir = "data/activations_human_ai"
os.makedirs(out_dir, exist_ok=True)

for i, item in enumerate(dataset):
    np.savez_compressed(
        os.path.join(out_dir, f"conversation_{i:04d}_activations.npz"),
        human_label=item["partner"],
        read_hidden=item["hidden_states"].numpy(),   # or split into read/user separately if you capture both
        user_hidden=item["hidden_states"].numpy(),
    )

print("✅ Saved activations to", out_dir)
