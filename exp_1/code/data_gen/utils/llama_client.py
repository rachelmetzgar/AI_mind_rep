"""
utils/llama_client.py

Provides lightweight wrapper for base LLaMA model.

Author: Rachel C. Metzgar
Date: 2025-10-08
"""

import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------- LLaMA Base Wrapper ---------------------------- #
class LlamaClient:
    """Wrapper for base (non-chat) LLaMA models via Hugging Face."""

    def __init__(self, model_name: str, temperature: float = 0.9, max_new_tokens: int = 200):
        print(f"Loading model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        """Generate a raw text continuation given a prompt string."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(prompt):].strip()
