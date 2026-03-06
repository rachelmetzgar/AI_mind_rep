"""
Wrapper for LLaMA models (base and chat).
Chat models use the [INST] template and accept OpenAI-style message lists.

Author: Rachel C. Metzgar
"""

import torch
from typing import List, Dict, Union
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaClient:
    """Wrapper for LLaMA-2-Chat models. Accepts OpenAI-style message lists."""

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"

    def __init__(self, model_name: str, temperature: float = 0.8, max_new_tokens: int = 500):
        print(f"Loading model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        system_msg = None

        if messages and messages[0]["role"] == "system":
            system_msg = messages[0]["content"]
            messages = messages[1:]

        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                if i == 0 and system_msg:
                    prompt += (
                        f"{self.BOS}{self.B_INST} "
                        f"{self.B_SYS}{system_msg}{self.E_SYS}"
                        f"{msg['content']} {self.E_INST}"
                    )
                else:
                    prompt += f"{self.BOS}{self.B_INST} {msg['content']} {self.E_INST}"
            elif msg["role"] == "assistant":
                prompt += f" {msg['content']} {self.EOS}"

        return prompt

    def generate(self, messages: Union[str, List[Dict[str, str]]], temperature: float = None) -> str:
        temp = temperature if temperature is not None else self.temperature

        if isinstance(messages, list):
            prompt = self._format_messages(messages)
        else:
            prompt = messages

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            output[0][prompt_len:], skip_special_tokens=True
        ).strip()

        return response
