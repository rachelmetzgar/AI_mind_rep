"""
Lightweight wrapper for OpenAI chat models.

Author: Rachel C. Metzgar
"""

from typing import List, Dict
from openai import OpenAI


class ChatClient:
    """Wrapper for OpenAI chat models (e.g., gpt-4o)."""

    def __init__(self, model: str, max_tokens: int, temperature: float = 0.9):
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()
