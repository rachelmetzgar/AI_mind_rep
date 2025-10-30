"""
utils/gpt_clients.py

Provides lightweight wrapper for OpenAI model.

Author: Rachel C. Metzgar
Date: 2025-10-08
"""

from typing import List, Dict
from openai import OpenAI


# --------------------------- OpenAI Chat Wrapper --------------------------- #
class ChatClient:
    """Wrapper for OpenAI chat models (e.g., gpt-4o)."""

    def __init__(self, model: str, max_tokens: int, temperature: float = 0.9):
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a completion given chat-formatted messages."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()

