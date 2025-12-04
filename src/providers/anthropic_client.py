from __future__ import annotations

from typing import Any, Dict, List

import anthropic

from .base import BaseProviderClient


class AnthropicProviderClient(BaseProviderClient):
    """
    Anthropic Claude client using anthropic SDK.
    """

    def __init__(self, model_id: str, params: Dict[str, Any]) -> None:
        super().__init__(model_id, params)
        self._client = anthropic.Anthropic()  # API key via env var

    def chat(self, messages: List[Dict[str, Any]]) -> Any:
        # Convert OpenAI-style messages to Claude messages
        claude_messages = []
        for m in messages:
            if m["role"] == "user":
                claude_messages.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                claude_messages.append({"role": "assistant", "content": m["content"]})

        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=self.params.get("max_tokens", 4096),
            temperature=self.params.get("temperature", 0.3),
            top_p=self.params.get("top_p", 0.9),
            messages=claude_messages,
        )
        return resp

    def embed(self, text: str) -> Any:
        # If you use Claude embeddings in future – placeholder
        raise NotImplementedError("Embeddings not implemented for Anthropic")
