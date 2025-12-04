from __future__ import annotations

from typing import Any, Dict, List

import google.generativeai as genai

from .base import BaseProviderClient


class GeminiProviderClient(BaseProviderClient):
    """
    Google Gemini client using google-generativeai SDK.
    """

    def __init__(self, model_id: str, params: Dict[str, Any]) -> None:
        super().__init__(model_id, params)
        # API key via env: GOOGLE_API_KEY
        genai.configure()
        self._model = genai.GenerativeModel(model_name=self.model_id)

    def chat(self, messages: List[Dict[str, Any]]) -> Any:
        # Flatten messages into a prompt (simple example)
        content = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages]
        )
        resp = self._model.generate_content(
            content,
            generation_config={
                "temperature": self.params.get("temperature", 0.2),
                "top_p": self.params.get("top_p", 0.8),
                "max_output_tokens": self.params.get("max_tokens", 4096),
            },
        )
        return resp

    def embed(self, text: str) -> Any:
        # If using Gemini embeddings – placeholder
        raise NotImplementedError("Embeddings not implemented for Gemini")
