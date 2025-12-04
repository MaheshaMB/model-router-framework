from __future__ import annotations

import json
from typing import Any, Dict, List

import boto3

from .base import BaseProviderClient


class BedrockProviderClient(BaseProviderClient):
    """
    Bedrock client wrapper using boto3 for text models and embeddings.
    """

    def __init__(self, model_id: str, params: Dict[str, Any]) -> None:
        super().__init__(model_id, params)
        self._runtime = boto3.client("bedrock-runtime")

    def chat(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Chat wrapper for Bedrock models using the Converse API.

        Supports all text/chat models that implement Converse
        (Amazon Nova, Anthropic Claude, Titan Text, Llama, Mistral, etc.).
        Expects messages in OpenAI-style:
          [{"role": "user"|"assistant"|"system", "content": "..."}, ...]
        """
        bedrock_messages = []
        system_prompts = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")

            if role == "system":
                # Collect system messages as a list of strings
                system_prompts.append({"text": content})
            else:
                # Bedrock Converse format: content is a list of blocks
                bedrock_messages.append(
                    {
                        "role": role,
                        "content": [
                            {
                                "text": content
                            }
                        ],
                    }
                )

        inference_config = {
            "maxTokens": self.params.get("max_tokens", 4096),
            "temperature": self.params.get("temperature", 0.2),
        }
        if "sonnet-4-5" not in self.model_id:
            # Anthropic Claude-specific config adjustments
            inference_config["topP"] = self.params.get("top_p", 0.8)
            
        body_kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": bedrock_messages,
            "inferenceConfig": inference_config,
        }
        if system_prompts:
            # Converse API: "system" is a list of content blocks
            body_kwargs["system"] = system_prompts

        print(f"==============> Bedrock converse model_id:{self.model_id} body: {body_kwargs}")
        print(f"<==============")

        # NOTE: Converse uses the 'converse' operation, not invoke_model
        resp = self._runtime.converse(**body_kwargs)

        # Typical Converse response shape:
        # resp["output"]["message"]["content"][0]["text"]
        return resp

    def embed(self, text: str) -> Any:
        body = {
            "inputText": text,
        }
        
        print(f"==============> Bedrock embedded model_id:{self.model_id} body: {json.dumps(body)}")
        print(f"<==============")

        resp = self._runtime.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        out = json.loads(resp["body"].read())
        # Adapt to your chosen embedding model's schema
        return out
