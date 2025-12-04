from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseProviderClient(ABC):
    """
    Abstract base for provider-specific clients.
    """

    def __init__(self, model_id: str, params: Dict[str, Any]) -> None:
        self.model_id = model_id
        self.params = params

    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> Any:
        raise NotImplementedError
