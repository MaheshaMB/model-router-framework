from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal


CostTier = Literal["low", "medium", "high"]
TenantTier = Literal["free", "standard", "premium", "internal"]
TaskType = Literal["chat", "embedding"]


@dataclass
class RetryPolicy:
    max_attempts: int = 2
    backoff_ms: int = 200


@dataclass
class ModelConfig:
    id: str
    provider: str
    type: TaskType
    model_id: str
    max_context_tokens: Optional[int] = None
    max_chunk_tokens: Optional[int] = None
    languages: Optional[List[str]] = None
    strengths: Optional[List[str]] = None
    cost_tier: CostTier = "medium"
    default_params: Dict[str, Any] = field(default_factory=dict)
    backup_model_id: Optional[str] = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass
class RuleConditionRange:
    gte: Optional[int] = None
    lt: Optional[int] = None


@dataclass
class RuleCondition:
    task_type: Optional[TaskType] = None
    complexity: Optional[str] = None
    language: Optional[str] = None
    tenant_tiers: Optional[List[TenantTier]] = None
    context_tokens_range: Optional[RuleConditionRange] = None
    chunk_tokens_range: Optional[RuleConditionRange] = None


@dataclass
class RoutingRule:
    id: str
    when: RuleCondition
    use_model: str
    override_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterConfig:
    models: Dict[str, ModelConfig]
    rules: List[RoutingRule]
    default_chat_model_id: Optional[str] = None
    default_embedding_model_id: Optional[str] = None


@dataclass
class TaskDescriptor:
    text: str
    task_type: TaskType = "chat"
    context_tokens: Optional[int] = None
    tenant_id: Optional[str] = None
    tenant_tier: TenantTier = "standard"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSummary:
    task_type: TaskType
    token_count: int
    size_class: str
    language: str
    complexity: str
    context_tokens: int
    tenant_tier: TenantTier


@dataclass
class ModelSelection:
    model_config: ModelConfig
    params: Dict[str, Any]


class ModelHandle:
    """
    Thin wrapper the router returns. Low-level apps call .chat() or .embed()
    and throttling/backup behavior is handled internally.
    """

    def __init__(
        self,
        selection: ModelSelection,
        provider_client: "BaseProviderClient",
        router: "ModelRouter",  # forward ref
    ):
        self.selection = selection
        self.provider_client = provider_client
        self.router = router

    def chat(self, messages: List[Dict[str, Any]]) -> Any:
        if self.selection.model_config.type != "chat":
            raise TypeError("ModelHandle.chat() called on non-chat model")
        return self.router._call_with_retry_and_fallback(
            self.selection, self.provider_client.chat, messages
        )

    def embed(self, text: str) -> Any:
        if self.selection.model_config.type != "embedding":
            raise TypeError("ModelHandle.embed() called on non-embedding model")
        return self.router._call_with_retry_and_fallback(
            self.selection, self.provider_client.embed, text
        )
