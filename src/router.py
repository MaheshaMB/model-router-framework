from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .config_loader import ConfigLoader
from .models import (
    ModelHandle,
    ModelSelection,
    RouterConfig,
    TaskDescriptor,
    FeatureSummary,
    ModelConfig,
)
from .providers import (
    BaseProviderClient,
    BedrockProviderClient,
    AnthropicProviderClient,
    GeminiProviderClient,
)
from .utils.text_analysis import build_feature_summary
from .utils.exceptions import ThrottlingError


class ModelRouter:
    """
    Main entry point:
      handle = router.select_model(text=user_query)
      answer = handle.chat(messages=[...])
    """

    def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
        self._config_loader = config_loader or ConfigLoader()
        self._config: RouterConfig = self._config_loader.get_config()

    # ---------- Public API ----------

    def select_model(
        self,
        text: str,
        task_type: str = "chat",
        context_tokens: Optional[int] = None,
        tenant_id: Optional[str] = None,
        tenant_tier: str = "standard",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelHandle:
        task = TaskDescriptor(
            text=text,
            task_type=task_type,  # "chat" | "embedding"
            context_tokens=context_tokens,
            tenant_id=tenant_id,
            tenant_tier=tenant_tier,  # "free"|"standard"|"premium"|"internal"
            metadata=metadata or {},
        )
        features = build_feature_summary(task)
        selection = self._select_model_for_features(features)
        provider_client = self._create_provider_client(selection.model_config, selection.params)
        return ModelHandle(selection, provider_client, self)

    # ---------- Internal: selection ----------

    def _select_model_for_features(self, features: FeatureSummary) -> ModelSelection:
        # Try rules
        for rule in self._config.rules:
            if self._rule_matches(rule, features):
                model_cfg = self._config.models[rule.use_model]
                params = dict(model_cfg.default_params)
                params.update(rule.override_params)
                # Pre-call context size check
                self._ensure_context_limit(model_cfg, features)
                return ModelSelection(model_config=model_cfg, params=params)

        # Fall back to defaults
        if features.task_type == "chat":
            default_id = self._config.default_chat_model_id
        else:
            default_id = self._config.default_embedding_model_id

        if not default_id:
            raise RuntimeError(f"No default model configured for task_type={features.task_type}")

        model_cfg = self._config.models[default_id]
        self._ensure_context_limit(model_cfg, features)
        return ModelSelection(model_config=model_cfg, params=dict(model_cfg.default_params))

    def _rule_matches(self, rule, features: FeatureSummary) -> bool:
        cond = rule.when
        # task_type
        if cond.task_type and cond.task_type != features.task_type:
            return False
        # complexity
        if cond.complexity and cond.complexity != features.complexity:
            return False
        # language
        if cond.language and cond.language != features.language:
            return False
        # tenant tier
        if cond.tenant_tiers and features.tenant_tier not in cond.tenant_tiers:
            return False
        # context tokens
        if cond.context_tokens_range:
            if cond.context_tokens_range.gte is not None and features.context_tokens < cond.context_tokens_range.gte:
                return False
            if cond.context_tokens_range.lt is not None and features.context_tokens >= cond.context_tokens_range.lt:
                return False
        # chunk tokens (for embedding)
        if cond.chunk_tokens_range:
            if cond.chunk_tokens_range.gte is not None and features.token_count < cond.chunk_tokens_range.gte:
                return False
            if cond.chunk_tokens_range.lt is not None and features.token_count >= cond.chunk_tokens_range.lt:
                return False
        return True

    def _ensure_context_limit(self, model_cfg: ModelConfig, features: FeatureSummary) -> None:
        if features.task_type == "chat" and model_cfg.max_context_tokens:
            if features.context_tokens > model_cfg.max_context_tokens:
                # Here you could auto-summarize, but for now just raise
                raise ValueError(
                    f"Context ({features.context_tokens}) exceeds model max_context_tokens "
                    f"({model_cfg.max_context_tokens}) for model {model_cfg.id}"
                )

        if features.task_type == "embedding" and model_cfg.max_chunk_tokens:
            if features.token_count > model_cfg.max_chunk_tokens:
                raise ValueError(
                    f"Chunk size ({features.token_count}) exceeds model max_chunk_tokens "
                    f"({model_cfg.max_chunk_tokens}) for model {model_cfg.id}"
                )

    # ---------- Internal: provider clients ----------

    def _create_provider_client(
        self, model_cfg: ModelConfig, params: Dict[str, Any]
    ) -> BaseProviderClient:
        if model_cfg.provider == "bedrock":
            return BedrockProviderClient(model_cfg.model_id, params)
        if model_cfg.provider == "anthropic":
            return AnthropicProviderClient(model_cfg.model_id, params)
        if model_cfg.provider == "gemini":
            return GeminiProviderClient(model_cfg.model_id, params)
        raise ValueError(f"Unsupported provider: {model_cfg.provider}")

    # ---------- Internal: throttling + fallback ----------

    def _call_with_retry_and_fallback(
        self,
        selection: ModelSelection,
        func,
        func_arg,
    ) -> Any:
        model_cfg = selection.model_config
        policy = model_cfg.retry_policy

        # 1) Retry primary model
        attempt = 0
        while True:
            attempt += 1
            try:
                return func(func_arg)
            except Exception as e:
                if not self._is_throttling_error(e):
                    raise
                if attempt >= policy.max_attempts:
                    break
                time.sleep(policy.backoff_ms / 1000.0)

        # 2) Fallback to backup model if configured
        if not model_cfg.backup_model_id:
            raise ThrottlingError(
                f"Primary model {model_cfg.id} throttled and no backup_model_id configured"
            )

        backup_cfg = self._config.models.get(model_cfg.backup_model_id)
        if not backup_cfg:
            raise ThrottlingError(
                f"Backup model {model_cfg.backup_model_id} for {model_cfg.id} not found in config"
            )

        # Use same base params, but from backup config defaults
        backup_params = dict(backup_cfg.default_params)
        backup_selection = ModelSelection(model_config=backup_cfg, params=backup_params)
        backup_client = self._create_provider_client(backup_cfg, backup_params)

        attempt = 0
        while True:
            attempt += 1
            try:
                if backup_cfg.type == "chat":
                    return backup_client.chat(func_arg)
                else:
                    return backup_client.embed(func_arg)
            except Exception as e:
                if not self._is_throttling_error(e):
                    raise
                if attempt >= backup_cfg.retry_policy.max_attempts:
                    break
                time.sleep(backup_cfg.retry_policy.backoff_ms / 1000.0)

        raise ThrottlingError(
            f"Primary model {model_cfg.id} and backup {backup_cfg.id} both throttled"
        )

    def _is_throttling_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        throttling_keywords = [
            "throttling",
            "rate limit",
            "too many requests",
            "tokens per minute",
        ]
        return any(k in msg for k in throttling_keywords)
