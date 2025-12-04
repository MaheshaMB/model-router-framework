from __future__ import annotations

import json
import os
from typing import Dict, Any, List

import boto3

from .models import (
    ModelConfig,
    RouterConfig,
    RoutingRule,
    RuleCondition,
    RuleConditionRange,
    RetryPolicy,
)


class ConfigLoader:
    """
    Loads models.json and routing_rules.json from S3 and parses into RouterConfig.
    Caches in memory for Lambda warm invocations.
    """

    def __init__(
        self,
        bucket: str | None = None,
        models_key: str | None = None,
        rules_key: str | None = None,
    ) -> None:
        # Decide source: S3 vs local filesystem
        self._fetch_from_s3 = os.environ.get("FETCH_DRIVE", "false").lower() == "true"

        # S3 configuration
        self.bucket = bucket or os.environ.get("MODEL_ROUTER_CONFIG_BUCKET")
        self.models_key = models_key or os.environ.get("MODEL_ROUTER_MODELS_KEY", "models.json")
        self.rules_key = rules_key or os.environ.get("MODEL_ROUTER_RULES_KEY", "routing_rules.json")

        # Local file configuration (project root / current working dir)
        self.models_path = os.environ.get("MODEL_ROUTER_MODELS_PATH", "models.json")
        self.rules_path = os.environ.get("MODEL_ROUTER_RULES_PATH", "routing_rules.json")

        if self._fetch_from_s3 and not self.bucket:
            raise ValueError("MODEL_ROUTER_CONFIG_BUCKET is required when FETCH_DRIVE=true")

        self._s3 = boto3.client("s3")
        self._cache: RouterConfig | None = None


    def get_config(self, force_reload: bool = False) -> RouterConfig:
        if self._cache is not None and not force_reload:
            return self._cache

        # Load config from S3 or local filesystem
        if self._fetch_from_s3:
            # Load models.json from S3
            models_resp = self._s3.get_object(Bucket=self.bucket, Key=self.models_key)
            models_data = json.loads(models_resp["Body"].read())

            # Load routing_rules.json from S3
            rules_resp = self._s3.get_object(Bucket=self.bucket, Key=self.rules_key)
            rules_data = json.loads(rules_resp["Body"].read())
        else:
            # Load models.json from local path
            with open(self.models_path, "r", encoding="utf-8") as f:
                models_data = json.load(f)

            # Load routing_rules.json from local path
            with open(self.rules_path, "r", encoding="utf-8") as f:
                rules_data = json.load(f)

        models_dict: Dict[str, ModelConfig] = {}
        for m in models_data.get("models", []):
            retry_raw = m.get("retry_policy", {})
            retry = RetryPolicy(
                max_attempts=retry_raw.get("max_attempts", 2),
                backoff_ms=retry_raw.get("backoff_ms", 200),
            )
            cfg = ModelConfig(
                id=m["id"],
                provider=m["provider"],
                type=m["type"],
                model_id=m["model_id"],
                max_context_tokens=m.get("max_context_tokens"),
                max_chunk_tokens=m.get("max_chunk_tokens"),
                languages=m.get("languages"),
                strengths=m.get("strengths"),
                cost_tier=m.get("cost_tier", "medium"),
                default_params=m.get("default_params", {}),
                backup_model_id=m.get("backup_model_id"),
                retry_policy=retry,
            )
            models_dict[cfg.id] = cfg

        rules: List[RoutingRule] = []
        for r in rules_data.get("rules", []):
            cond_raw = r.get("when", {})
            ctx_range = cond_raw.get("context_tokens")
            chunk_range = cond_raw.get("chunk_tokens")

            condition = RuleCondition(
                task_type=cond_raw.get("task_type"),
                complexity=cond_raw.get("complexity"),
                language=cond_raw.get("language"),
                tenant_tiers=cond_raw.get("tenant_tier"),
                context_tokens_range=RuleConditionRange(
                    gte=ctx_range.get("gte"), lt=ctx_range.get("lt")
                )
                if ctx_range
                else None,
                chunk_tokens_range=RuleConditionRange(
                    gte=chunk_range.get("gte"), lt=chunk_range.get("lt")
                )
                if chunk_range
                else None,
            )

            rule = RoutingRule(
                id=r["id"],
                when=condition,
                use_model=r["use_model"],
                override_params=r.get("override_params", {}),
            )
            rules.append(rule)

        defaults = rules_data.get("defaults", {})
        cfg = RouterConfig(
            models=models_dict,
            rules=rules,
            default_chat_model_id=defaults.get("chat"),
            default_embedding_model_id=defaults.get("embedding"),
        )
        self._cache = cfg
        return cfg
