from __future__ import annotations

from typing import Literal

from ..models import FeatureSummary, TaskDescriptor, TenantTier


def estimate_tokens(text: str) -> int:
    # Very simple heuristic: ~4 chars per token
    return max(1, len(text) // 4)


def classify_size(token_count: int) -> str:
    if token_count < 256:
        return "small"
    if token_count < 2048:
        return "medium"
    return "large"


def detect_language(text: str) -> str:
    # Very simple heuristic: if mostly ASCII, assume "en", else "multi"
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    ratio = ascii_chars / max(1, len(text))
    return "en" if ratio > 0.8 else "multi"


def estimate_complexity(text: str) -> str:
    length = len(text)
    lowercase = text.lower()
    if "explain in detail" in lowercase or "architecture" in lowercase:
        return "high"
    if length > 1000:
        return "high"
    if length > 300:
        return "medium"
    return "low"


def build_feature_summary(task: TaskDescriptor) -> FeatureSummary:
    tokens = estimate_tokens(task.text)
    size_class = classify_size(tokens)
    language = detect_language(task.text)
    complexity = estimate_complexity(task.text)
    context_tokens = task.context_tokens or 0
    tenant_tier: TenantTier = task.tenant_tier
    return FeatureSummary(
        task_type=task.task_type,
        token_count=tokens,
        size_class=size_class,
        language=language,
        complexity=complexity,
        context_tokens=context_tokens,
        tenant_tier=tenant_tier,
    )
