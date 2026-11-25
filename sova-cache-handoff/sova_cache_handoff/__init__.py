"""Native KV prefix handoff for engines with explicit cache access."""

from .cache_handoff import CacheScope, NativePrefixRegistry
from .handoff import CacheHandoff, HandoffResult

__all__ = ["CacheHandoff", "CacheScope", "HandoffResult", "NativePrefixRegistry"]
