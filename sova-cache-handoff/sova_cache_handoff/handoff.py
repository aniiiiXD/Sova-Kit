"""In-process same-model KV handoff for an engine with explicit cache access."""
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .cache_handoff import CacheScope, NativePrefixRegistry, _token_ids


Output = TypeVar("Output")


@dataclass(frozen=True)
class HandoffResult(Generic[Output]):
    output: Output
    reused_tokens: int
    processed_tokens: int


class CacheHandoff(Generic[Output]):
    """Reuse cached prefixes while giving each request its own writable state.

    ``advance(scope, tokens, cache)`` returns ``(output, updated_cache)``. A
    ``None`` cache starts at position zero. The cache has ``get_seq_length()``.
    The engine adapter is responsible for serving the checkpoint in the scope.
    """

    def __init__(
        self,
        registry: NativePrefixRegistry,
        advance: Callable[[CacheScope, tuple[int, ...], object | None], tuple[Output, object]],
        state_bytes: Callable[[object], int],
    ):
        self.registry = registry
        self._advance = advance
        self._state_bytes = state_bytes

    def remember(self, scope: CacheScope, prefix, state: object) -> bool:
        """Register an engine-produced native cache for these exact tokens."""
        return self.registry.store_native(scope, prefix, state, byte_count=self._state_bytes(state))

    def prime(self, scope: CacheScope, prefix) -> bool:
        """Populate a shared prefix once and return whether it is retained."""
        tokens = _token_ids(prefix)
        if not tokens:
            raise ValueError("Shared cache prefix must be nonempty")
        hit = self.registry.longest_native(scope, tokens)
        if hit.token_count == len(tokens):
            return True
        _, state = self._advance(scope, tokens[hit.token_count:], hit.state)
        return self.remember(scope, tokens, state)

    def run(self, scope: CacheScope, prefix, suffix) -> HandoffResult[Output]:
        """Run a private suffix, processing the full input on a cache miss."""
        shared = _token_ids(prefix)
        private = _token_ids(suffix)
        if not private:
            raise ValueError("A request needs at least one suffix token")
        hit = self.registry.longest_native(scope, shared)
        remaining = shared[hit.token_count:] + private
        output, _ = self._advance(scope, remaining, hit.state)
        return HandoffResult(output, hit.token_count, len(remaining))
