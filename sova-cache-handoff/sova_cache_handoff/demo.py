"""Exercise the public handoff API with synthetic cache state."""
from dataclasses import dataclass
import json

from .cache_handoff import CacheScope, NativePrefixRegistry
from .handoff import CacheHandoff


@dataclass
class FixtureCache:
    tokens: list[int]

    def get_seq_length(self):
        return len(self.tokens)


def run():
    def advance(scope, tokens, cache):
        if cache is None:
            cache = FixtureCache([])
        cache.tokens.extend(tokens)
        return tuple(cache.tokens), cache

    registry = NativePrefixRegistry(lambda state: FixtureCache(state.tokens.copy()),
                                    max_bytes=4096, ttl_seconds=60)
    handoff = CacheHandoff(registry, advance, lambda state: len(state.tokens))
    scope = CacheScope("team-a", "fixture-v1", "byte-tokenizer-v1", "fixture-cache-v1")
    other = CacheScope("team-b", scope.model, scope.tokenizer, scope.layout)
    prefix = b"Chat: deploy failed\nTool result: config check passed\n"
    suffix_a = b"Agent A: find the error"
    suffix_b = b"Agent B: write a status update"

    cold = handoff.run(scope, prefix, suffix_a)
    handoff.prime(scope, prefix[:20])
    partial = handoff.run(scope, prefix, suffix_a)
    handoff.prime(scope, prefix)
    warm = handoff.run(scope, prefix, suffix_a)
    branch = handoff.run(scope, prefix, suffix_b)
    repeated = handoff.run(scope, prefix, suffix_a)
    isolated = handoff.run(other, prefix, suffix_a)

    checks = {
        "cold_fallback": cold.reused_tokens == 0,
        "partial_hit": partial.reused_tokens == 20 and partial.output == cold.output,
        "full_hit": warm.reused_tokens == len(prefix) and warm.output == cold.output,
        "independent_branches": branch.output == tuple(prefix + suffix_b) and repeated == warm,
        "tenant_isolation": isolated.reused_tokens == 0 and isolated.output == cold.output,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Cache handoff smoke failed: {checks}")
    result = {"checks": checks, "shared_tokens": len(prefix),
              "cold_processed_tokens": cold.processed_tokens,
              "warm_processed_tokens": warm.processed_tokens}
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
