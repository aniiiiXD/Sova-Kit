# SOVA Cache Handoff

For vLLM export, mapper preparation, and receiver setup, see the [vLLM handoff guide](VLLM_HANDOFF.md).

Cache Handoff is an in-process Python module for agent workers sharing a model. An agent system can register an exact token prefix from chat history, tool results, source files, or other shared context. Each worker gets an independent cache branch and adds its own suffix.

## Install and try it

Use Python 3.10 or newer. From this folder, run:

```sh
python -m pip install .
kvverify handoff-smoke
```

For a wheel, install it with `python -m pip install ./sova_kvverify-0.1.0-py3-none-any.whl`. The smoke demo runs offline with no runtime dependencies, model weights, API credentials, or GPU. It reports cold, partial, and full cache hits, branch isolation, and tenant isolation.

For a tiny Qwen3 CPU test, install `.[hf]` on a Python version supported by PyTorch, then run `kvverify handoff-smoke --hf`. This uses random weights and checks that cached paths reproduce cold-path logits.

## Connect a model engine

Pass an engine callback that can accept and return writable KV state. Use a scope derived from your authenticated request and from the exact model, tokenizer and cache configuration running in your engine.

```python
from kvverify.cache_handoff import CacheScope, NativePrefixRegistry
from kvverify.handoff import CacheHandoff

registry = NativePrefixRegistry(engine.clone_cache, max_bytes=64 * 1024 * 1024,
                                ttl_seconds=300)
handoff = CacheHandoff(registry, engine.advance, engine.cache_bytes)
scope = CacheScope(tenant_id, checkpoint_id, tokenizer_id, cache_layout_id)

handoff.prime(scope, shared_tokens)
result = handoff.run(scope, shared_tokens, worker_tokens)
```

- `engine.advance(scope, tokens, cache)` returns `(output, updated_cache)`. It starts from position zero when `cache` is `None`. The updated cache provides `get_seq_length()`.
- `engine.clone_cache(state)` returns an independent writable copy. `engine.cache_bytes(state)` reports the bytes retained in the registry.
- `remember(scope, prefix, state)` registers native KV state an engine call already produced, so that the call can also serve as the first worker's prefill.
- `run` appends a nonempty suffix. It processes the full input on a miss, or processes the uncached tail on a partial hit. It returns the engine's output plus reused and processed token counts. Engine errors propagate to the caller.

This preview supports length-preserving native caches with positions starting at zero. Render and tokenize the full prompt before choosing the split. Include checkpoint revision, dtype and other cache compatibility settings in the scope. The registry is process-local. For separate vLLM workers on the same host, see the [connector setup and validation guide](VLLM_HANDOFF.md).
