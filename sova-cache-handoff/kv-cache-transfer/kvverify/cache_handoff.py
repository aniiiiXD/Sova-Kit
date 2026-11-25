"""Process-local immutable storage for native evidence prefixes."""
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from time import monotonic


def _token_ids(values):
    tokens = tuple(values)
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Cache prefix must contain nonnegative integer token IDs")
    return tokens


@dataclass(frozen=True)
class CacheScope:
    tenant: str
    model: str
    tokenizer: str
    layout: str

    def __post_init__(self):
        if not all((self.tenant, self.model, self.tokenizer, self.layout)):
            raise ValueError("Cache identity fields are required")


@dataclass(frozen=True)
class CacheHit:
    token_count: int
    state: object


@dataclass(frozen=True)
class _Entry:
    state: object
    byte_count: int
    expires_at: float


class NativePrefixRegistry:
    """An LRU of native caches for contiguous, zero-origin prompt positions.

    The supplied clone function must return independent writable cache state.
    Translated caches are ineligible for this registry by design.
    """

    def __init__(self, clone, *, max_bytes, ttl_seconds, clock=monotonic):
        if max_bytes <= 0 or ttl_seconds <= 0:
            raise ValueError("Positive cache capacity and TTL are required")
        self._clone = clone
        self._max_bytes = max_bytes
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = RLock()

    def _drop(self, key):
        self._bytes -= self._entries.pop(key).byte_count

    def _expire(self, now):
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            self._drop(key)

    def store_native(self, scope, tokens, state, *, byte_count):
        tokens = _token_ids(tokens)
        if not tokens:
            raise ValueError("Stored cache prefix must be nonempty")
        if state.get_seq_length() != len(tokens):
            raise ValueError("Cache length and prefix token count differ")
        if type(byte_count) is not int or byte_count < 1:
            raise ValueError("Positive cache byte count required")
        if byte_count > self._max_bytes:
            return False
        key = (scope, tokens)
        with self._lock:
            copied = self._clone(state)
            now = self._clock()
            self._expire(now)
            if key in self._entries:
                self._drop(key)
            while self._bytes + byte_count > self._max_bytes:
                self._drop(next(iter(self._entries)))
            self._entries[key] = _Entry(copied, byte_count, now + self._ttl)
            self._bytes += byte_count
        return True

    def longest_native(self, scope, evidence_tokens):
        tokens = _token_ids(evidence_tokens)
        if not tokens:
            return CacheHit(0, None)
        with self._lock:
            self._expire(self._clock())
            key = max((key for key in self._entries if key[0] == scope
                       and tokens[:len(key[1])] == key[1]), key=lambda key: len(key[1]),
                      default=None)
            if key is None:
                return CacheHit(0, None)
            self._entries.move_to_end(key)
            return CacheHit(len(key[1]), self._clone(self._entries[key].state))
