"""Select small native evidence excerpts from a history available as text."""

from collections import Counter
from dataclasses import dataclass, replace
from math import log
import re

_WORD = re.compile(r"[^\W_]+")
_STOP = frozenset(("the", "are", "and", "with", "from", "was", "for", "into", "who", "which", "what",
                   "were", "when", "that", "this", "their", "these", "than", "them", "will", "have",
                   "does", "would", "about", "according", "following", "document", "answer", "passage"))


def terms(text):
    return [word for word in _WORD.findall(text.casefold()) if len(word) > 2 and word not in _STOP]


def _balanced_windows(windows, score, question_terms, option_terms, max_windows):
    """Assign the strongest distinct evidence window to each option in turn."""
    rankings = {}
    for label, words in option_terms.items():
        # Normalize each query so a long distractor has no automatic advantage.
        query = {word: 1 / max(1, len(words)) for word in words}
        for word in question_terms:
            query[word] = query.get(word, 0.) + .5 / max(1, len(question_terms))
        rankings[label] = sorted(((score(window, query), window[:4]) for window in windows),
                                 key=lambda item: (-item[0], item[1][0]))

    chosen = []
    while rankings and len(chosen) < max_windows:
        eligible = [(value, span, label) for label, ranking in rankings.items()
                    for value, span in ranking
                    if all(span[1] <= old[0] or span[0] >= old[1] for old in chosen)]
        if not eligible:
            break
        _, span, label = min(eligible, key=lambda item: (-item[0], item[1][0], item[2]))
        chosen.append(span)
        del rankings[label]
    return chosen


def select_excerpts(tokenizer, context, question, choices, *, window_tokens=224, max_windows=4, balanced=False):
    """Query/option lexical search over indexed source text; no answer labels."""
    if window_tokens < 4 or max_windows < 1:
        raise ValueError("Invalid evidence budget")
    offsets = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    windows = []
    stride = 3 * window_tokens // 4
    for start in range(0, len(offsets), stride):
        stop = min(start + window_tokens, len(offsets))
        left, right = offsets[start][0], offsets[stop - 1][1]
        if left < right:
            windows.append((start, stop, left, right, Counter(terms(context[left:right]))))
        if stop == len(offsets):
            break
    if not windows:
        raise ValueError("Cannot retrieve evidence from empty context")
    frequencies = Counter(word for *_, counts in windows for word in counts)
    length_average = max(sum(sum(counts.values()) for *_, counts in windows) / len(windows), 1)
    query = {word: 1.0 for value in choices.values() for word in terms(value)}
    query.update({word: 3.0 for word in terms(question)})

    def score(window, weights=query):
        counts = window[-1]
        denominator = 1.2 * (.25 + .75 * sum(counts.values()) / length_average)
        return sum(weight * log(1 + (len(windows) - frequencies[word] + .5) / (frequencies[word] + .5))
                   * (2.2 * counts[word]) / (counts[word] + denominator)
                   for word, weight in weights.items() if counts[word])
    chosen = []
    if balanced:
        chosen = _balanced_windows(windows, score, sorted(set(terms(question))),
                                   {label: sorted(set(terms(text))) for label, text in choices.items()}, max_windows)
    if len(chosen) < max_windows:
        for window in sorted(windows, key=lambda item: (-score(item), item[0])):
            start, stop, left, right, _ = window
            if all(stop <= other[0] or start >= other[1] for other in chosen):
                chosen.append((start, stop, left, right))
            if len(chosen) == max_windows:
                break

    chunks = tuple(context[left:right] for _, _, left, right in sorted(chosen))
    return chunks, tuple((start, stop) for start, stop, _, _ in sorted(chosen))


@dataclass(frozen=True)
class EvidenceReplay:
    short_ids: list
    cache_tokens: int
    augmented_suffix: list
    source_spans: tuple


def prepare_evidence_replay(runtime, example, original_prefix, *, max_prompt_tokens=32768, balanced=False):
    chunks, spans = select_excerpts(runtime.tokenizer, example.context, example.question, example.choices,
                                    balanced=balanced)
    quoted = "\n\n[… omitted …]\n\n".join(chunks)
    short = replace(example, context=quoted)
    augmented = replace(example, context=example.context + "\n\nSelected excerpts from this document:\n" + quoted)
    short_ids = list(runtime.tokenize(short))
    augmented_ids = list(runtime.tokenize(augmented))
    count = next((index for index, (left, right) in enumerate(zip(original_prefix, augmented_ids))
                  if left != right), min(len(original_prefix), len(augmented_ids)))
    if (len(short_ids) >= len(original_prefix) or not 256 < count < len(augmented_ids)
            or len(augmented_ids) > max_prompt_tokens or len(original_prefix) - count > 8):
        raise ValueError("Evidence prompt exceeds its budget or does not share the exact source cache prefix")
    return EvidenceReplay(short_ids, count, augmented_ids[count:], spans)
