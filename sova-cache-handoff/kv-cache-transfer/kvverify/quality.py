"""Prospective decision and free-generation metrics, never a rescore of pilot v1.

The decision metric restricts one next-token distribution to option letters. It
does not establish free-generation quality or probability calibration. Helpers
must run outside timed model execution when reporting model-only latency.
"""
import math
import re

SCORING_VERSION = 'decision-and-generation-v2'


def _labels(choices):
    labels = list(choices)
    if not 2 <= len(labels) <= 26 or any(not isinstance(x, str) or len(x) != 1
                                          or x not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for x in labels):
        raise ValueError('Choices require 2-26 distinct uppercase option letters')
    return sorted(labels)


def option_token_ids(tokenizer, choices):
    """Fail closed if an option is not exactly one unique, decodable token."""
    result = {}
    for label in _labels(choices):
        ids = tokenizer(label, add_special_tokens=False)['input_ids']
        if len(ids) != 1 or type(ids[0]) is not int or ids[0] < 0:
            raise ValueError('Each option letter must encode as one token: '+label)
        if tokenizer.decode(ids, skip_special_tokens=False) != label:
            raise ValueError('Option token does not decode exactly: '+label)
        result[label] = ids[0]
    if len(set(result.values())) != len(result):
        raise ValueError('Option letters share a token ID')
    return result


def _logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(v-maximum) for v in values))


def score_option_logits(logits, token_ids, gold):
    """Score unmasked next-token logits; supports 1D lists and Torch tensors.

    The emitted option probabilities sum to one *conditional on choosing an
    option*. Full-vocabulary option mass records how much that conditioning
    discarded. A model preferring prose can get a constrained decision correct
    while failing the separately reported generation-format contract.
    """
    labels = _labels(token_ids)
    if gold not in labels or len(set(token_ids.values())) != len(labels):
        raise ValueError('Invalid gold label or duplicate option token IDs')
    if hasattr(logits, 'detach'):
        vector = logits.detach().float()
        if vector.ndim != 1:
            raise ValueError('Expected one full-vocabulary next-token vector')
        if any(type(i) is not int or not 0 <= i < vector.numel() for i in token_ids.values()):
            raise ValueError('Option token ID outside vocabulary')
        values = [float(vector[token_ids[label]].item()) for label in labels]
        denominator = float(vector.logsumexp(dim=-1).item())
        top = int(vector.argmax().item())
    else:
        vector = list(logits)
        if not vector or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
            raise ValueError('Full-vocabulary logits must be finite')
        if any(type(i) is not int or not 0 <= i < len(vector) for i in token_ids.values()):
            raise ValueError('Option token ID outside vocabulary')
        values = [float(vector[token_ids[label]]) for label in labels]
        denominator = _logsumexp(vector)
        top = max(range(len(vector)), key=vector.__getitem__)
    if not all(math.isfinite(v) for v in [*values, denominator]):
        raise ValueError('Nonfinite option/full-vocabulary distribution')
    normalizer = _logsumexp(values)
    probabilities = {label: math.exp(v-normalizer) for label, v in zip(labels, values)}
    answer = labels[max(range(len(values)), key=values.__getitem__)]
    return {
        'scoring_version': SCORING_VERSION,
        'decision_answer': answer,
        'decision_correct': answer == gold,
        'option_token_ids': dict(token_ids),
        'option_logits': dict(zip(labels, values)),
        'option_probabilities': probabilities,
        'option_log_probabilities': {label: value-normalizer for label, value in zip(labels, values)},
        'option_probability_mass': min(1., math.exp(normalizer-denominator)),
        'full_vocab_top_token_id': top,
        'full_vocab_top_token_is_option': top in set(token_ids.values()),
        'decision_rule': 'argmax over single next-token option logits; conditioned probabilities are not calibrated confidence',
    }


def generation_metrics(text, gold, choices, finish_reason='unknown'):
    """Keep exact-letter correctness separate from permissive diagnostics.

    finish_reason is eos, length, or unknown; a length-capped answer is flagged
    even if its decoded text happens to be exactly one valid letter.
    """
    labels = _labels(choices)
    if gold not in labels or not isinstance(text, str) or finish_reason not in ('eos', 'length', 'unknown'):
        raise ValueError('Invalid generation scoring input')
    answer = text.strip()
    match = re.match(r'^([A-Z])(?:[.)\s:]|$)', answer)
    prefix = match.group(1) if match and match.group(1) in labels else None
    valid = answer in labels
    return {
        'scoring_version': SCORING_VERSION,
        'free_generation_strict_correct': valid and answer == gold,
        'free_generation_format_valid': valid,
        'free_generation_invalid_format': not valid,
        'free_generation_prefix_diagnostic': prefix,
        'free_generation_prefix_diagnostic_correct': prefix == gold,
        'free_generation_finish_reason': finish_reason,
        'free_generation_truncated': finish_reason == 'length',
    }
