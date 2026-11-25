"""Distill the operational question readout, with no task gold in the loss."""


def question_readout_loss(runtime, mapper, full_ids, prefix_ids, suffix_ids,
                          *, native_prefix_tokens=0):
    """Match the native target's entire first-answer distribution.

    The document prefix is transferred; the exact fresh question/template suffix
    is consumed by the frozen target. Optional native initial KV entries are an
    explicit partial-recomputation policy, never a pure-transfer claim.
    """
    import torch
    from torch.nn import functional as F
    from transformers import DynamicCache
    from .train_mapper import model_forward
    from .hf_backend import legacy
    from .native_prefix import replace_prefix

    if not prefix_ids or not suffix_ids or full_ids != prefix_ids + suffix_ids:
        raise ValueError('Exact nonempty document/question handoff required')
    if type(native_prefix_tokens) is not int or not 0 <= native_prefix_tokens < len(prefix_ids):
        raise ValueError('Native prefix must leave translated document tokens')
    with torch.no_grad():
        native = model_forward(runtime.target, full_ids, logits_to_keep=1)
        teacher = native.logits[0, -1:].detach().float().clone()
        native_prefix = None
        if native_prefix_tokens:
            native_prefix = DynamicCache.from_legacy_cache(tuple(
                tuple(t[:, :, :native_prefix_tokens].detach().clone() for t in row)
                for row in legacy(native.past_key_values)))
        del native
        source = model_forward(runtime.source, prefix_ids, logits_to_keep=1).past_key_values
    cache = mapper.apply_checkpointed(source, runtime.source, runtime.target, token_block_size=256)
    if native_prefix_tokens:
        cache = replace_prefix(cache, native_prefix, native_prefix_tokens)
    prediction = model_forward(runtime.target, suffix_ids, cache, logits_to_keep=1)
    logits = prediction.logits[0, -1:].float()
    if logits.shape != teacher.shape:
        raise ValueError('Native/student first-answer vocabulary shapes differ')
    loss = F.kl_div(F.log_softmax(logits, -1), F.softmax(teacher, -1), reduction='batchmean')
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite question-readout distillation')
    return loss, {'loss': float(loss.detach()), 'supervised_tokens': 1,
                  'mapped_prefix_tokens': len(prefix_ids) - native_prefix_tokens,
                  'native_prefix_tokens': native_prefix_tokens,
                  'native_suffix_tokens': len(suffix_ids), 'hard_ce_weight': 0.,
                  'gold_qa_used': False, 'objective': 'question_first_answer_full_vocab_KL'}
