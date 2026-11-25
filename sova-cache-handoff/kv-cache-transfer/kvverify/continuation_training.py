"""Frozen-receiver soft distillation on unlabeled optimization continuations."""
import random


class ContinuationSampler:
    def __init__(self,documents,seed=19,tail_tokens=32):
        if not documents or any(len(ids)<tail_tokens+8 for ids in documents.values()):raise ValueError('Long enough unlabeled documents required')
        self.documents=documents;self.rng=random.Random(seed);self.order=[];self.tail_tokens=tail_tokens
    def next(self):
        if not self.order:
            self.order=sorted(self.documents);self.rng.shuffle(self.order)
        document=self.order.pop();ids=self.documents[document]
        prefix=min(self.rng.choice([512,1024,1024,1024,2048,2048,4096]),len(ids)-self.tail_tokens)
        start=self.rng.randrange(len(ids)-prefix-self.tail_tokens+1)
        return document,ids[start:start+prefix],ids[start+prefix:start+prefix+self.tail_tokens],start


def continuation_loss(runtime,mapper,prefix_ids,continuation_ids):
    """Reserve the final prefix token as a native bridge, score every tail token.

    Labels never enter the loss. Native full-context distributions supervise the
    same causal positions as the mapped prefix plus native bridge/continuation.
    """
    import torch
    from torch.nn import functional as F
    from .train_mapper import model_forward
    if len(prefix_ids)<2 or len(continuation_ids)<1:raise ValueError('Nonempty causal bridge/tail required')
    count=len(continuation_ids)
    with torch.no_grad():
        native=model_forward(runtime.target,prefix_ids+continuation_ids[:-1],logits_to_keep=count)
        teacher=native.logits[0].detach().float();del native
        source=model_forward(runtime.source,prefix_ids[:-1],logits_to_keep=1).past_key_values
    cache=mapper.apply_checkpointed(source,runtime.source,runtime.target,token_block_size=256)
    prediction=model_forward(runtime.target,[prefix_ids[-1]]+continuation_ids[:-1],cache,logits_to_keep=count)
    logits=prediction.logits[0].float()
    if teacher.shape!=logits.shape or teacher.shape[0]!=count:raise ValueError('Teacher/student causal positions differ')
    loss=F.kl_div(F.log_softmax(logits,-1),F.softmax(teacher,-1),reduction='batchmean')
    if not torch.isfinite(loss):raise ValueError('Nonfinite continuation distillation')
    metrics={'loss':float(loss.detach()),'supervised_tokens':count,'mapped_prefix_tokens':len(prefix_ids)-1,
        'native_suffix_tokens':count,'hard_ce_weight':0.,'gold_qa_used':False}
    return loss,metrics
