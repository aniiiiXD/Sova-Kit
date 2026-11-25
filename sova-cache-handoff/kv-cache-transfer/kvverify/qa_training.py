"""Task-aligned cache training with full documents and fresh native questions.

Historical window training is unchanged. This protocol transfers only the
document-prefix token IDs, then lets the frozen target read the remaining
question/template tokens and teacher-forces its own generated answer tokens.
No gold label is used in the training loss. Torch imports are execution-local
so the shared prompt-boundary helper can be used in preparation and tests.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

from .data import assert_disjoint, check_heldout, json_digest
from .hf_backend import legacy


def split_document_prompt(runtime, example):
    """Return (full_ids, prefix_ids, suffix_ids) from one exact tokenization.

    Fast tokenizer offsets identify the last token entirely within the document
    prefix. A token straddling the document boundary stays in the native suffix,
    avoiding a different BPE merge from tokenizing the two strings separately.
    prefix+suffix must equal runtime.tokenize(example) exactly. No text is cut.
    """
    tokenizer = runtime.tokenizer
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('Document handoff requires fast-tokenizer character offsets')
    content = example.prompt()
    document_prefix = 'Document:\n'+example.context
    if not content.startswith(document_prefix):
        raise ValueError('Unknown Example prompt layout; cannot locate document boundary')
    text = tokenizer.apply_chat_template([{'role':'user','content':content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    location = text.find(content)
    if location < 0 or text.find(content, location+1) >= 0:
        raise ValueError('Chat template must contain the exact user message once')
    boundary = location+len(document_prefix)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded['input_ids'])
    offsets = encoded.get('offset_mapping')
    if offsets is None or len(offsets) != len(ids):
        raise ValueError('Missing or mismatched tokenizer offsets')
    cut = 0
    prior_end = 0
    for index, pair in enumerate(offsets):
        start, end = pair
        if not 0 <= start <= end <= len(text) or end < prior_end:
            raise ValueError('Invalid or unordered tokenizer offsets')
        prior_end = end
        if end <= boundary and cut == index:
            cut += 1
    if not 0 < cut < len(ids):
        raise ValueError('Document prefix and native question suffix must be nonempty')
    prefix, suffix = ids[:cut], ids[cut:]
    expected = list(runtime.tokenize(example))
    if prefix+suffix != expected:
        raise ValueError('Handoff token IDs differ from the original exact prompt')
    return ids, prefix, suffix


def group_questions(examples):
    """Deduplicate document storage, never discard distinct question records."""
    groups, contexts, identifiers = defaultdict(list), {}, set()
    for example in examples:
        if example.id in identifiers:
            raise ValueError('Duplicate question ID in training pool')
        identifiers.add(example.id)
        if example.document_id in contexts and contexts[example.document_id] != example.context_hash:
            raise ValueError('A document ID refers to different contexts')
        contexts[example.document_id] = example.context_hash
        groups[example.document_id].append(example)
    return {doc: sorted(rows, key=lambda e:e.id) for doc, rows in sorted(groups.items())}


class DocumentQuestionSampler:
    """Cycle a seeded document permutation and each document's question pool."""
    def __init__(self, examples, seed=19):
        self.groups = group_questions(examples)
        if not self.groups:
            raise ValueError('Question sampler requires nonempty documents')
        self.rng = random.Random(seed)
        self.documents = list(self.groups)
        self.rng.shuffle(self.documents)
        self.position = 0
        self.document_cycles = 0
        self.question_positions = {doc:0 for doc in self.groups}
        self.question_cycles = {doc:0 for doc in self.groups}
        for questions in self.groups.values():
            self.rng.shuffle(questions)

    def next(self):
        if self.position == len(self.documents):
            self.rng.shuffle(self.documents)
            self.position = 0
            self.document_cycles += 1
        document = self.documents[self.position]
        self.position += 1
        questions = self.groups[document]
        if self.question_positions[document] == len(questions):
            self.rng.shuffle(questions)
            self.question_positions[document] = 0
            self.question_cycles[document] += 1
        index = self.question_positions[document]
        self.question_positions[document] += 1
        return questions[index]


def select_mapper_validation(examples, seed=19):
    """Freeze shared-hypothesis coverage, or one question per unique article.

    ContractNLI-like repeated question types (at most32) use one of each, spread
    across documents. Open QA with many unique questions uses one per document.
    This selection is made before any model outputs and stays fixed throughout
    checkpoint selection, rather than changing the validation set at each step.
    """
    groups = group_questions(examples)
    signatures = defaultdict(list)
    for example in examples:
        signatures[' '.join(example.question.split())].append(example)
    rng = random.Random(seed)
    selected = []
    if len(signatures) <= 32:
        used = Counter()
        for signature, candidates in sorted(signatures.items()):
            candidates = sorted(candidates, key=lambda e:e.id)
            rng.shuffle(candidates)
            candidate = min(candidates, key=lambda e:used[e.document_id])
            selected.append(candidate)
            used[candidate.document_id] += 1
        policy = 'One example per shared question/hypothesis type, up to32types; balance document coverage'
    else:
        for document, candidates in groups.items():
            selected.append(rng.choice(candidates))
        policy = 'One frozen seeded question per held-out article; all questions remain eligible in the data manifest'
    return selected, policy


def checkpointed_mapper_cache(mapper, source_cache, source_model, target_model, enabled=True):
    """Checkpoint each target-layer/token block, never the whole mapper graph."""
    import torch
    from transformers import DynamicCache
    from torch.utils.checkpoint import checkpoint
    if enabled and torch.is_grad_enabled() and hasattr(mapper,'apply_checkpointed'):
        return mapper.apply_checkpointed(source_cache,source_model,target_model,token_block_size=256)
    rows = legacy(source_cache)
    with torch.inference_mode(False):
        flat = tuple(t.detach().clone() if torch.is_inference(t) else t.detach()
                     for row in rows for t in row)
    def convert(*tensors):
        cache = DynamicCache.from_legacy_cache(tuple(zip(tensors[0::2],tensors[1::2])))
        mapped = mapper.apply(cache,source_model,target_model)
        return tuple(t for row in legacy(mapped) for t in row)
    if enabled and torch.is_grad_enabled() and any(p.requires_grad for p in mapper.parameters()):
        result = checkpoint(convert,*flat,use_reentrant=False)
    else:
        result = convert(*flat)
    return DynamicCache.from_legacy_cache(tuple(zip(result[0::2],result[1::2])))


def generate_teacher_trace(runtime, full_ids, prefix_tokens, answer_tokens=8, keep_prefix=False):
    """Greedy native-target answer trace; detached logits supervise these tokens."""
    import torch
    from .train_mapper import model_forward
    if answer_tokens < 1 or not 0 < prefix_tokens < len(full_ids):
        raise ValueError('Invalid native teacher trace boundary or answer budget')
    teacher_logits, generated = [], []
    eos = runtime.target.config.eos_token_id
    eos = set(eos if isinstance(eos,list) else [eos])
    target_prefix = None
    with torch.no_grad():
        out = model_forward(runtime.target,full_ids,logits_to_keep=1)
        if keep_prefix:
            target_prefix = tuple((k[:,:,:prefix_tokens].detach(),v[:,:,:prefix_tokens].detach())
                                  for k,v in legacy(out.past_key_values))
        cache = out.past_key_values
        for index in range(answer_tokens):
            logits = out.logits[0,-1].detach().float().clone()
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite native teacher logits')
            token = int(logits.argmax())
            teacher_logits.append(logits)
            generated.append(token)
            if token in eos:
                break
            if index+1 < answer_tokens:
                out = model_forward(runtime.target,[token],cache,logits_to_keep=1)
                cache = out.past_key_values
    return {'token_ids':generated,'logits':torch.stack(teacher_logits),
            'target_prefix':target_prefix,'finish_reason':'eos' if generated[-1] in eos else 'length'}


def qa_training_loss(runtime,mapper,full_ids,prefix_ids,suffix_ids,*,answer_tokens=8,
                     ce_weight=1.,kl_weight=1.,kv_weight=0.,checkpoint_mapper=True):
    """Teacher-answer CE+KL after exact full-context document/query handoff."""
    import torch
    from torch.nn import functional as F
    from .train_mapper import model_forward
    if full_ids != prefix_ids+suffix_ids or not prefix_ids or not suffix_ids:
        raise ValueError('Student/teacher document-question token boundaries differ')
    if any(not math.isfinite(w) or w < 0 for w in (ce_weight,kl_weight,kv_weight)) or ce_weight+kl_weight+kv_weight <= 0:
        raise ValueError('Positive finite objective weights required')
    teacher = generate_teacher_trace(runtime,full_ids,len(prefix_ids),answer_tokens,keep_prefix=kv_weight>0)
    with torch.no_grad():
        source = model_forward(runtime.source,prefix_ids,logits_to_keep=1).past_key_values
    mapped = checkpointed_mapper_cache(mapper,source,runtime.source,runtime.target,checkpoint_mapper)
    reconstruction = torch.zeros((),device=teacher['logits'].device)
    if kv_weight:
        for predicted, expected in zip(legacy(mapped),teacher['target_prefix']):
            for value, target in zip(predicted,expected):
                target = target.float()
                scale = target.square().mean(dim=(-2,-1),keepdim=True).clamp_min(1e-6)
                reconstruction = reconstruction+((value.float()-target).square()/scale).mean()
        reconstruction = reconstruction/(2*len(teacher['target_prefix']))
    tokens = teacher['token_ids']
    # The last suffix token predicts answer[0]; each teacher answer token then
    # predicts the next. Native teacher and mapped student share every suffix ID.
    out = model_forward(runtime.target,suffix_ids+tokens[:-1],mapped,logits_to_keep=len(tokens))
    logits = out.logits[0].float()
    labels = torch.tensor(tokens,dtype=torch.long,device=logits.device)
    ce = F.cross_entropy(logits,labels)
    kl = F.kl_div(F.log_softmax(logits,dim=-1),F.softmax(teacher['logits'],dim=-1),reduction='batchmean')
    loss = ce_weight*ce+kl_weight*kl+kv_weight*reconstruction
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite task-aligned mapper objective')
    metrics = {'loss':float(loss.detach()),'ce_loss':float(ce.detach()),'kl_loss':float(kl.detach()),
        'kv_loss':float(reconstruction.detach()),'transferred_tokens':len(prefix_ids),
        'native_query_tokens':len(suffix_ids),'full_prompt_tokens':len(full_ids),
        'teacher_answer_tokens':len(tokens),'teacher_token_ids':tokens,
        'teacher_text':runtime.tokenizer.decode(tokens,skip_special_tokens=True),
        'teacher_finish_reason':teacher['finish_reason'],'gold_answer_used':False}
    return loss,metrics


def train_qa_mapper(runtime,training_examples,validation_examples,output,*,steps=400,
                    max_prompt_tokens=8192,answer_tokens=8,rank=16,lr=3e-4,
                    ce_weight=1.,kl_weight=1.,kv_weight=0.,checkpoint_every=50,
                    callback=None,max_seconds=1800,seed=19,checkpoint_mapper=True,
                    memory_preflight=False,initial_mapper=None):
    """Train on all eligible question pools and save validation-best mapper.

    Signature and output semantics mirror train_mapper: best/{manifest,weights},
    checkpoints/step-N, training-log.jsonl, training-summary.json. Also writes
    qa-protocol.json and question-consumption.jsonl. No model is downloaded and
    no cloud job is created. Caller supplies the approved pretrained runtime.
    """
    import torch
    from .train_mapper import NeuralMapper,RidgeMapper,bounded_optimizer_update,MapperGradientOverflow
    if steps<1 or max_prompt_tokens<2 or answer_tokens<1 or checkpoint_every<1 or max_seconds<=0 or not lr>0:
        raise ValueError('Invalid bounded QA training configuration')
    if not training_examples or not validation_examples or any(e.split!='calibration' for e in [*training_examples,*validation_examples]):
        raise ValueError('Nonempty calibration-only optimization and mapper selection required')
    assert_disjoint(training_examples,validation_examples)
    group_questions(training_examples)
    group_questions(validation_examples)
    root = Path(output)
    root.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    deadline = started+max_seconds
    torch.manual_seed(seed)
    for model in (runtime.source,runtime.target):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad=None
    native_limit=min(runtime.source.config.max_position_embeddings,runtime.target.config.max_position_embeddings)
    boundaries,eligible,excluded={},{},[]
    for role,examples in [('optimization',training_examples),('mapper_validation',validation_examples)]:
        eligible[role]=[]
        for example in examples:
            ids,prefix,suffix=split_document_prompt(runtime,example)
            if len(ids)>max_prompt_tokens or len(ids)+answer_tokens>native_limit:
                excluded.append({'role':role,'id':example.id,'document_id':example.document_id,
                    'full_prompt_tokens':len(ids),'reason':'full prompt exceeds declared/native context limit; no cropping'})
                continue
            boundaries[example.id]=(ids,prefix,suffix)
            eligible[role].append(example)
    if not eligible['optimization'] or not eligible['mapper_validation']:
        raise ValueError('No eligible complete prompts for optimization or mapper selection')
    sampler=DocumentQuestionSampler(eligible['optimization'],seed)
    validation,validation_policy=select_mapper_validation(eligible['mapper_validation'],seed)
    registry=[*training_examples,*validation_examples]
    protocol={'version':'qa-document-prefix-answer-distillation-v1','seed':seed,
        'boundary':'Tokenize exact full chat message once; offset-align document end; map full-token prefix; target processes original native question/template suffix. Straddling token stays native.',
        'sampling':'Seeded document permutation cycled uniformly; each document cycles all its eligible questions before reuse.',
        'full_context_policy':'No labeled context cropping. Explicitly exclude and record over-limit full prompts before sampling.',
        'objective':'Native target greedy answer teacher-token CE plus forward KL on answer positions; no gold answer supervision.',
        'reasoning_mode':'Existing runtime chat template enable_thinking=False',
        'answer_tokens':answer_tokens,'max_prompt_tokens':max_prompt_tokens,'native_context_limit':native_limit,
        'ce_weight':ce_weight,'kl_weight':kl_weight,'kv_weight':kv_weight,
        'checkpointed_mapper':checkpoint_mapper,
        'mapper_checkpoint_protocol':'per_target_layer_token_blocks_256_absolute_rope_v1' if checkpoint_mapper else 'none',
        'memory_preflight_requested':memory_preflight,'validation_policy':validation_policy,
        'validation_question_ids':[e.id for e in validation],
        'input_training_questions':len(training_examples),'input_validation_questions':len(validation_examples),
        'eligible_training_questions':len(eligible['optimization']),
        'eligible_validation_questions':len(eligible['mapper_validation']),
        'eligible_training_documents':len(sampler.groups),
        'excluded':excluded,'training_examples_sha256':json_digest([asdict(e) for e in training_examples]),
        'validation_examples_sha256':json_digest([asdict(e) for e in validation_examples])}
    (root/'qa-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    device=next(runtime.target.parameters()).device
    identities=[('source_sha256',runtime.identities['source']['sha256']),
                ('target_sha256',runtime.identities['target']['sha256']),
                ('tokenizer_sha256',runtime.tokenizer_sha256)]
    if initial_mapper is None:
        ridge=runtime.mapper
        if not isinstance(ridge,RidgeMapper):
            ridge=runtime.fit([questions[0] for questions in sampler.groups.values()],prefix_tokens=min(1024,max_prompt_tokens))
        for field,expected in identities:
            if ridge.manifest.get(field)!=expected:
                raise ValueError('Warm-start mapper identity mismatch: '+field)
        check_heldout(ridge.manifest,validation_examples)
        if set(ridge.manifest['calibration_documents'])-set(sampler.groups):
            raise ValueError('Ridge warm start contains documents outside the optimization split')
        mapper=NeuralMapper.from_ridge(ridge,runtime.target,rank=rank).to(device)
        protocol['initialization']='New neural mapper from optimization-only ridge fit'
    else:
        if not isinstance(initial_mapper,NeuralMapper):
            raise ValueError('initial_mapper must be a loaded NeuralMapper step-zero artifact')
        initial=initial_mapper.manifest
        for field,expected in identities:
            if initial.get(field)!=expected:
                raise ValueError('Initial mapper identity mismatch: '+field)
        if initial.get('trained_steps')!=0 or initial.get('selected_at_step',0)!=0 or initial.get('best_step',0)!=0:
            raise ValueError('Only exact untrained step-zero mapper initialization may be reused')
        if initial.get('rank')!=rank or tuple(initial.get('source_geometry',()))!=tuple(runtime.source_geometry) or tuple(initial.get('target_geometry',()))!=tuple(runtime.target_geometry):
            raise ValueError('Initial mapper rank or KV geometry does not match this run')
        prior_protocol=initial.get('task_training_protocol',{})
        for field in ('training_examples_sha256','validation_examples_sha256'):
            if prior_protocol.get(field)!=protocol[field]:
                raise ValueError('Initial mapper input examples changed: '+field)
        if prior_protocol.get('version')!=protocol['version']:
            raise ValueError('Initial mapper task protocol version changed')
        expected_registry={
            'training_documents':sorted({e.document_id for e in training_examples}),
            'mapper_validation_documents':sorted({e.document_id for e in validation_examples}),
            'calibration_ids':sorted({e.id for e in registry}),
            'calibration_documents':sorted({e.document_id for e in registry}),
            'calibration_context_hashes':sorted({e.context_hash for e in registry}),
            'training_question_ids':sorted(e.id for e in eligible['optimization']),
            'mapper_validation_question_ids':[e.id for e in validation]}
        for field,expected in expected_registry.items():
            if initial.get(field)!=expected:
                raise ValueError('Initial mapper eligible QA/source registry changed: '+field)
        protocol['initialization']='Exact reused step-zero neural mapper; no ridge refit or mapper reinitialization; fresh optimizer'
        protocol['initial_mapper_manifest_sha256']=json_digest(initial)
        protocol['initial_mapper_weights_sha256']=initial.get('weights_sha256')
        mapper=initial_mapper.to(device)
        for parameter in mapper.parameters():parameter.requires_grad_(True)
    # Save current execution/memory protocol separately from immutable upstream
    # artifacts; only the newly created output directory is written.
    (root/'qa-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    mapper.manifest.update(method='qa_document_prefix_functional_kl_ce_topk_residual',
        task_training_protocol=protocol,
        calibration_ids=sorted({e.id for e in registry}),
        calibration_documents=sorted({e.document_id for e in registry}),
        calibration_context_hashes=sorted({e.context_hash for e in registry}),
        training_documents=sorted({e.document_id for e in training_examples}),
        mapper_validation_documents=sorted({e.document_id for e in validation_examples}),
        training_question_ids=sorted(e.id for e in eligible['optimization']),
        mapper_validation_question_ids=[e.id for e in validation],thesis_verified=False)
    optimizer=torch.optim.AdamW(mapper.parameters(),lr=lr,weight_decay=.001)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',init_scale=1024.)
    best={'loss':math.inf,'step':0,'state':None}
    completed=0
    overflow_count=0
    consumed=Counter()
    hypotheses=Counter()
    last_heartbeat=time.perf_counter()

    def gpu_memory():
        result={}
        for current in {next(m.parameters()).device for m in (runtime.source,runtime.target)}:
            if current.type=='cuda':
                result[str(current)]={'allocated_mib':torch.cuda.memory_allocated(current)/2**20,
                    'reserved_mib':torch.cuda.memory_reserved(current)/2**20,
                    'peak_allocated_mib':torch.cuda.max_memory_allocated(current)/2**20,
                    'peak_reserved_mib':torch.cuda.max_memory_reserved(current)/2**20,
                    'total_mib':torch.cuda.get_device_properties(current).total_memory/2**20}
        return result

    def emit(row):
        with (root/'training-log.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        print('qa_mapper '+json.dumps(row),flush=True)

    def loss_for(example):
        ids,prefix,suffix=boundaries[example.id]
        return qa_training_loss(runtime,mapper,ids,prefix,suffix,answer_tokens=answer_tokens,
            ce_weight=ce_weight,kl_weight=kl_weight,kv_weight=kv_weight,checkpoint_mapper=checkpoint_mapper)

    if memory_preflight:
        candidates=[*eligible['optimization'],*eligible['mapper_validation']]
        longest=max(candidates,key=lambda e:len(boundaries[e.id][1]))
        largest_attention=max(candidates,key=lambda e:len(boundaries[e.id][1])*(len(boundaries[e.id][2])+answer_tokens))
        probes=list({e.id:e for e in (longest,largest_attention)}.values())
        preflight={'status':'running','optimizer_updates':0,'probes':[],
            'selection':'Longest eligible source prefix and greatest prefix_tokens*(native_query_tokens+answer_budget), deduplicated',
            'checkpoint_protocol':protocol['mapper_checkpoint_protocol'],
            'optimizer_state_materialized':False}
        snapshots={name:p.detach().cpu().clone() for name,p in mapper.named_parameters()}
        cpu_rng=torch.get_rng_state().clone()
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            for example in probes:
                optimizer.zero_grad(set_to_none=True)
                for current in {next(m.parameters()).device for m in (runtime.source,runtime.target)}:
                    if current.type=='cuda':torch.cuda.reset_peak_memory_stats(current)
                ids,prefix,suffix=boundaries[example.id]
                probe={'id':example.id,'document_id':example.document_id,'status':'running',
                    'prefix_tokens':len(prefix),'native_query_tokens':len(suffix),
                    'full_prompt_tokens':len(ids),'answer_budget':answer_tokens}
                preflight['active_probe']=dict(probe)
                (root/'memory-preflight.json').write_text(json.dumps(preflight,indent=2,allow_nan=False)+'\n')
                if callback:callback({'phase':'memory_preflight_start','step':0,**probe,'gpu_memory':gpu_memory()})
                loss=None
                gradients=None
                try:
                    loss,metrics=loss_for(example)
                    loss.backward()  # Deliberately no scaler or optimizer step.
                    gradients=[p.grad for p in mapper.parameters() if p.grad is not None]
                    if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
                        raise RuntimeError('Memory preflight has missing or nonfinite mapper gradients')
                    if any(p.grad is not None for m in (runtime.source,runtime.target) for p in m.parameters()):
                        raise RuntimeError('Memory preflight reached frozen base-model gradients')
                    if optimizer.state or any(not torch.equal(p.detach().cpu(),snapshots[name]) for name,p in mapper.named_parameters()):
                        raise RuntimeError('Memory preflight mutated mapper parameters or optimizer state')
                    probe.update(status='passed',gradients_finite=True,base_models_frozen=True,
                                 mapper_parameters_unchanged=True,optimizer_updates=0,**metrics)
                except BaseException as exc:
                    probe.update(status='failed',error_type=type(exc).__name__,error=str(exc))
                    preflight['status']='failed'
                    raise
                finally:
                    optimizer.zero_grad(set_to_none=True)
                    loss=None
                    gradients=None
                    probe.update(gpu_memory=gpu_memory(),elapsed_seconds=time.perf_counter()-started)
                    preflight['probes'].append(probe)
                    (root/'memory-preflight.json').write_text(json.dumps(preflight,indent=2,allow_nan=False)+'\n')
                    if callback:callback({'phase':'memory_preflight','step':0,**probe})
            preflight['status']='passed'
            preflight.pop('active_probe',None)
            (root/'memory-preflight.json').write_text(json.dumps(preflight,indent=2,allow_nan=False)+'\n')
        finally:
            optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
            del snapshots

    def validate(step):
        mapper.eval()
        measurements=[]
        with torch.no_grad():
            for example in validation:
                _,metrics=loss_for(example)
                measurements.append(metrics)
        value=sum(row['loss'] for row in measurements)/len(measurements)
        if value<best['loss']:
            best.update(loss=value,step=step,state={k:v.detach().cpu().clone() for k,v in mapper.state_dict().items()})
        mapper.manifest.update(trained_steps=step,selected_at_step=step)
        path=root/'checkpoints'/f'step-{step:06d}'
        mapper.save(path)
        row={'step':step,'validation_loss':value,'best_step':best['step'],'best_validation_loss':best['loss'],
            'validation_questions':len(validation),'validation_documents':len({e.document_id for e in validation}),
            'checkpoint_path':str(path),'elapsed_seconds':time.perf_counter()-started,'loss_scale':scaler.get_scale()}
        emit(row)
        if callback:callback(row)
        mapper.train()

    validate(0)
    for step in range(1,steps+1):
        if time.perf_counter()>=deadline:break
        example=sampler.next()
        def overflow(event):
            nonlocal overflow_count
            overflow_count+=1
            row={'status':'gradient_overflow_retry','step':completed,'attempted_step':step,
                 'completed_updates':completed,'question_id':example.id,'document_id':example.document_id,
                 'elapsed_seconds':time.perf_counter()-started,**event}
            emit(row)
            if callback:callback(row)
        try:
            metrics=bounded_optimizer_update(lambda:loss_for(example),mapper,optimizer,scaler,
                deadline=deadline,on_overflow=overflow)
        except TimeoutError:break
        except MapperGradientOverflow as exc:
            failure={'status':'nonfinite_gradients','attempted_step':step,'completed_updates':completed,
                     'question_id':example.id,'overflow_attempts_total':overflow_count,**exc.details}
            (root/'training-failure.json').write_text(json.dumps(failure,indent=2)+'\n')
            if callback:callback(failure)
            raise
        completed=step
        consumed[example.id]+=1
        signature=json_digest(' '.join(example.question.split()))
        hypotheses[signature]+=1
        row={'step':step,'completed_updates':completed,'question_id':example.id,'document_id':example.document_id,
             'question_type_sha256':signature,'prompt_sha256':json_digest(boundaries[example.id][0]),
             'elapsed_seconds':time.perf_counter()-started,**metrics}
        emit(row)
        with (root/'question-consumption.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        if callback and time.perf_counter()-last_heartbeat>=30:
            callback({'phase':'training_heartbeat','step':completed,'completed_updates':completed,
                'loss':metrics['loss'],'elapsed_seconds':time.perf_counter()-started,'gpu_memory':gpu_memory()})
            last_heartbeat=time.perf_counter()
        if step%checkpoint_every==0 or step==steps:validate(step)
    if completed%checkpoint_every and completed!=steps:validate(completed)
    mapper.load_state_dict(best['state'])
    mapper.eval()
    mapper.manifest.update(trained_steps=completed,best_step=best['step'],selected_at_step=best['step'],
        best_validation_loss=best['loss'],consumed_question_ids=sorted(consumed),
        question_consumption_counts=dict(consumed))
    mapper.save(root/'best')
    summary={'status':'completed' if completed==steps else 'time_budget_reached',
        'requested_steps':steps,'completed_steps':completed,'completed_updates':completed,
        'best_step':best['step'],'best_validation_loss':best['loss'],
        'training_documents':len(sampler.groups),'training_questions_available':len(eligible['optimization']),
        'training_questions_consumed':len(consumed),'training_question_type_counts':dict(hypotheses),
        'mapper_validation_documents':len({e.document_id for e in validation}),
        'mapper_validation_questions':len(validation),'overflow_attempts':overflow_count,
        'forward_backward_attempts':completed+overflow_count,
        'base_models_frozen':all(not p.requires_grad and p.grad is None for m in (runtime.source,runtime.target) for p in m.parameters()),
        'loss_scale':scaler.get_scale(),'elapsed_seconds':time.perf_counter()-started,
        'best_mapper':str(root/'best'),'protocol':'qa-document-prefix-answer-distillation-v1',
        'gold_answer_used':False,'thesis_verified':False}
    (root/'training-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    runtime.mapper=mapper
    if callback:callback(summary)
    return mapper
