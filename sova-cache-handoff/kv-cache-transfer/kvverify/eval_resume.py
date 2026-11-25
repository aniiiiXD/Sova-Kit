"""Execution-status-only reference continuation; never choose by correctness."""
import hashlib
import json

from .data import json_digest
from .reference_eval import VARIANTS

STATUSES = {'ok', 'error', 'over_limit', 'not_run_time_budget'}


def quality_items(examples):
    return [{'id':e.id,'document_id':e.document_id,'context_hash':e.context_hash,
             'gold_answer':e.answer} for e in examples]


def hellaswag_items(records):
    return [{'id':r['benchmark_id'],'document_id':r['source_id'],
             'context_hash':hashlib.sha256(r['query'].encode()).hexdigest(),
             'gold_index':r['gold_index']} for r in records]


def _validated_rows(items, rows, protocol, config):
    expected = {item['id']:item for item in items}
    if len(expected) != len(items) or not expected:
        raise ValueError('Frozen item list must be nonempty with unique IDs')
    if rows and config is None:
        raise ValueError('Rows require their exact source config')
    if config is not None:
        if config.get('protocol') != protocol:
            raise ValueError('Source protocol differs from frozen continuation protocol')
        unsigned = {k:v for k,v in config.items() if k != 'run_config_hash'}
        if config.get('run_config_hash') != json_digest(unsigned):
            raise ValueError('Source evaluation config hash is invalid')
    indexed = {}
    for row in rows:
        key = row.get('id'),row.get('variant')
        if key in indexed or key[0] not in expected or key[1] not in VARIANTS:
            raise ValueError('Duplicate or unknown result ID/variant')
        if row.get('protocol') != protocol or row.get('status') not in STATUSES or row.get('fallback') is not False:
            raise ValueError('Unknown protocol/status or hidden fallback')
        if row.get('run_config_hash') != config['run_config_hash']:
            raise ValueError('Result config hash mismatch')
        if row.get('repeat') != 0 or not isinstance(row.get('prompt_hash'),str):
            raise ValueError('Reference repeat/prompt identity is missing')
        for field,value in expected[key[0]].items():
            if row.get(field) != value:
                raise ValueError('Result differs from frozen ID/document/context/label: '+field)
        indexed[key] = row
    return indexed


def plan_resume(items, parent_rows, protocol, parent_config=None):
    """Missing/error/over-limit/budget rows trigger ALL variants of that ID."""
    indexed = _validated_rows(items,parent_rows,protocol,parent_config)
    complete = [item['id'] for item in items if all(
        indexed.get((item['id'],variant),{}).get('status') == 'ok' for variant in VARIANTS)]
    complete_set = set(complete)
    return {'protocol':protocol,'frozen_items':items,'frozen_items_sha256':json_digest(items),
        'parent_rows_sha256':json_digest(parent_rows),'parent_config':parent_config,
        'complete_parent_ids':complete,
        'pending_ids':[item['id'] for item in items if item['id'] not in complete_set],
        'selection_policy':'Any missing/non-ok execution variant; never correctness, loss, confidence, or generation text',
        'variant_order_policy':'Subset evaluation reseeds shuffled variant order; continuation timing is not comparable across jobs',
        'latency_headline_permitted':False}


def merge_resume(plan,parent_rows,child_rows,child_config,*,parent_job,child_job,mapper_weights_sha256):
    """Whole-triplet replacement only; retain the full fixed denominator."""
    items = plan['frozen_items']
    if json_digest(items) != plan['frozen_items_sha256'] or json_digest(parent_rows) != plan['parent_rows_sha256']:
        raise ValueError('Continuation plan inputs changed')
    parent = _validated_rows(items,parent_rows,plan['protocol'],plan['parent_config'])
    complete = set(plan['complete_parent_ids'])
    expected_complete = {item['id'] for item in items if all(
        parent.get((item['id'],variant),{}).get('status') == 'ok' for variant in VARIANTS)}
    if complete != expected_complete or plan['pending_ids'] != [i['id'] for i in items if i['id'] not in complete]:
        raise ValueError('Continuation plan selection changed')
    pending_items = [item for item in items if item['id'] not in complete]
    child = _validated_rows(pending_items,child_rows,plan['protocol'],child_config) if pending_items else {}
    if not pending_items and child_rows:
        raise ValueError('Completed parent cannot be rerun')
    merged = []
    for item in items:
        use_parent = item['id'] in complete
        source,config,job = (parent,plan['parent_config'],parent_job) if use_parent else (child,child_config,child_job)
        template=next((source[(item['id'],v)] for v in VARIANTS if (item['id'],v) in source),{})
        for variant in VARIANTS:
            row = source.get((item['id'],variant))
            if row is None:
                row = {**item,'variant':variant,'protocol':plan['protocol'],'fallback':False,
                       'status':'not_run_time_budget','repeat':0,'prompt_hash':template.get('prompt_hash','unavailable'),
                       'run_config_hash':config['run_config_hash'] if config else 'not_started'}
            merged.append({**row,'resume_provenance':{
                'parent_job':parent_job,'continuation_job':child_job,'source_job':job,
                'source_row_sha256':json_digest(row),'source_run_config_hash':row['run_config_hash'],
                'mapper_weights_sha256':mapper_weights_sha256,
                'whole_triplet_origin':'parent' if use_parent else 'continuation',
                'latency_headline_permitted':False}})
    return merged


def verify_evaluation_config(config,*,mapper_weights_sha256,runtime_identities,quality,limit=9000,freegen=16):
    """Resume only the same weights, pretrained pair, context and generation."""
    if config is None:
        return
    if config.get('mapper',{}).get('weights_sha256') != mapper_weights_sha256 or config.get('runtime') != runtime_identities:
        raise ValueError('Parent/child mapper or pretrained model identity changed')
    if quality:
        if config.get('prompt_tokens_limit') != limit or config.get('freegen_max_new_tokens') != freegen:
            raise ValueError('QuALITY context/generation protocol changed')
    elif config.get('max_tokens') != limit:
        raise ValueError('HellaSwag context protocol changed')
