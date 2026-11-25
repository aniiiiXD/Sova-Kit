"""Cheap changed-input competitor: read only the first native document tokens."""
import json
from pathlib import Path
import time


def run_prefix_competitor(runtime,examples,output,*,prefix_tokens=256,max_seconds=300,
                          native_answers=None,callback=None):
    from .data import json_digest
    from .qa_training import split_document_prompt
    from .quality import option_token_ids,score_option_logits
    if not examples or prefix_tokens<1 or max_seconds<=0:raise ValueError('Bounded nonempty prefix competitor required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False);rows=[];deadline=time.monotonic()+max_seconds
    with(root/'rows.jsonl').open('w')as stream:
        for index,e in enumerate(examples):
            row={'id':e.id,'document_id':e.document_id,'competitor':'native_prefix_only',
                 'protocol':f'First{prefix_tokens}document-prefix tokens plus the exact full native question suffix; fresh short positions; no mapped remainder'}
            if time.monotonic()>=deadline:row.update(status='not_run_time_budget')
            else:
                try:
                    full,prefix,suffix=split_document_prompt(runtime,e)
                    kept=min(prefix_tokens,len(prefix));ids=prefix[:kept]+suffix
                    out=runtime.forward(runtime.target,ids)
                    score=score_option_logits(out.logits[0,-1].detach().float().cpu(),option_token_ids(runtime.tokenizer,e.choices),e.answer)
                    del out
                    row.update(status='ok',**score,native_prefix_tokens=kept,
                               full_prefix_tokens=len(prefix),omitted_prefix_tokens=len(prefix)-kept,
                               prompt_hash=json_digest(ids),original_prompt_hash=json_digest(full))
                    if native_answers is not None:row['native_agreement']=score['decision_answer']==native_answers.get(e.id)
                except Exception as exc:row.update(status='error',error_type=type(exc).__name__,error=str(exc))
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if callback and(index+1)%25==0:callback({'phase':'native_prefix_competitor','completed':index+1,'total':len(examples)})
    n=len(rows);summary={'planned_items':n,'completed_items':sum(r['status']=='ok'for r in rows),
        'errors':sum(r['status']=='error'for r in rows),'not_run_time_budget':sum(r['status']=='not_run_time_budget'for r in rows),
        'decision_correct':sum(r['status']=='ok'and r.get('decision_correct',False)for r in rows)/n,
        'native_agreement':sum(r['status']=='ok'and r.get('native_agreement',False)for r in rows)/n,
        'full_context_read':False,'note':'Changed-input competitor; do not call short-prompt accuracy evidence of full-document state transfer'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
