"""Summarize completed stored diagnostic records; no model evaluation/inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def flatten(value):
    if isinstance(value,list):
        for item in value:yield from flatten(item)
    else:
        if not isinstance(value,(int,float))or not math.isfinite(value):raise ValueError('Nonfinite diagnostic statistic')
        yield float(value)


def mean_nested(value):return statistics.fmean(flatten(value))


def report(directory):
    directory=Path(directory);data=directory/'collected'
    summary=json.loads((data/'summary.json').read_text());rows=json.loads((data/'rows.json').read_text())
    if summary['status']!='completed_diagnostic'or len(rows)!=24 or len({r['id']for r in rows})!=24:
        raise ValueError('Only a complete24-document diagnostic may generate this report')
    if len({r['document_id']for r in rows})!=24 or any(r['status']!='ok'or not r['native_control']['passed']for r in rows):
        raise ValueError('Document coverage or native numerical controls failed')
    arms=['step0_rms_on','step0_rms_off','step48_rms_on','step48_rms_off']
    interventions=['mapped_k_native_v','native_k_mapped_v','mapped_k_mapped_v']
    extracted={};layer_values={};hashes={}
    for row in rows:
        if set(row['variants'])!=set(arms):raise ValueError('Four complete mapper variants required per document')
        logits=data/row['logits_file']
        if hashlib.sha256(logits.read_bytes()).hexdigest()!=row['logits_sha256']:raise ValueError('Stored raw-logit hash mismatch')
        hashes[row['logits_file']]=row['logits_sha256']
        for arm in arms:
            variant=row['variants'][arm]
            payload=json.loads((data/variant['readouts_file']).read_text())
            if payload['id']!=row['id']or payload['arm']!=arm or len(payload['layers'])!=28:
                raise ValueError('Readout identity or layer coverage mismatch')
            doc={}
            for intervention in interventions:
                per_layer=[mean_nested(layer['arms'][intervention]['relative_output_error'])for layer in payload['layers']]
                doc[intervention]=statistics.fmean(per_layer)
                layer_values.setdefault((arm,intervention),[]).append(per_layer)
            doc['query_relative_error']=statistics.fmean(mean_nested(layer['actual_transfer_query_relative_error'])for layer in payload['layers'])
            extracted[(row['id'],arm)]=doc
    computed={'documents':24,'native_correct':sum(r['native_score']['decision_correct']for r in rows),
        'native_controls_passed':24,'variants':{},'paired_rms_effects':{},'length_bands':{},
        'aggregation':'Readout/query errors averaged across sampled queries/heads within layer, then28layers within document; median across24documents. No token-count pseudoreplication.',
        'interpretation':'Already-selected validation questions; exploratory mechanistic diagnostic, no final-test or serving-speed claim',
        'verified_raw_logit_hashes':hashes,'peak_gpu_allocated_mib':summary['peak_gpu_allocated_mib']}
    for arm in arms:
        values=[extracted[(r['id'],arm)]for r in rows]
        computed['variants'][arm]={
            'correct':sum(r['variants'][arm]['score']['decision_correct']for r in rows),
            'native_agreement':sum(r['variants'][arm]['native_choice_agreement']for r in rows),
            'mean_native_kl_nats':statistics.fmean(r['variants'][arm]['distribution']['native_to_candidate_kl_nats']for r in rows),
            'median_native_kl_nats':statistics.median(r['variants'][arm]['distribution']['native_to_candidate_kl_nats']for r in rows),
            'median_option_mass':statistics.median(r['variants'][arm]['score']['option_probability_mass']for r in rows),
            'median_document_readout_error':{name:statistics.median(v[name]for v in values)for name in interventions},
            'median_document_query_relative_error':statistics.median(v['query_relative_error']for v in values),
            'mean_error_by_layer':{name:[statistics.fmean(v[i]for v in layer_values[(arm,name)])for i in range(28)]for name in interventions}}
    for checkpoint in('step0','step48'):
        on=checkpoint+'_rms_on';off=checkpoint+'_rms_off'
        differences=[r['variants'][off]['distribution']['native_to_candidate_kl_nats']-r['variants'][on]['distribution']['native_to_candidate_kl_nats']for r in rows]
        correct_delta=sum(r['variants'][off]['score']['decision_correct']-r['variants'][on]['score']['decision_correct']for r in rows)
        computed['paired_rms_effects'][checkpoint]={'off_minus_on_correct_count':correct_delta,
            'off_lower_native_kl_documents':sum(d<0 for d in differences),'off_higher_native_kl_documents':sum(d>0 for d in differences),
            'median_off_minus_on_native_kl_nats':statistics.median(differences),
            'mean_off_minus_on_native_kl_nats':statistics.fmean(differences)}
    for band in range(3):
        selected=[r for r in rows if r['length_band']==band]
        if len(selected)!=8:raise ValueError('Each predeclared length band needs8documents')
        computed['length_bands'][str(band)]={'count':8,'min_prompt_tokens':min(r['prompt_tokens']for r in selected),
            'max_prompt_tokens':max(r['prompt_tokens']for r in selected),'native_correct':sum(r['native_score']['decision_correct']for r in selected),
            'variants':{a:{'correct':sum(r['variants'][a]['score']['decision_correct']for r in selected),
                'mean_native_kl_nats':statistics.fmean(r['variants'][a]['distribution']['native_to_candidate_kl_nats']for r in selected)}for a in arms}}
    aws_file=directory/'aws-final.json'
    if aws_file.exists():
        status=json.loads(aws_file.read_text());receipt=json.loads((directory/'submission.json').read_text())
        seconds=status.get('BillableTimeInSeconds')
        computed['worker']={'job_name':status['TrainingJobName'],'status':status['TrainingJobStatus'],'billable_seconds':seconds,
            'estimated_runtime_compute_usd':seconds*receipt['price']['usd_per_hour']/3600 if seconds is not None else None,
            'pricing_excludes':'storage/transfer/billing extras/tax'}
    (directory/'diagnostic-analysis.json').write_text(json.dumps(computed,indent=2,allow_nan=False)+'\n')
    table=['| Arm | Correct /24 | Native choice agreement /24 | Mean native→mapped KL (nats) | Median option mass |',
           '|---|---:|---:|---:|---:|']
    for arm,values in computed['variants'].items():
        table.append(f"| {arm} | {values['correct']} | {values['native_agreement']} | {values['mean_native_kl_nats']:.4f} | {values['median_option_mass']:.4g} |")
    readouts=['| Arm | K-only readout error | V-only readout error | Both mapped | Actual query drift |',
              '|---|---:|---:|---:|---:|']
    for arm,values in computed['variants'].items():
        m=values['median_document_readout_error']
        readouts.append(f"| {arm} | {m['mapped_k_native_v']:.4f} | {m['native_k_mapped_v']:.4f} | {m['mapped_k_mapped_v']:.4f} | {values['median_document_query_relative_error']:.4f} |")
    text=['# State diagnostic v2: complete stored measurements','',
        'All24validation documents completed all four variants. Native cache numerical controls and exact-shape native attention replay passed. No training occurred. These are development diagnostics, not final-test accuracy or serving results.','',
        f"Native target correctness: **{computed['native_correct']}/24**.",'',*table,'',
        'Readout entries below are median document-level mean relative L2 errors under native queries. K-only and V-only are oracle interventions; actual query drift is measured from a real mapped-cache suffix forward. They are not interchangeable measures.','',*readouts,'',
        'Aggregation, paired effects, length strata, layer profiles and raw-logit hashes are in [diagnostic-analysis.json](diagnostic-analysis.json). No speedup was measured.','',
        '[Raw rows](collected/rows.json) · [Worker summary](collected/summary.json) · [Protocol](collected/protocol.json) · [Submission receipt](submission.json)']
    (directory/'MEASUREMENTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in computed.items()if k not in('variants','verified_raw_logit_hashes')},indent=2))
    print('\n'.join(table));print('\n'.join(readouts))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    report(parser.parse_args().directory)
