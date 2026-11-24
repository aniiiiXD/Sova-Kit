"""GitHub REST export → provenance-preserving activity evidence, not model traces."""
from copy import deepcopy
import re

from switchit.contracts import fingerprint, require
from switchit.events import normalize_event
from switchit.redaction import redact_value

COLLECTIONS = ('issues', 'pull_requests', 'comments', 'reviews', 'review_comments', 'commits', 'runs', 'jobs')


def repository_name(value):
    require(isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', value))
            and all(p not in ('.','..') for p in value.split('/')), 'repository must be owner/name')
    return value.lower()


def import_github(document):
    """Read the documented bundle schema; linked records use explicit REST IDs only."""
    require(isinstance(document, dict) and document.get('github_export_version') == 1, 'unsupported GitHub export version')
    repo = repository_name(document.get('repository'))
    require(set(document) <= {'github_export_version','repository','scope','coverage',*COLLECTIONS}, 'unknown GitHub export field')
    data = redact_value(deepcopy(document))
    revision = fingerprint(data)
    events, identities, pending_links = [], {}, []
    counters = {}

    def emit(collection, identity, raw, kind='state', parent=None, trace=None, payload=None, timestamp=None):
        key = (collection, str(identity))
        require(key not in identities, f'duplicate GitHub {collection} identity: {identity}')
        stable = f'github:{repo}:{collection}:{identity}'
        eid = stable+':'+revision[:20]
        identities[key] = eid
        trace = trace or stable
        index = counters.get(trace,0);counters[trace]=index+1
        body = payload if payload is not None else raw
        event = normalize_event({'event_id':eid,'record_id':stable,'trace_id':trace,'session_id':trace,
            'index':index,'kind':kind,'timestamp':timestamp or raw.get('submitted_at') or raw.get('created_at') or raw.get('started_at'),
            'source':{'importer':'github','repository':repo,'revision':revision,'record_type':collection,
                      'url':raw.get('html_url') or f'https://github.com/{repo}'},
            'evidence':'metadata_only','payload':{'github_type':collection,**body},
            'uncertainty':['agent_execution_not_observed','github_snapshot_not_complete_history']})
        if parent:
            pending_links.append((eid,parent,'parent'))
        events.append(event)
        return eid

    records = {}
    for collection in COLLECTIONS:
        rows = data.get(collection,[])
        require(isinstance(rows,list) and all(isinstance(row,dict) for row in rows), f'{collection} must be an object array')
        records[collection]=rows
    def rid(row, key='id'):
        value=row.get(key)
        require(isinstance(value,(str,int)) and not isinstance(value,bool) and str(value), f'GitHub record requires {key}')
        return value
    def thread(number):
        require(isinstance(number,int) and not isinstance(number,bool) and number>0,'issue_number must be positive')
        return f'github:{repo}:thread:{number}'
    prs={rid(row,'number') for row in records['pull_requests']}
    for collection in ('issues','pull_requests'):
        for row in records[collection]:
            number=rid(row,'number')
            if collection=='issues' and number in prs:
                continue  # GitHub PRs are also issues; retain the richer selected PR record.
            emit('threads',number,row,kind='user_input',trace=thread(number),payload={
                'text':row.get('title',''), 'title':row.get('title'), 'body':row.get('body'),
                'number':number,'thread_type':collection,'state':row.get('state'),'merged':row.get('merged'),
                'author':(row.get('user') or {}).get('login'),'head_sha':(row.get('head') or {}).get('sha'),
                'updated_at':row.get('updated_at'),'closed_at':row.get('closed_at'),'merged_at':row.get('merged_at')})
    for collection in ('comments','reviews','review_comments'):
        for row in records[collection]:
            number=row.get('issue_number') or row.get('pull_number')
            emit(collection,rid(row),row,trace=thread(number),parent=('threads',number),payload={
                k:row.get(k) for k in ('body','state','commit_id','path','line','in_reply_to_id','updated_at','user')})
    for row in records['commits']:
        sha=rid(row,'sha');commit=row.get('commit') or {}
        number=row.get('pull_number')
        emit('commits',sha,row,kind='artifact',trace=thread(number) if number else None,
             parent=('threads',number) if number else None,
             timestamp=(commit.get('committer') or {}).get('date'),payload={
                 'sha':sha,'message':commit.get('message'),'parents':[p.get('sha') for p in row.get('parents',[])],
                 'author':(row.get('author') or {}).get('login')})
    for row in records['runs']:
        run_id=rid(row);attempt=row.get('run_attempt',1)
        require(isinstance(attempt,int) and attempt>0,'run_attempt must be positive')
        identity=f'{run_id}:{attempt}'
        emit('runs',identity,row,payload={k:row.get(k) for k in ('id','name','event','status','conclusion','head_sha','run_attempt','updated_at')})
        for pr in row.get('pull_requests') or []:
            if isinstance(pr,dict) and pr.get('number'):
                pending_links.append((identities[('runs',identity)],('threads',pr['number']),'references_pull_request'))
        if row.get('head_sha'):
            pending_links.append((identities[('runs',identity)],('commits',row['head_sha']),'tested_commit'))
    for row in records['jobs']:
        jobid=rid(row);runid=rid(row,'run_id');attempt=row.get('run_attempt',1)
        trace=f'github:{repo}:runs:{runid}:{attempt}'
        emit('jobs',jobid,row,trace=trace,parent=('runs',f'{runid}:{attempt}'),payload={
            k:row.get(k) for k in ('id','run_id','run_attempt','name','status','conclusion','started_at','completed_at','head_sha')})
        for step in row.get('steps') or []:
            require(isinstance(step,dict),'job steps must be objects')
            number=rid(step,'number')
            emit('steps',f'{jobid}:{number}',{**step,'html_url':row.get('html_url')},trace=trace,parent=('jobs',jobid),
                 payload={k:step.get(k) for k in ('number','name','status','conclusion','started_at','completed_at')})
    links, unresolved=[],[]
    by_id={e['event_id']:e for e in events}
    for child,parent,relation in pending_links:
        target=identities.get((parent[0],str(parent[1])))
        if not target:
            unresolved.append({'event_id':child,'relation':relation,'reference':list(parent)})
            by_id[child]['uncertainty'].append('github_reference_not_imported')
        elif relation=='parent':
            by_id[child]['parent_id']=target
        else:
            links.append({'source':child,'target':target,'relation':relation,'evidence':'recorded', 'causal':False})
            by_id[child]['payload'].setdefault('github_references',[]).append({'event_id':target,'relation':relation})
    events=[normalize_event(e) for e in events]
    require(events,'GitHub export contains no supported records')
    return {'events':events,'relationships':links,'diagnostics':unresolved,'repository':repo,'revision':revision,
            'coverage':data.get('coverage',[]),'limitations':['GitHub activity does not contain hidden model reasoning, prompts, tool calls, or token usage.',
                'Only selected resources and observed relationships are included; commit authors are not inferred to be AI agents.',
                'Actions step conclusions are observations, not reusable tool fixtures. Logs and artifact contents are not downloaded.']}
