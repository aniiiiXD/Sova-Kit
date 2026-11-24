"""Exact, noncausal references between agent events and imported GitHub evidence."""
import re
from urllib.parse import urlsplit

from .contracts import fingerprint, require
from .events import normalize_event, events_from_calls, validate_event
from .importers.github import repository_name

_URL = re.compile(r'https?://[^\s<>"\x27`]+', re.IGNORECASE)
_SHA = re.compile(r'(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\Z')


def read_events(document):
    """Accept canonical events, an active revisioned store, or a canonical calls store."""
    if isinstance(document, dict) and 'event_store_version' in document:
        from .import_pipeline import active_events
        rows = active_events(document)
    elif isinstance(document, dict) and 'calls' in document:
        rows = events_from_calls(document['calls'])
    else:
        rows = document.get('events') if isinstance(document,dict) else document
    require(isinstance(rows,list),'provide an events array, event store, or canonical calls store')
    return [normalize_event(event) for event in rows]


def combine_evidence(documents):
    events = {}
    for document in documents:
        for event in read_events(document):
            key = event['event_id']
            require(key not in events or events[key] == event, f'conflicting event ID: {key}')
            events[key] = event
    require(events,'no evidence supplied')
    return list(events.values())


def _strings(value, path=()):
    if isinstance(value,str):
        yield path,value
    elif isinstance(value,dict):
        for key,child in value.items():
            yield from _strings(child,path+(key,))
    elif isinstance(value,list):
        for index,child in enumerate(value):
            yield from _strings(child,path+(index,))


def _url_reference(text):
    parsed = urlsplit(text.rstrip('.,;:!?)]}'))
    if parsed.hostname != 'github.com' or parsed.username is not None or parsed.password is not None:
        return None
    if parsed.netloc.lower() != 'github.com':
        return None
    parts = parsed.path.strip('/').split('/')
    if len(parts)<4:
        return None
    try:
        repo=repository_name('/'.join(parts[:2]))
    except ValueError:
        return None
    kind,identity=parts[2:4]
    if kind in ('pull','issues') and re.fullmatch(r'[1-9][0-9]*',identity):
        if len(parts)>4 and parts[4:] not in (['files'],['commits'],['checks']):
            return None
        return (repo,'pull' if kind=='pull' else 'issue',str(int(identity)))
    if kind=='commit' and len(parts)==4 and _SHA.fullmatch(identity):
        return repo,'commit',identity.lower()
    if parts[2:4]==['actions','runs'] and len(parts)==5 and re.fullmatch(r'[1-9][0-9]*',parts[4]):
        return repo,'run',str(int(parts[4]))
    return None


def _target_keys(event):
    if event['source'].get('importer')!='github':
        return []
    try:
        repo=repository_name(event['source'].get('repository'))
    except ValueError:
        return []
    payload=event['payload'];kind=payload.get('github_type')
    if kind=='threads':
        number=payload.get('number')
        if not isinstance(number,int) or isinstance(number,bool) or number<=0:
            return []
        keys=[(repo,'issue',str(number))]
        if payload.get('thread_type')=='pull_requests':keys.append((repo,'pull',str(number)))
        return keys
    if kind=='commits' and isinstance(payload.get('sha'),str) and _SHA.fullmatch(payload['sha']):
        return [(repo,'commit',payload['sha'].lower())]
    if kind=='runs' and isinstance(payload.get('id'),int) and payload['id']>0:
        return [(repo,'run',str(payload['id']))]
    return []


def correlate_github(events):
    """A match proves only that a recorded payload references an imported resource.

    Names, timestamps, branch names, bare issue numbers and abbreviated SHAs never
    establish identity. Multiple imported snapshots/attempts remain ambiguous.
    """
    index={}
    for event in events:
        validate_event(event)
        for key in _target_keys(event):
            index.setdefault(key,[]).append(event['event_id'])
    links,unresolved=[],[]
    for event in events:
        if event['source'].get('importer')=='github':
            continue
        references={}
        for path,text in _strings(event['payload']):
            for match in _URL.finditer(text):
                try:
                    key=_url_reference(match.group())
                except ValueError:
                    key=None
                if key:
                    references.setdefault(key,[]).append({'payload_path':list(path),'method':'exact_url'})
        explicit=event['payload'].get('github_reference')
        if isinstance(explicit,dict):
            try:
                repo=repository_name(explicit.get('repository'))
                for field,kind in (('pull_number','pull'),('issue_number','issue'),('commit_sha','commit'),('run_id','run')):
                    if field not in explicit:continue
                    value=explicit[field]
                    valid=bool(_SHA.fullmatch(value)) if kind=='commit' and isinstance(value,str) else (
                        kind!='commit' and isinstance(value,int) and not isinstance(value,bool) and value>0)
                    require(valid,'invalid explicit GitHub identity')
                    key=(repo,kind,str(value).lower())
                    references.setdefault(key,[]).append({'payload_path':['github_reference',field],'method':'explicit_identity'})
            except ValueError:
                unresolved.append({'source_event_id':event['event_id'],'reason':'invalid_explicit_reference'})
        for key,observations in sorted(references.items()):
            matches=sorted(set(index.get(key,[])))
            resource={'repository':key[0],'kind':key[1],'identity':key[2]}
            finding={'source_event_id':event['event_id'],'resource':resource,'observations':observations}
            if len(matches)==1:
                links.append({**finding,'target_event_id':matches[0],'relation':'references_github',
                              'evidence':'recorded','causal':False,'proves_authorship':False})
            else:
                unresolved.append({**finding,'reason':'not_imported' if not matches else 'ambiguous_snapshot',
                                   'candidate_event_ids':matches})
    return {'correlation_version':1,'evidence_digest':fingerprint(events),'links':links,'unresolved':unresolved,
            'limitations':['A recorded reference does not establish authorship, execution success, or workflow equivalence.',
                           'No fuzzy matches, short SHA matching, network fetches, or timestamp-based identity inference.']}
