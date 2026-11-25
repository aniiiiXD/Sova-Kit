"""Bounded, GET-only collection from an explicitly selected local Aegis daemon."""
import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from switchit.contracts import require


class AegisReadError(ValueError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise AegisReadError('Aegis redirects are not allowed')


class AegisClient:
    def __init__(self, url, *, timeout=10, max_bytes=4 * 1024 * 1024):
        parsed = urlsplit(url)
        require(parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1', '::1')
                and parsed.port and not parsed.username and not parsed.password
                and parsed.path in ('', '/') and not parsed.query and not parsed.fragment,
                'Aegis URL must be http://127.0.0.1:PORT (or localhost/[::1]), without credentials or a path')
        host = '[::1]' if parsed.hostname == '::1' else '127.0.0.1'
        self.url = f'http://{host}:{parsed.port}'
        self.timeout, self.max_bytes = timeout, max_bytes
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def get(self, path, params=None):
        require(path.startswith('/api/') and '?' not in path and '#' not in path, 'invalid Aegis API path')
        url = self.url + path + ('?' + urlencode(params) if params else '')
        try:
            with self.opener.open(Request(url, headers={'Accept': 'application/json'}, method='GET'),
                                  timeout=self.timeout) as response:
                raw = response.read(self.max_bytes + 1)
            require(len(raw) <= self.max_bytes, 'Aegis response exceeded size limit')
            return json.loads(raw)
        except HTTPError as error:
            error.close()
            raise AegisReadError(f'Aegis GET failed (HTTP {error.code})') from None
        except (URLError, TimeoutError, OSError):
            raise AegisReadError('Cannot read local Aegis daemon; check its URL and that it is running') from None
        except (UnicodeError, json.JSONDecodeError):
            raise AegisReadError('Aegis returned invalid JSON') from None


def collect_aegis(url, task_id, *, max_pages=10, client=None):
    require(isinstance(task_id, str) and task_id.strip() and len(task_id) <= 200,
            'Aegis task ID must be a nonempty string of at most 200 characters')
    require(type(max_pages) is int and 1 <= max_pages <= 20, 'max_pages must be between 1 and 20')
    client = client or AegisClient(url)
    base = '/api/tasks/' + quote(task_id, safe='')
    bundle = client.get(base + '/bundle')
    require(isinstance(bundle, dict) and bundle.get('source_task_id') == task_id, 'Aegis task identity mismatch')
    runs = client.get(base + '/runs')
    verifications = client.get(base + '/verifications')
    require(isinstance(runs, list) and isinstance(verifications, list), 'invalid Aegis runs/verifications response')
    # Aegis uses a signed 64-bit exclusive sequence cursor. Pin the initial page
    # and walk backwards so new events cannot shift the rest of the traversal.
    cursor, events, seen = 9223372036854775807, [], set()
    has_more = True
    for _ in range(max_pages):
        page = client.get('/api/events', {'task_id': task_id, 'before_sequence': cursor, 'limit': 500})
        require(isinstance(page, dict) and isinstance(page.get('events'), list)
                and type(page.get('has_more')) is bool, 'invalid Aegis events page')
        rows = page['events']
        require(len(rows) <= 500, 'Aegis page exceeds requested limit')
        for row in rows:
            require(isinstance(row, dict) and type(row.get('sequence')) is int
                    and 0 <= row['sequence'] < cursor and isinstance(row.get('id'), str),
                    'invalid Aegis event cursor or identity')
            require(row['id'] not in seen, 'duplicate Aegis event across pages')
            seen.add(row['id'])
        events.extend(rows)
        has_more = page['has_more']
        if not has_more:
            break
        next_cursor = page.get('next_cursor')
        require(rows and type(next_cursor) is int and 0 < next_cursor < cursor
                and next_cursor == min(r['sequence'] for r in rows), 'Aegis pagination did not advance')
        cursor = next_cursor
    return {'format': 'aegis-snapshot', 'version': 1, 'bundle': bundle, 'runs': runs,
            'events': sorted(events, key=lambda row: (row['sequence'], row['id'])),
            'verifications': verifications,
            'coverage': [
                {'resource': 'events', 'status': 'limited' if has_more else 'complete', 'records': len(events)},
                {'resource': 'runs', 'status': 'complete', 'records': len(runs)},
                {'resource': 'verifications', 'status': 'bounded', 'records': len(verifications)},
                {'resource': 'bundle', 'status': 'bounded', 'note': 'Aegis bounds messages and findings; reads are not an atomic snapshot.'},
            ]}
