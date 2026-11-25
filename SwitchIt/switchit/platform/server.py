"""Loopback-first stdlib HTTP API and same-origin static workspace."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
from threading import BoundedSemaphore
from urllib.parse import unquote, urlsplit

from switchit.contracts import require
from switchit.data_paths import workspace_dir
from switchit.switchup.capture import invoke
from switchit.switchup.contracts import LOOPBACK
from switchit.switchup.loop import activate
from switchit.switchup.store import SwitchUpConflict, Workspace, project_summary

from .jobs import JobConflict, JobManager

MAX_BODY_BYTES = 2_000_000
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/improvements': ('improvements.html', 'text/html; charset=utf-8'),
          '/switchup.js': ('switchup.js', 'text/javascript; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8')}


class PlatformServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, manager, allowed_host, switchup, allow_live_models):
        self.jobs = manager
        self.allowed_host = allowed_host
        self.switchup = switchup
        self.allow_live_models = allow_live_models
        self.model_slots = BoundedSemaphore(4)
        self._closed = False
        super().__init__(address, PlatformHandler)

    def server_close(self):
        super().server_close()
        if not self._closed:
            self._closed = True
            self.jobs.close()


class PlatformHandler(BaseHTTPRequestHandler):
    server_version = 'SwitchUPLocal/1'

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args):
        # Request bodies and imported paths never enter the access log.
        pass

    def _send(self, status, body, content_type='application/json; charset=utf-8', filename=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _origin_allowed(self):
        host = self.headers.get('Host', '')
        parsed = urlsplit('http://' + host)
        allowed = {self.server.allowed_host, '127.0.0.1', 'localhost', '::1'}
        try:
            valid = parsed.hostname in allowed and parsed.port == self.server.server_address[1]
        except ValueError:
            valid = False
        if not valid:
            self._send(403, {'error': 'Host is not allowed for this local server.'})
            return False
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + host:
            self._send(403, {'error': 'Cross-origin requests are not allowed.'})
            return False
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            self._send(403, {'error': 'Cross-site requests are not allowed.'})
            return False
        return True

    def _body(self):
        if self.headers.get('Transfer-Encoding'):
            raise ValueError('Transfer encoding is not supported.')
        if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
            raise ValueError('Use Content-Type: application/json.')
        try:
            length = int(self.headers.get('Content-Length', ''))
        except ValueError:
            raise ValueError('A valid Content-Length is required.')
        if not 0 < length <= MAX_BODY_BYTES:
            raise OverflowError(f'Request body must contain 1–{MAX_BODY_BYTES} bytes.')
        self.connection.settimeout(10)
        data = self.rfile.read(length)
        if len(data) != length:
            raise ValueError('Request body ended early.')
        def invalid_constant(value):
            raise ValueError('JSON numbers must be finite.')
        def finite_float(value):
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError('JSON numbers must be finite.')
            return parsed
        def unique_object(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError('Duplicate JSON fields are not allowed.')
                result[key] = item
            return result
        try:
            value = json.loads(data.decode('utf-8'), parse_constant=invalid_constant,
                               parse_float=finite_float, object_pairs_hook=unique_object)
        except RecursionError:
            raise ValueError('JSON nesting is too deep.') from None
        pending = [(value, 0)]
        while pending:
            item, depth = pending.pop()
            if depth > 50:
                raise ValueError('JSON nesting exceeds 50 levels.')
            if isinstance(item, dict):
                pending.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
        if not isinstance(value, dict):
            raise ValueError('Request must be a JSON object.')
        return value

    def do_GET(self):
        self._dispatch(False)

    def do_POST(self):
        self._dispatch(True)

    def do_OPTIONS(self):
        self._send(405, {'error': 'Cross-origin API access is not supported.'})

    def _switchup(self, path, mutation):
        if self.server.allowed_host not in LOOPBACK:
            return self._send(403, {'error': 'SwitchUP uses a loopback workspace.'})
        workspace = self.server.switchup
        if path == '/api/switchup/projects' and not mutation:
            return self._send(200, {'projects': workspace.status()})
        match = re.fullmatch(r'/api/switchup/projects/([a-z0-9][a-z0-9-]{0,62})(?:/(active|events|feedback|invoke|activate|rollback)|/rounds/([a-f0-9]{32}))?', path)
        if match is None:
            return self._send(404, {'error': 'SwitchUP route not found.'})
        project, action, round_id = match.groups()
        if not mutation:
            if round_id:
                return self._send(200, workspace.round(project, round_id))
            if action is None:
                return self._send(200, workspace.status(project)[0])
            if action == 'active':
                record = workspace.project(project)
                return self._send(200, {'project': project, 'generation': record['generation'],
                                        'recipe': record['active_recipe']})
            return self._send(405, {'error': 'Use POST for this action.'})
        if action not in ('events', 'feedback', 'invoke', 'activate', 'rollback'):
            return self._send(405, {'error': 'Use GET for this resource.'})
        payload = self._body()
        if action == 'events':
            require(set(payload) == {'events'}, 'provide an events array')
            return self._send(201, workspace.ingest(project, payload['events']))
        if action == 'feedback':
            require(set(payload) <= {'trace_id', 'correction', 'event_id'}, 'invalid feedback fields')
            return self._send(201, workspace.feedback(project, payload.get('trace_id'), payload.get('correction'), payload.get('event_id')))
        if action == 'invoke':
            if not self.server.allow_live_models:
                return self._send(403, {'error': 'Start the local server with --allow-live-models to connect model requests.'})
            if not self.server.model_slots.acquire(blocking=False):
                return self._send(429, {'error': 'Connected model request slots are full.'})
            try:
                result = invoke(workspace, project, payload, allow_live=True)
            finally:
                self.server.model_slots.release()
            return self._send(200 if result['status'] == 'completed' else 502, result)
        generation = payload.get('expected_generation')
        require(type(generation) is int and generation >= 0, 'expected_generation must be a nonnegative integer')
        if action == 'activate':
            require(set(payload) == {'round_id', 'expected_generation'}, 'provide round_id and expected_generation')
            record = activate(workspace, project, payload['round_id'], expected_generation=generation)
        else:
            require(set(payload) == {'expected_generation'}, 'provide expected_generation')
            record = workspace.rollback(project, generation)
        return self._send(200, project_summary(record))

    def _dispatch(self, mutation):
        try:
            if not self._origin_allowed():
                return
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            if not mutation and path in STATIC:
                name, mime = STATIC[path]
                return self._send(200, (Path(__file__).parent / 'static' / name).read_bytes(), mime)
            if not mutation and path == '/api/health':
                execution = 'connected_models_enabled' if self.server.allow_live_models else 'recorded_fixtures_only'
                return self._send(200, {'ok': True, 'mode': 'local', 'execution': execution})
            if path.startswith('/api/switchup/'):
                return self._switchup(path, mutation)
            if path == '/api/jobs':
                if mutation:
                    return self._send(202, self.server.jobs.submit(self._body()))
                return self._send(200, {'jobs': self.server.jobs.list()})
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})(?:/(review|cancel|resume)|/artifacts/([a-z-]+\.json))?', path)
            if not match:
                return self._send(404, {'error': 'Route not found.'})
            job_id, action, artifact = match.groups()
            if mutation:
                body = self._body()
                if action == 'review':
                    return self._send(202, self.server.jobs.review(job_id, body))
                if action in ('cancel', 'resume'):
                    if body:
                        raise ValueError('This action expects an empty JSON object.')
                    return self._send(202, getattr(self.server.jobs, action)(job_id))
                return self._send(405, {'error': 'This route does not accept POST.'})
            if action:
                return self._send(405, {'error': 'Use POST for this action.'})
            if artifact:
                return self._send(200, self.server.jobs.artifact(job_id, artifact), filename=artifact if parsed.query == 'download=1' else None)
            return self._send(200, self.server.jobs.get(job_id))
        except KeyError:
            self._send(404, {'error': 'Job or artifact not found.'})
        except (JobConflict, SwitchUpConflict) as error:
            self._send(409, {'error': str(error)})
        except OverflowError as error:
            self._send(413, {'error': str(error)})
        except (ValueError, TypeError, UnicodeError) as error:
            self._send(400, {'error': str(error)[:500]})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return
        except Exception:
            self._send(500, {'error': 'The local server could not complete this request.'})


def create_server(host='127.0.0.1', port=8787, data_dir=None, *, workers=2,
                  switchup_dir=None, allow_live_models=False):
    require(not allow_live_models or host in LOOPBACK, 'connected model requests require a loopback server')
    if data_dir is None:
        data_dir = workspace_dir() / 'platform'
    manager = JobManager(data_dir, workers=workers)
    try:
        workspace = Workspace(switchup_dir or Path(data_dir).expanduser() / 'switchup')
        return PlatformServer((host, port), manager, host, workspace, allow_live_models)
    except BaseException:
        manager.close()
        raise


def serve(host='127.0.0.1', port=8787, data_dir=None, *, switchup_dir=None, allow_live_models=False):
    server = create_server(host, port, data_dir, switchup_dir=switchup_dir, allow_live_models=allow_live_models)
    print(f'SwitchUP local workspace: http://{host}:{server.server_address[1]}')
    print('Connected model requests enabled.' if allow_live_models else 'Recorded evidence and release review enabled.')
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
