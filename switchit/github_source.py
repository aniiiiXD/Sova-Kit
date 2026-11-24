"""Bounded read-only GitHub.com REST collection. No repository code is executed."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .contracts import require
from .importers.github import repository_name

API_VERSION = '2026-03-10'


class GitHubReadError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward an authorization header to redirected storage/hosts.


class GitHubClient:
    def __init__(self, *, token=None, max_pages=3, max_requests=60, timeout=20):
        require(isinstance(max_pages,int) and 1<=max_pages<=20,'max_pages must be 1..20')
        require(isinstance(max_requests,int) and 1<=max_requests<=200,'max_requests must be 1..200')
        self.token=token if token is not None else os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
        self.max_pages,self.max_requests,self.timeout=max_pages,max_requests,timeout
        self.requests=0
        self.opener=urllib.request.build_opener(_NoRedirect())

    def get(self, path, params=None):
        require(path.startswith('/repos/') and '?' not in path and '#' not in path,'invalid GitHub API path')
        if self.requests>=self.max_requests:
            raise GitHubReadError('GitHub request budget exhausted')
        self.requests+=1
        url='https://api.github.com'+path
        if params:url+='?'+urllib.parse.urlencode(params)
        headers={'Accept':'application/vnd.github+json','X-GitHub-Api-Version':API_VERSION,'User-Agent':'switchit-harness'}
        if self.token:headers['Authorization']='Bearer '+self.token
        try:
            with self.opener.open(urllib.request.Request(url,headers=headers),timeout=self.timeout) as response:
                body=response.read(8_000_001)
                if len(body)>8_000_000:raise GitHubReadError('GitHub response exceeded 8 MB limit')
                return json.loads(body),response.headers
        except urllib.error.HTTPError as error:
            code=error.code
            error.close()
            raise GitHubReadError(f'GitHub HTTP {code}; check read access, resource availability, or rate limits') from None
        except (urllib.error.URLError,TimeoutError,ValueError) as error:
            raise GitHubReadError(f'GitHub request failed ({type(error).__name__})') from None

    def pages(self, path, *, key=None, params=None):
        rows=[]
        for page in range(1,self.max_pages+1):
            try:
                data,headers=self.get(path,{**(params or {}),'per_page':100,'page':page})
                values=data.get(key) if key and isinstance(data,dict) else data
                if not isinstance(values,list) or not all(isinstance(v,dict) for v in values):
                    raise GitHubReadError('Unexpected GitHub collection schema')
                rows.extend(values)
            except GitHubReadError as error:
                return rows,{'path':path,'status':'partial' if rows else 'error','records':len(rows),'error':str(error)}
            if 'rel="next"' not in headers.get('Link',''):
                return rows,{'path':path,'status':'complete','records':len(rows)}
        return rows,{'path':path,'status':'limited','records':len(rows),'reason':'page_budget_exhausted'}


def collect_github(repo, *, pr=None, issue=None, run=None, client=None):
    repo=repository_name(repo)
    require(any(value is not None for value in (pr,issue,run)),'select at least one PR, issue, or Actions run')
    for value in (pr,issue,run):
        require(value is None or (isinstance(value,int) and not isinstance(value,bool) and value>0),'resource IDs must be positive integers')
    client=client or GitHubClient()
    base='/repos/'+repo
    bundle={'github_export_version':1,'repository':repo,'scope':{'pr':pr,'issue':issue,'run':run},'coverage':[]}
    def one(path):
        data,_=client.get(path)
        require(isinstance(data,dict),'unexpected GitHub resource schema')
        bundle['coverage'].append({'path':path,'status':'complete','records':1})
        return data
    def many(collection,path,*,key=None,params=None,extra=None):
        rows,coverage=client.pages(path,key=key,params=params)
        bundle['coverage'].append(coverage)
        annotated=[{**row,**(extra or {})} for row in rows]
        bundle.setdefault(collection,[]).extend(annotated)
        return annotated
    if issue is not None:
        bundle['issues']=[one(f'{base}/issues/{issue}')]
    if pr is not None:
        pull=one(f'{base}/pulls/{pr}');bundle['pull_requests']=[pull]
        many('reviews',f'{base}/pulls/{pr}/reviews',extra={'pull_number':pr})
        many('review_comments',f'{base}/pulls/{pr}/comments',extra={'pull_number':pr})
        many('commits',f'{base}/pulls/{pr}/commits',extra={'pull_number':pr})
        # A paginated PR commit response is capped by GitHub, even without a next link.
        if len(bundle.get('commits',[]))>=250:
            bundle['coverage'][-1].update(status='limited',reason='pull_request_commit_endpoint_cap')
        sha=(pull.get('head') or {}).get('sha')
        if sha:
            many('runs',f'{base}/actions/runs',key='workflow_runs',params={'head_sha':sha})
    for number in sorted({n for n in (pr,issue) if n is not None}):
        many('comments',f'{base}/issues/{number}/comments',extra={'issue_number':number})
    if run is not None:
        selected=one(f'{base}/actions/runs/{run}')
        bundle['runs']=[r for r in bundle.get('runs',[]) if r.get('id')!=selected.get('id')]+[selected]
    for execution in bundle.get('runs',[]):
        run_id=execution.get('id');attempt=execution.get('run_attempt',1)
        require(isinstance(run_id,int) and run_id>0 and isinstance(attempt,int) and attempt>0,'invalid GitHub run identity')
        many('jobs',f'{base}/actions/runs/{run_id}/attempts/{attempt}/jobs',key='jobs',extra={'run_id':run_id,'run_attempt':attempt})
    return bundle
