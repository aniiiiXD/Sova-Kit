"""MCP definitions for workflow APIs, sharing the actual CLI argument parser."""
import contextlib
import io
import json


def definitions():
    from .mcp_server import _schema, _str, _bool, _int
    definitions = [
        ('build', 'Reconstruct trace hierarchy and workflow drafts from an event or calls file.',
         {'events': _str('Input events/store JSON path.'), 'output': _str('Artifact directory.')}, ['events', 'output']),
        ('review', 'Record explicitly supplied reviewer, outcome checks and optional bindings on a draft.',
         {k: _str(v) for k, v in {'workflow':'Draft JSON path.', 'workflow_id':'ID to select from drafts.',
          'checks':'JSON array file of required checks.', 'reviewer':'Review attribution.',
          'bindings':'Optional JSON contract edits.', 'output':'Reviewed workflow output path.'}.items()},
         ['workflow', 'checks', 'reviewer', 'output']),
        ('cases', 'Create whole-workflow cases with exact fixtures and session-separated holdout.',
         {**{k: _str(k+' JSON path.') for k in ('workflow','graph','candidates','output')},
          'holdout_fraction': {'type':'number','exclusiveMinimum':0,'exclusiveMaximum':1}},
         ['workflow','graph','candidates','output']),
        ('run', 'Execute one reviewed workflow case and evaluate fixed criteria. Defaults to offline fixtures.',
         {**{k: _str(k+' value or file path.') for k in ('workflow','case','case_id','output','provider','model','recipe','command_config')},
          'runtime': {'type':'string','enum':['fixture','provider','command']},
          'remote': _bool('Explicit remote runtime opt-in.'),
          'allow_execution': _bool('Explicit configured command/business-tool opt-in.'),
          'max_seconds': {'type':'number','exclusiveMinimum':0}, 'max_tokens': _int('Whole-run token budget.')},
         ['workflow','case','output']),
        ('export_skill', 'Export a reviewed workflow as a skill; optional cases remain private local evidence.',
         {k: _str(k+' value or path.') for k in ('workflow','output','name','cases')}, ['workflow','output']),
    ]
    return [{'name':'switchit_workflow_'+name,'description':description,'inputSchema':_schema(properties,required)}
            for name,description,properties,required in definitions]


def call(name, arguments):
    from .cli import build_parser
    schema = next(d['inputSchema'] for d in definitions() if d['name'] == name)
    if not isinstance(arguments, dict) or set(arguments) - set(schema['properties']):
        raise ValueError('unknown workflow tool arguments')
    if any(key not in arguments for key in schema['required']):
        raise ValueError('missing required workflow tool argument')
    argv = ['workflow', name.removeprefix('switchit_workflow_').replace('_','-')]
    for key, value in arguments.items():
        if value is None:
            continue
        expected = schema['properties'][key]['type']
        valid = {'boolean': isinstance(value,bool), 'string': isinstance(value,str),
                 'number': isinstance(value,(int,float)) and not isinstance(value,bool),
                 'integer': isinstance(value,int) and not isinstance(value,bool)}[expected]
        if not valid:
            raise ValueError('invalid type for '+key)
        flag = '--'+key.replace('_','-')
        if expected == 'boolean':
            if value:
                argv.append(flag)
        else:
            argv.append(flag+'='+str(value))
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            args = build_parser().parse_args(argv)
            code = args.func(args)
    except SystemExit as error:
        raise ValueError(buffer.getvalue() or str(error)) from error
    if code:
        # Keep the evaluation's artifact path and verdict available to callers.
        raise RuntimeError(buffer.getvalue().strip())
    return buffer.getvalue().strip()
