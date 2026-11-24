"""Expose finite source/harness operations using the CLI's argument contracts."""
import argparse
import contextlib
import io

COMMANDS = {
    'switchit_sources_aegis': ('sources','aegis'),
    'switchit_sources_correlate': ('sources','correlate'),
    'switchit_sources_github': ('sources','github'),
    'switchit_sources_inventory': ('sources','inventory'),
    'switchit_sources_import': ('sources','import'),
    'switchit_harness_propose': ('harness','propose'),
    'switchit_harness_compare': ('harness','compare'),
    'switchit_harness_promote': ('harness','promote'),
    'switchit_harness_rollback': ('harness','rollback'),
    'switchit_harness_status': ('harness','status'),
    'switchit_orchestration_status': ('orchestrate','status'),
    'switchit_orchestration_cancel': ('orchestrate','cancel'),
    'switchup_status': ('improve', 'status'),
    'switchup_ingest': ('improve', 'ingest'),
    'switchup_feedback': ('improve', 'feedback'),
    'switchup_prepare': ('improve', 'prepare'),
    'switchup_propose': ('improve', 'propose'),
    'switchup_verify': ('improve', 'verify'),
    'switchup_show': ('improve', 'show'),
    'switchup_activate': ('improve', 'activate'),
    'switchup_rollback': ('improve', 'rollback'),
    'switchup_active': ('improve', 'active'),
}


def _parsers():
    from .cli import build_parser
    parser = build_parser()
    branches = next(a for a in parser._actions if isinstance(a,argparse._SubParsersAction)).choices
    result = {}
    for name,(branch,child) in COMMANDS.items():
        children = next(a for a in branches[branch]._actions if isinstance(a,argparse._SubParsersAction)).choices
        result[name] = children[child]
    return result


def definitions():
    result=[]
    for name,parser in _parsers().items():
        properties,required={},[]
        for action in parser._actions:
            if action.dest == 'help':
                continue
            kind = ('boolean' if isinstance(action,argparse._StoreTrueAction) else
                    'integer' if action.type is int else 'number' if action.type is float else 'string')
            field={'type':kind,'description':action.help or action.dest.replace('_',' ')}
            if action.choices:
                field['enum']=list(action.choices)
            properties[action.dest]=field
            if action.required:
                required.append(action.dest)
        result.append({'name':name,'description':(parser.description or 'SwitchIt '+ ' '.join(COMMANDS[name]))+'. Uses the same validation and artifacts as the CLI.',
            'inputSchema':{'type':'object','properties':properties,'required':required,'additionalProperties':False}})
    return result


def call(name,arguments):
    parser=_parsers()[name]
    fields={a.dest:a for a in parser._actions if a.dest!='help'}
    if not isinstance(arguments,dict) or set(arguments)-set(fields):
        raise ValueError('unknown tool argument')
    argv=[]
    for key,value in arguments.items():
        action=fields[key]
        flag=action.option_strings[0]
        if isinstance(action,argparse._StoreTrueAction):
            if not isinstance(value,bool):
                raise ValueError(key+' must be a boolean')
            if value:
                argv.append(flag)
        else:
            if isinstance(value,(dict,list,bool)) or value is None:
                raise ValueError(key+' must be a scalar value')
            argv.append(flag+'='+str(value))
    output=io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            from .cli import build_parser
            args=build_parser().parse_args([*COMMANDS[name], *argv])
            code=args.func(args)
    except SystemExit as error:
        raise ValueError(output.getvalue() or str(error)) from error
    if code:
        raise RuntimeError(output.getvalue().strip())
    return output.getvalue().strip()
