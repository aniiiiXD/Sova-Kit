"""Check recursive project-local worker imports before allocating a GPU."""
import ast
from pathlib import Path


def verify_worker_imports(files,entry):
    pending=[entry];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        if name not in files:raise ValueError('Missing worker dependency: '+name)
        tree=ast.parse(Path(files[name]).read_text(),filename=name)
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[alias.name for alias in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    package=name[:-3].split('/')[:-1]
                    package=package[:len(package)-node.level+1]
                    modules=['.'.join([*package,*((node.module or '').split('.')if node.module else [])])]
                elif node.module:modules=[node.module]
            for module in modules:
                if module.split('.')[0]not in('cloud','kvverify'):continue
                dependency=module.replace('.','/')+'.py'
                if dependency not in files:raise ValueError('Missing worker dependency: '+dependency+' (from '+name+')')
                pending.append(dependency)
    return sorted(seen)
