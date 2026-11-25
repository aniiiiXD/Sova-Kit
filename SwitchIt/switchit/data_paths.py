"""Resolve Sova-Kit artifacts separately from the source checkout."""

import os
from pathlib import Path


def kit_data_dir(kit_root=None):
    configured = os.environ.get('SOVA_KIT_DATA_DIR')
    if configured:
        return Path(configured).expanduser().resolve()
    if kit_root is None:
        current = Path.cwd()
        source = Path(__file__).resolve().parents[2]
        kit_root = next((path for path in (current, *current.parents, source)
                         if (path / 'SwitchIt' / 'pyproject.toml').is_file()
                         and (path / 'SwitchUp' / 'LoRA-Route').is_dir()), None)
    if kit_root is not None:
        return kit_root.with_name(f'{kit_root.name}-data')
    return None


def workspace_dir():
    root = kit_data_dir()
    return root / 'workspace' if root is not None else Path('.switchit')
