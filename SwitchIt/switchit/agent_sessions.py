"""Compatibility imports for the former agent_sessions.py implementation.

Existing imports keep working. New code should import the owning modules
listed in docs/module-map.md; implementation lives there, not in this facade.
"""

from switchit.importers.aider import (
    AIDER_HEADER_PATTERN,
    import_aider,
)
from switchit.importers.claude_code import (
    import_claude_code,
)
from switchit.importers.cline import (
    import_cline,
    _cline_sdk_messages_file,
    _cline_result_text,
    _cline_legacy_task,
)
from switchit.importers.codex import (
    import_codex,
)
from switchit.importers.common import (
    REDACT_MARKER,
    _parse_time,
    _open_text,
    _iter_jsonl,
    _content_text,
    _usage_recorded,
    _tool_result_call,
    _safe_json,
    _gemini_text,
)
from switchit.importers.copilot_cli import (
    import_copilot_cli,
)
from switchit.importers.cursor import (
    import_cursor,
    _cursor_assistant_content,
)
from switchit.importers.gemini_cli import (
    import_gemini_cli,
)
from switchit.importers.grok_build import (
    import_grok_build,
)
from switchit.importers.muse import (
    import_muse,
    import_muse_tree,
)
from switchit.importers.opencode import (
    import_opencode,
)
from switchit.importers.registry import (
    IMPORTERS_BY_AGENT,
)
from switchit.session_catalog import (
    codex_thread_catalog,
    catalog_stub_calls,
    collect_session_meta,
)
from switchit.source_discovery import (
    _VSCODE_GLOBAL_STORAGE_CANDIDATES,
    AGENT_DEFAULTS,
)

from switchit import source_discovery
from switchit.importers import registry


def discover_agent_sessions(kinds=None, limit: int = 200) -> list:
    """Discover sessions while honoring legacy AGENT_DEFAULTS overrides."""
    return source_discovery.discover_agent_sessions(kinds, limit, defaults=AGENT_DEFAULTS)


def import_agent_sessions(sessions: list) -> list:
    """Import sessions while honoring legacy IMPORTERS_BY_AGENT overrides."""
    return registry.import_agent_sessions(sessions, importers=IMPORTERS_BY_AGENT)

__all__ = [
    "REDACT_MARKER",
    "import_claude_code",
    "import_codex",
    "import_cursor",
    "import_opencode",
    "import_cline",
    "AIDER_HEADER_PATTERN",
    "import_aider",
    "import_gemini_cli",
    "import_copilot_cli",
    "import_muse",
    "import_muse_tree",
    "import_grok_build",
    "AGENT_DEFAULTS",
    "IMPORTERS_BY_AGENT",
    "discover_agent_sessions",
    "codex_thread_catalog",
    "catalog_stub_calls",
    "collect_session_meta",
    "import_agent_sessions",
]
