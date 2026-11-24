"""SwitchIt! — evidence-based migration of AI workflows between providers.

The harness imports authorized traces, normalizes them into a canonical
store, maps provisional workflows, selects replay cases, replays them
through recorded/offline or live provider adapters, and produces a local
JSON + HTML migration report.

Boundaries kept honest throughout:
- Workflow labels and similarity scores are triage aids, not measured accuracy.
- Replay uses recorded fixtures; configured command runtimes require explicit execution opt-in.
- A historical model response is reference behavior, not proof of success.
- Token provenance stays separated: recorded, estimated, unknown.
"""

__version__ = "0.3.0"

USAGE_BASES = ("recorded", "estimated", "unknown")

__all__ = ["__version__", "core", "providers", "runner", "USAGE_BASES"]


def __getattr__(name):
    """Keep legacy package attributes without loading every pipeline stage."""
    if name in ("core", "providers", "runner"):
        from importlib import import_module

        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
