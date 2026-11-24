"""Compatibility imports for the former runner.py implementation.

Existing imports keep working. New code should import the owning modules
listed in docs/module-map.md; implementation lives there, not in this facade.
"""

from switchit._run_utils import (
    utc_now_iso,
    clip,
)
from switchit.aggregation import (
    _merge_usage_results,
    aggregate,
    _ids_for_capability,
)
from switchit.checkpoints import (
    RUNNER_VERSION,
    run_fingerprint,
    load_checkpoint,
    _checkpoint_path,
)
from switchit.evaluation import (
    VERDICTS,
    CHECK_TYPES,
    _tool_call_names,
    _deep_subset,
    evaluate_check,
    evaluate_checks,
    historical_result,
    compare_tools,
    finalize_case,
)
from switchit.replay import (
    MAX_CHAIN_TURNS,
    _chain_target_turns,
    replay_case,
    run_replay,
)
from switchit.report_html import (
    render_html,
)
from switchit.reporting import (
    BOUNDARIES,
    build_report,
    write_report,
)
from switchit.storage import (
    atomic_write_json,
)

__all__ = [
    "RUNNER_VERSION",
    "VERDICTS",
    "MAX_CHAIN_TURNS",
    "BOUNDARIES",
    "CHECK_TYPES",
    "utc_now_iso",
    "atomic_write_json",
    "run_fingerprint",
    "clip",
    "evaluate_check",
    "evaluate_checks",
    "historical_result",
    "compare_tools",
    "replay_case",
    "finalize_case",
    "load_checkpoint",
    "run_replay",
    "aggregate",
    "build_report",
    "write_report",
    "render_html",
]
