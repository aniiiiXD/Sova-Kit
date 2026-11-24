"""Compatibility imports for the former core.py implementation.

Existing imports keep working. New code should import the owning modules
listed in docs/module-map.md; implementation lives there, not in this facade.
"""

from switchit.cases import (
    CASES_VERSION,
    ORIGIN_PRIORITY,
    _case_usage_value,
    _seed_call,
    default_checks,
    select_cases,
    save_cases,
    load_cases,
)
from switchit.hierarchy import (
    MAX_USER_REQUESTS_PER_TRACE,
    build_traces,
    build_sessions,
)
from switchit.importers.files import (
    _parse_json_value,
    parse_jsonl_lines,
    _jsonl_record,
    import_jsonl,
    _otlp_attr_value,
    _otlp_attributes,
    _otlp_message_list,
    import_otlp,
    _otlp_latency_ms,
    import_langfuse,
    PARQUET_BASIS_MAP,
    import_parquet,
    IMPORTERS,
)
from switchit.ingestion import (
    import_store,
)
from switchit.normalization import (
    _stable_id,
    normalize_tool_call,
    normalize_response,
    normalize_request,
    make_call,
    _coerce_int,
    _coerce_latency,
)
from switchit.profiling import (
    _json_type,
    infer_tool_schema,
    build_profile,
)
from switchit.redaction import (
    MESSAGE_CAP,
    EVIDENCE_CAP,
    _SECRET_FIELDS,
    _SECRET_LABEL_PATTERN,
    _EMAIL_PATTERN,
    _AWS_KEY_PATTERN,
    _SK_KEY_PATTERN,
    _PHONE_PATTERN,
    _WHITESPACE_PATTERN,
    redact_text,
    excerpt,
    redact_value,
    redact_messages,
    redact_tools,
)
from switchit.storage import (
    STORE_VERSION,
    build_store,
    atomic_write_json,
    save_store,
    load_store,
    digest_file,
    digest_data,
)
from switchit.usage import (
    empty_usage,
    normalize_basis,
    add_usage,
    merge_usage,
    usage_from_record,
)
from switchit.workflow_discovery import (
    WORKFLOW_MAP_VERSION,
    STOPWORDS,
    tokenize,
    TfidfIndex,
    tool_jaccard,
    trace_evidence_level,
    task_document,
    group_workflows,
    _representative_latency,
)
from switchit.workflow_review import (
    save_workflows,
    load_workflows,
    apply_edits,
)

__all__ = [
    "STORE_VERSION",
    "WORKFLOW_MAP_VERSION",
    "CASES_VERSION",
    "MESSAGE_CAP",
    "EVIDENCE_CAP",
    "MAX_USER_REQUESTS_PER_TRACE",
    "redact_text",
    "excerpt",
    "redact_value",
    "redact_messages",
    "redact_tools",
    "empty_usage",
    "normalize_basis",
    "add_usage",
    "merge_usage",
    "usage_from_record",
    "normalize_tool_call",
    "normalize_response",
    "normalize_request",
    "make_call",
    "build_traces",
    "build_sessions",
    "build_store",
    "atomic_write_json",
    "save_store",
    "load_store",
    "digest_file",
    "digest_data",
    "parse_jsonl_lines",
    "import_jsonl",
    "import_otlp",
    "import_langfuse",
    "PARQUET_BASIS_MAP",
    "import_parquet",
    "IMPORTERS",
    "import_store",
    "STOPWORDS",
    "tokenize",
    "TfidfIndex",
    "tool_jaccard",
    "trace_evidence_level",
    "task_document",
    "group_workflows",
    "save_workflows",
    "load_workflows",
    "apply_edits",
    "infer_tool_schema",
    "build_profile",
    "ORIGIN_PRIORITY",
    "default_checks",
    "select_cases",
    "save_cases",
    "load_cases",
]
