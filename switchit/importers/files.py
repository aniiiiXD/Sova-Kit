"""Parse JSON/JSONL, OTLP, Langfuse, and optional Parquet trace exports."""

from __future__ import annotations

from pathlib import Path
import json
import re


def _parse_json_value(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return default
    return value if value is not None else default


def parse_jsonl_lines(lines, source: str) -> list:
    """Parse generic JSON/JSONL records. Accepts flexible key spellings."""
    raw_calls = []
    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON ({error})") from error
        if isinstance(record, list):
            for item in record:
                raw_calls.append(_jsonl_record(item, f"{source}:{line_number}"))
        else:
            raw_calls.append(_jsonl_record(record, f"{source}:{line_number}"))
    return raw_calls


def _jsonl_record(record: dict, location: str) -> dict:
    if not isinstance(record, dict):
        raise ValueError(f"{location}: record is not an object")
    request = record.get("request", record.get("input"))
    if request is None:
        messages = _parse_json_value(record.get("messages", record.get("input_messages")), None)
        if messages is None:
            messages = _parse_json_value(record.get("prompt"), None)
            request = {"prompt": messages} if isinstance(messages, str) else (messages or [])
        else:
            request = {"messages": messages, "tools": _parse_json_value(record.get("tools"), [])}
    response = record.get("response", record.get("output"))
    if response is None:
        output_messages = _parse_json_value(record.get("output_messages"), None)
        if output_messages is not None:
            response = {"messages": output_messages}
        else:
            text = record.get("response_text", record.get("output_text"))
            response = (
                {"text": text, "tool_calls": _parse_json_value(record.get("tool_calls"), []),
                 "finish_reason": record.get("finish_reason"), "error": record.get("error")}
                if text is not None or record.get("tool_calls") else None
            )
    return {
        "trace_id": record.get("trace_id", record.get("traceId", record.get("conversation_id"))),
        "call_id": record.get("call_id", record.get("id")),
        "session_id": record.get("session_id", record.get("sessionId", record.get("session"))),
        "span_id": record.get("span_id", record.get("spanId")),
        **{key: record[key] for key in ("timestamp", "parent_id", "parentId", "parentUuid", "parent_span_id",
           "parentSpanId", "agent_id", "parent_agent_id", "retry_of", "handoff_from", "context") if key in record},
        "call_index": record.get("call_index", record.get("index")),
        "provider": record.get("provider"),
        "model": record.get("model"),
        "request": request,
        "response": response,
        "latency_ms": record.get("latency_ms"),
        "error": record.get("error"),
        "input_tokens": record.get("input_tokens"),
        "output_tokens": record.get("output_tokens"),
        "usage": record.get("usage"),
        "usage_basis": record.get("usage_basis"),
        "usage_estimator": record.get("usage_estimator", record.get("estimator")),
    }


def import_jsonl(path) -> list:
    """Import JSON or JSONL call records from a file path."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as stream:
        return parse_jsonl_lines(stream, str(path))


def _otlp_attr_value(value: dict):
    if not isinstance(value, dict):
        return None
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            return value[key]
    if "arrayValue" in value and isinstance(value["arrayValue"], dict):
        return [item.get("stringValue") for item in value["arrayValue"].get("values", [])]
    return None


def _otlp_attributes(span: dict) -> dict:
    attributes = {}
    for attribute in span.get("attributes", []) or []:
        key = attribute.get("key")
        if key:
            attributes[key] = _otlp_attr_value(attribute.get("value"))
    return attributes


def _otlp_message_list(attributes: dict, prefix: str) -> list:
    """Read gen_ai.prompt.{n}.role/.content style attributes into messages."""
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.(role|content)$")
    messages = {}
    for key, value in attributes.items():
        match = pattern.match(key)
        if not match:
            continue
        entry = messages.setdefault(int(match.group(1)), {})
        if match.group(2) == "role":
            entry["role"] = value
        elif isinstance(value, list):
            entry["content"] = " ".join(str(item) for item in value if item is not None)
        else:
            entry["content"] = value
    return [messages[index] for index in sorted(messages)]


def import_otlp(path) -> list:
    """Import OpenTelemetry/OTLP JSON spans using gen_ai semantic conventions."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    raw_calls = []
    resource_spans = document.get("resourceSpans") or []
    for resource_index, resource in enumerate(resource_spans):
        resource_attributes = _otlp_attributes(resource.get("resource") or {})
        for scope in resource.get("scopeSpans", []) or []:
            for span_index, span in enumerate(scope.get("spans", []) or []):
                attributes = _otlp_attributes(span)
                merged = {**resource_attributes, **attributes}
                gen_ai_keys = [key for key in merged if key.startswith("gen_ai.")]
                if not gen_ai_keys:
                    continue
                messages = []
                for prefix in ("gen_ai.input", "gen_ai.prompt", "gen_ai.request"):
                    candidate = _otlp_message_list(merged, prefix)
                    if candidate:
                        messages = candidate
                        break
                output_messages = _otlp_message_list(merged, "gen_ai.output")
                if not output_messages:
                    output_messages = _otlp_message_list(merged, "gen_ai.completion")
                error = None
                status = span.get("status") or {}
                if str(status.get("code", "")) not in ("", "1", "STATUS_CODE_UNSET", "UNSET"):
                    error = status.get("message") or "span status error"
                tool_definitions = []
                for key, value in merged.items():
                    if key.startswith("gen_ai.tool.definition") and isinstance(value, str):
                        try:
                            tool_definitions.append(json.loads(value))
                        except ValueError:
                            tool_definitions.append({"name": value})
                raw_calls.append(
                    {
                        "trace_id": span.get("traceId"),
                        "span_id": span.get("spanId"),
                        "parent_span_id": span.get("parentSpanId"),
                        "timestamp": span.get("startTimeUnixNano"),
                        "call_index": span_index,
                        "provider": merged.get("gen_ai.system"),
                        "model": merged.get("gen_ai.request.model", merged.get("gen_ai.response.model")),
                        "session_id": merged.get("session.id", merged.get("gen_ai.session.id")),
                        "request": {"messages": messages, "tools": tool_definitions},
                        "response": {
                            "messages": output_messages,
                            "finish_reason": merged.get("gen_ai.response.finish_reasons"),
                            "error": error,
                        },
                        "latency_ms": _otlp_latency_ms(span),
                        "input_tokens": merged.get("gen_ai.usage.input_tokens",
                                                   merged.get("gen_ai.usage.prompt_tokens")),
                        "output_tokens": merged.get("gen_ai.usage.output_tokens",
                                                    merged.get("gen_ai.usage.completion_tokens")),
                        "usage_basis": "recorded" if (
                            merged.get("gen_ai.usage.input_tokens") is not None
                            or merged.get("gen_ai.usage.output_tokens") is not None
                        ) else "unknown",
                    }
                )
    return raw_calls


def _otlp_latency_ms(span: dict):
    try:
        start = int(span.get("startTimeUnixNano"))
        end = int(span.get("endTimeUnixNano"))
    except (TypeError, ValueError):
        return None
    return max(0, (end - start) // 1_000_000)


def import_langfuse(path) -> list:
    """Import a Langfuse-style trace export (traces with observations)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    raw_calls = []
    traces = document.get("data") if isinstance(document, dict) else document
    for trace in traces or []:
        trace_id = trace.get("id")
        session_id = trace.get("sessionId") or trace.get("session_id")
        observations = trace.get("observations") or []
        generations = [o for o in observations if str(o.get("type", "")).upper() == "GENERATION"]
        if not generations and observations:
            generations = observations
        for index, observation in enumerate(generations):
            usage = observation.get("usage") or {}
            input_tokens = usage.get("input")
            output_tokens = usage.get("output")
            basis = "recorded" if (input_tokens is not None or output_tokens is not None) else "unknown"
            error = None
            if str(observation.get("level", "")).upper() == "ERROR":
                error = observation.get("statusMessage") or "observation level ERROR"
            latency_ms = None
            try:
                from datetime import datetime

                start = datetime.fromisoformat(str(observation.get("startTime")).replace("Z", "+00:00"))
                end = datetime.fromisoformat(str(observation.get("endTime")).replace("Z", "+00:00"))
                latency_ms = max(0, int((end - start).total_seconds() * 1000))
            except (ValueError, TypeError):
                latency_ms = None
            model_parameters = observation.get("modelParameters")
            if not isinstance(model_parameters, dict):
                model_parameters = {}
            raw_calls.append(
                {
                    "trace_id": trace_id,
                    "call_id": observation.get("id"),
                    "parent_id": observation.get("parentObservationId"),
                    "timestamp": observation.get("startTime"),
                    "session_id": session_id,
                    "call_index": index,
                    "provider": observation.get("provider") or model_parameters.get("provider"),
                    "model": observation.get("model"),
                    "request": observation.get("input"),
                    "response": observation.get("output"),
                    "latency_ms": latency_ms,
                    "error": error,
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "basis": basis},
                }
            )
    return raw_calls


PARQUET_BASIS_MAP = {
    "source_reported": "recorded",
    "reported": "recorded",
    "recorded": "recorded",
}


def import_parquet(path) -> list:
    """Import the Parquet trace shape; requires the optional pyarrow package."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "Parquet import requires pyarrow. Install it with: pip install pyarrow"
        ) from error
    path = Path(path)
    if path.suffix.lower() == ".zip":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            matches = [name for name in archive.namelist() if Path(name).name == "traces.parquet"]
            if len(matches) != 1:
                raise ValueError("ZIP must contain exactly one traces.parquet")
            data = archive.read(matches[0])
        parquet_file = pq.ParquetFile(pa.BufferReader(data))
    else:
        parquet_file = pq.ParquetFile(path)
    columns = [
        "trace_id", "span_id", "call_index", "input_tokens", "output_tokens",
        "usage_basis", "input_messages_json", "output_messages_json", "extra_json",
    ]
    raw_calls = []
    for batch in parquet_file.iter_batches(batch_size=200, columns=columns):
        for row in batch.to_pylist():
            extra = _parse_json_value(row.get("extra_json"), {})
            if not isinstance(extra, dict):
                extra = {}
            messages = _parse_json_value(row.get("input_messages_json"), [])
            if isinstance(extra.get("system_instructions"), str) and extra["system_instructions"].strip():
                messages = [{"role": "system", "content": extra["system_instructions"]}] + list(messages or [])
            error = extra.get("error")
            raw_calls.append(
                {
                    "trace_id": row.get("trace_id"),
                    "span_id": row.get("span_id"),
                    "call_index": row.get("call_index"),
                    "model": extra.get("model"),
                    "provider": extra.get("provider"),
                    "session_id": extra.get("session_id"),
                    "request": {"messages": messages},
                    "response": {"messages": _parse_json_value(row.get("output_messages_json"), []),
                                 "error": error,
                                 "finish_reason": extra.get("finish_reason")},
                    "latency_ms": extra.get("latency_ms"),
                    "input_tokens": row.get("input_tokens"),
                    "output_tokens": row.get("output_tokens"),
                    "usage_basis": PARQUET_BASIS_MAP.get(str(row.get("usage_basis") or "").lower(),
                                                         row.get("usage_basis") or "unknown"),
                    "usage_estimator": "cl100k_base"
                    if str(row.get("usage_basis") or "").startswith("estimated") else None,
                }
            )
    return raw_calls


IMPORTERS = {
    "jsonl": import_jsonl,
    "otlp": import_otlp,
    "langfuse": import_langfuse,
    "parquet": import_parquet,
}
