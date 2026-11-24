"""Redaction and bounded excerpts for imported messages, tools, and values."""

from __future__ import annotations

import re

MESSAGE_CAP = 8000


EVIDENCE_CAP = 1200


_SECRET_FIELDS = [
    "password",
    "passwd",
    "api[_ -]?key",
    "apikey",
    "secret",
    "token",
    "bearer",
    "authorization",
]


_SECRET_LABEL_PATTERN = re.compile(
    r"(?i)\b(" + "|".join(_SECRET_FIELDS) + r")(\s*[:=]\s*|\s+)[^\s,;\"']+"
)


_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


_AWS_KEY_PATTERN = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")


_SK_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")


_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)")


_WHITESPACE_PATTERN = re.compile(r"\s+")


def redact_text(value) -> str:
    """Redact obvious secrets from free text.

    Emails, phone-like numbers, AWS access keys, ``sk-``-style API keys and
    ``password=``/``bearer``-style assignments are replaced with markers.
    Whitespace is collapsed so excerpts stay compact in reports.
    """
    if value is None:
        return ""
    text = _WHITESPACE_PATTERN.sub(" ", str(value)).strip()
    text = _SECRET_LABEL_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    text = _AWS_KEY_PATTERN.sub("[REDACTED_KEY]", text)
    text = _SK_KEY_PATTERN.sub("[REDACTED_KEY]", text)
    text = _EMAIL_PATTERN.sub("[EMAIL]", text)
    text = _PHONE_PATTERN.sub("[NUMBER]", text)
    return text


def redact_literal(value, limit: int = MESSAGE_CAP) -> str:
    """Redact a freeform tool payload without changing whitespace/code semantics."""
    text = str(value) if value is not None else ""
    text = _SECRET_LABEL_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    text = _AWS_KEY_PATTERN.sub("[REDACTED_KEY]", text)
    text = _SK_KEY_PATTERN.sub("[REDACTED_KEY]", text)
    text = _EMAIL_PATTERN.sub("[EMAIL]", text)
    text = _PHONE_PATTERN.sub("[NUMBER]", text)
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.75))
    tail = max(1, limit - head - 5)
    return text[:head] + " […] " + text[-tail:]


def redact_literal_value(value):
    """Deep redaction for custom input/output blocks whose whitespace is evidence."""
    if isinstance(value, str):
        return redact_literal(value)
    if isinstance(value, list):
        return [redact_literal_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_literal_value(item) for key, item in value.items()}
    return value


def excerpt(text: str, limit: int = MESSAGE_CAP) -> str:
    """Cap text length with head/tail elision so both ends stay visible."""
    text = redact_text(text)
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.75))
    tail = max(1, limit - head - 5)
    return text[:head] + " […] " + text[-tail:]


def redact_value(value):
    """Deep-walk any JSON-ish value, redacting every string inside it."""
    if isinstance(value, str):
        return excerpt(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    return value


def redact_messages(messages) -> list:
    """Normalize message dicts to {role, content} plus optional fields, redacted."""
    clean = []
    for message in messages or []:
        if not isinstance(message, dict):
            clean.append({"role": "user", "content": excerpt(str(message))})
            continue
        entry = {
            "role": str(message.get("role") or "user"),
            "content": excerpt(message.get("content") if message.get("content") is not None else ""),
        }
        for key in ("name", "tool_call_id"):
            if message.get(key):
                entry[key] = redact_text(str(message[key]))[:200]
        if message.get("tool_format") == "custom":
            entry["content"] = redact_literal_value(message.get("content"))
            entry["tool_format"] = "custom"
            if "content_blocks" in message:
                entry["content_blocks"] = redact_literal_value(message["content_blocks"])
            if message.get("output_format"):
                entry["output_format"] = str(message["output_format"])
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            entry["tool_calls"] = [redact_literal_value(call) if isinstance(call, dict) and call.get("type") == "custom"
                                   else redact_value(call) for call in calls]
        clean.append(entry)
    return clean


def redact_tools(tools) -> list:
    """Normalize tool schemas; descriptions are redacted, names are kept short."""
    clean = []
    for tool in tools or []:
        if isinstance(tool, dict):
            if "function" in tool and isinstance(tool["function"], dict):
                tool = tool["function"]
            entry = {
                "name": redact_text(str(tool.get("name") or ""))[:200],
                "description": excerpt(str(tool.get("description") or ""), 400),
                "parameters": redact_value(tool.get("parameters") or {}),
            }
            if entry["name"]:
                clean.append(entry)
        elif isinstance(tool, str):
            name = redact_text(tool)[:200]
            if name:
                clean.append({"name": name, "description": "", "parameters": {}})
    return clean
