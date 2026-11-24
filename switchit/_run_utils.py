"""Small formatting and time helpers shared by replay and reporting."""

from __future__ import annotations

import datetime


def utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clip(text, limit=600):
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: int(limit * 0.75)] + " […] " + text[-int(limit * 0.25) :]
