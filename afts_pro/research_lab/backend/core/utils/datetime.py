"""Datetime helpers for the research backend."""

from __future__ import annotations

from datetime import datetime, timezone


def ensure_utc_datetime(value: datetime | str) -> datetime:
    """Ensure datetime values are timezone-aware in UTC."""

    if isinstance(value, str):
        iso_value = value.replace("Z", "+00:00") if value.endswith("Z") else value
        parsed = datetime.fromisoformat(iso_value)
    else:
        parsed = value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


__all__ = ["ensure_utc_datetime"]
