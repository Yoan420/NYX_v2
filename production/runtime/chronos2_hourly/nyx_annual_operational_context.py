"""Explicit, process-local snapshot contract for an operational annual run.

The logical recipe origin remains capped at D-1 08:00 Paris.  Public responses
may be retrieved later, with their real timestamps and an unqualified marker;
this context never establishes historical point-in-time availability.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, timedelta
import json
import os

import pandas as pd


ENV = "NYX_ANNUAL_OPERATIONAL_CONTEXT"
PROTOCOL = "nyx_annual_operational_snapshot_v1"
FIELDS = {"protocol", "delivery_day", "requested_at_utc", "asof_utc",
          "nominal_cutoff_utc"}


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _utc(value, label: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Operational snapshot: invalid {label}") from error
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"Operational snapshot: timezone-aware {label} required")
    return stamp.tz_convert("UTC")


def _day(value) -> date:
    try:
        day = date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Operational snapshot: delivery day must be YYYY-MM-DD") from error
    if day.isoformat() != value:
        raise ValueError("Operational snapshot: delivery day must be YYYY-MM-DD")
    return day


def _nominal(day: date) -> pd.Timestamp:
    return pd.Timestamp(f"{day - timedelta(days=1)} 08:00",
                        tz="Europe/Paris").tz_convert("UTC")


def _validate(context: object, *, now_utc=None) -> dict:
    if (not isinstance(context, dict) or set(context) != FIELDS
            or context.get("protocol") != PROTOCOL):
        raise ValueError("Operational snapshot: context identity differs")
    day = _day(context["delivery_day"])
    requested = _utc(context["requested_at_utc"], "request time")
    asof = _utc(context["asof_utc"], "as-of time")
    nominal = _utc(context["nominal_cutoff_utc"], "nominal cutoff")
    now = _now() if now_utc is None else _utc(now_utc, "current time")
    if day != requested.tz_convert("Europe/Paris").date() + timedelta(days=1):
        raise ValueError("Operational snapshot: delivery must be tomorrow at request time in Paris")
    if nominal != _nominal(day) or asof != min(requested, nominal):
        raise ValueError("Operational snapshot: as-of time or nominal cutoff differs")
    if requested > now or asof > now:
        raise ValueError("Operational snapshot: future request or as-of time forbidden")
    expected = {"protocol": PROTOCOL, "delivery_day": day.isoformat(),
                "requested_at_utc": requested.isoformat(),
                "asof_utc": asof.isoformat(),
                "nominal_cutoff_utc": nominal.isoformat()}
    if context != expected:
        raise ValueError("Operational snapshot: canonical UTC context required")
    return expected


def make_context(delivery_day: str, now_utc=None) -> dict:
    """Fix the launch timestamp once; only tomorrow in Paris may be launched."""
    day = _day(delivery_day)
    now = _now() if now_utc is None else _utc(now_utc, "request time")
    nominal = _nominal(day)
    context = {"protocol": PROTOCOL, "delivery_day": day.isoformat(),
               "requested_at_utc": now.isoformat(),
               "asof_utc": min(now, nominal).isoformat(),
               "nominal_cutoff_utc": nominal.isoformat()}
    return _validate(context, now_utc=now)


def operational_context(day: str | None = None) -> dict | None:
    """Read and validate the inherited context; historical days stay strict."""
    raw = os.environ.get(ENV)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("Operational snapshot: invalid environment context JSON") from error
    context = _validate(value)
    if day is not None and _day(day).isoformat() != context["delivery_day"]:
        return None
    return context


def effective_cutoff(day: str, canonical: pd.Timestamp) -> pd.Timestamp:
    context = operational_context(day)
    return pd.Timestamp(context["asof_utc"]) if context else canonical


def is_operational(day: str | None = None) -> bool:
    return operational_context(day) is not None


def metadata(day: str) -> dict:
    context = operational_context(day)
    return {"operational_snapshot": context} if context else {}


@contextmanager
def context_scope(context: dict):
    """Inherit one immutable context in child processes and restore on exit."""
    context = _validate(context)
    previous = os.environ.get(ENV)
    os.environ[ENV] = json.dumps(context, sort_keys=True, separators=(",", ":"))
    try:
        yield context.copy()
    finally:
        if previous is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = previous
