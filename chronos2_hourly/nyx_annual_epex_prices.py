"""One explicit EPEX hourly price source for annual training and auction lags.

The selected source is read freshly at the recorded fit cutoff. Missing prices
remain blocking; no second provider, completion formula, fill or numeric
comparison is used. Native market hours are preserved on the UTC timeline.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from chronos2_modular.saturn import fetch_saturn_series_from_client


EPEX_PRICE_POLICY = "epex_only_v1"
EPEX_SERIES = {
    "FR": "power.price.fr.euromwh.h.obs.epex",
    "DE": "power.price.de.euromwh.h.obs.epex",
    "BE": "power.price.be.euromwh.h.obs.epex",
    "NL": "power.price.nl.euromwh.h.obs.epex",
}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def price_identity() -> dict:
    return {"price_reference": "EPEX", "price_reference_policy": EPEX_PRICE_POLICY,
            "source_mixing": False, "price_fallback_policy": "none",
            "actual_values_imputed": False, "price_naive_timezone": "UTC",
            "price_request_padding_hours": 2, "price_nocache": True}


def validate_price_identity(receipt: dict) -> None:
    expected = price_identity()
    _require(all(receipt.get(key) == value for key, value in expected.items()),
             "Annual price source must declare the explicit EPEX-only policy")
    _require(receipt.get("source_mixing") is False
             and receipt.get("actual_values_imputed") is False,
             "Annual EPEX price source cannot mix providers or impute prices")
    _require(not any(key in receipt for key in
        ("equivalent_target_completion", "target_completion_code_sha256")),
        "EPEX-only price receipt cannot declare canonical completion")


class _StrictPriceClient:
    """Reject raw source defects before generic normalization can hide them."""

    def __init__(self, client: Any, expected: pd.DatetimeIndex):
        self._client = client
        self._expected = expected

    def get(self, series_name: str, **kwargs):
        raw = self._client.get(series_name, **kwargs)
        if raw is None:
            return raw
        _require(isinstance(raw, pd.Series)
                 and isinstance(raw.index, pd.DatetimeIndex),
                 f"{series_name}: raw EPEX hourly Series with datetime timestamps required")
        index = raw.index
        _require(not index.hasnans and index.is_unique,
                 f"{series_name}: missing or duplicate raw EPEX timestamps rejected")
        # The existing explicit EPEX contract interprets naive source labels as
        # UTC. Aware labels are already physical instants, including DST folds.
        index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
        _require(bool(((index.minute == 0) & (index.second == 0)
                       & (index.microsecond == 0) & (index.nanosecond == 0)).all()),
                 f"{series_name}: raw EPEX prices must be on the physical-hour UTC grid")
        _require(not np.iscomplexobj(raw.to_numpy()),
                 f"{series_name}: complex EPEX prices rejected")
        selected = pd.to_numeric(raw.to_numpy()[index.isin(self._expected)], errors="coerce")
        _require(np.isfinite(np.asarray(selected, dtype=float)).all(),
                 f"{series_name}: nonnumeric or nonfinite raw EPEX prices rejected")
        return raw


def fetch_epex_prices(client: Any, zone: str, expected: pd.DatetimeIndex,
                      revision_utc: pd.Timestamp, *, fetcher: Callable | None = None) -> pd.Series:
    """Read one complete EPEX price history at the exact supplied revision."""
    _require(zone in EPEX_SERIES, f"{zone}: no explicit annual EPEX hourly price contract")
    _require(isinstance(expected, pd.DatetimeIndex) and len(expected) > 0
             and str(expected.tz) == "UTC" and expected.is_unique
             and expected.is_monotonic_increasing and not expected.hasnans,
             "Annual EPEX prices require a nonempty increasing unique UTC grid")
    _require(expected.equals(pd.date_range(expected[0], periods=len(expected), freq="h"))
             and expected[0] == expected[0].floor("h"),
             "Annual EPEX prices require a continuous physical-hour UTC grid")
    revision = pd.Timestamp(revision_utc)
    _require(not pd.isna(revision) and revision.tzinfo is not None,
             "Annual EPEX prices require an explicit timezone-aware revision")
    revision = revision.tz_convert("UTC")
    fetch = fetch_saturn_series_from_client if fetcher is None else fetcher
    raw = fetch(_StrictPriceClient(client, expected), EPEX_SERIES[zone], expected[0], expected[-1], "UTC",
        revision_date=revision, naive_timezone="UTC", incomplete_dst_policy="raise", nocache=True,
        request_padding_hours=2)
    _require(isinstance(raw, pd.Series) and isinstance(raw.index, pd.DatetimeIndex)
             and str(raw.index.tz) == "UTC" and raw.index.is_unique
             and raw.index.is_monotonic_increasing and not raw.index.hasnans,
             f"{zone}: normalized EPEX source requires unique increasing UTC timestamps")
    _require(bool(((raw.index.minute == 0) & (raw.index.second == 0)
                   & (raw.index.microsecond == 0) & (raw.index.nanosecond == 0)).all()),
             f"{zone}: EPEX source contains subhourly prices")
    selected = pd.to_numeric(raw.reindex(expected), errors="coerce").astype(float)
    missing = ~np.isfinite(selected.to_numpy(dtype=float))
    _require(not bool(missing.any()),
             f"{zone}: EPEX-only {EPEX_SERIES[zone]} has {int(missing.sum())} "
             f"missing/nonfinite hourly prices at cutoff {revision.isoformat()}; first UTC: "
             + ", ".join(stamp.isoformat() for stamp in expected[missing][:5]))
    return selected
