"""Complete canonical annual price snapshots with an audited same-cutoff source.

The canonical response remains authoritative. Only its missing/nonfinite hours
can use a centrally admitted equivalent after a fresh 168-hour compatibility
check. This helper neither reads nor writes a shared latest-price cache.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np
import pandas as pd

from chronos2_modular.saturn import (
    AUDITED_EQUIVALENT_TARGET_FALLBACKS, fetch_saturn_series_from_client,
)


COMPLETION_PROTOCOL = "nyx_annual_audited_target_completion_v1"
PRIMARY_CONTRACT_COMMIT = "6f44ee290df4a2c32adb27ce52297caa5ae3b227"
PRIMARY_CONTRACT_PATH = "nyx_quarterhour/sources.py"
# UTC-aware, null-formula EPEX primaries audited 2026-09-16 in the fixed
# source contract above. These are raw 15-minute observations from 2025-10-01.
# The contract's fetch_range() reads latest labels; it is deliberately not used.
AUDITED_NATIVE_PRICE_PRIMARIES = {
    "power.price.da.fr.bzn.hourly.entsoe.utc.cdh.eurmwh": "60454",
    "power.price.da.de_lu.bzn.hourly.entsoe.utc.cdh.eurmwh": "60452",
    "power.price.da.be.bzn.hourly.entsoe.utc.cdh.eurmwh": "60451",
    "power.price.da.nl.bzn.hourly.entsoe.utc.cdh.eurmwh": "60453",
}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _utc_series(raw: Any, label: str) -> pd.Series:
    _require(isinstance(raw, pd.Series)
             and isinstance(raw.index, pd.DatetimeIndex)
             and str(raw.index.tz) == "UTC"
             and raw.index.is_unique and raw.index.is_monotonic_increasing
             and not raw.index.hasnans,
             f"{label}: exact, unique, increasing UTC source timestamps required")
    _require(bool(((raw.index.minute == 0) & (raw.index.second == 0)
                   & (raw.index.microsecond == 0) & (raw.index.nanosecond == 0)).all()),
             f"{label}: source timestamps must be aligned to physical hours")
    return pd.to_numeric(raw, errors="coerce").astype(float)


def _comparison(canonical: pd.Series, candidate: pd.Series,
                validation: pd.DatetimeIndex, tolerance: float) -> dict:
    paired = pd.DataFrame({"canonical": canonical.reindex(validation),
                           "candidate": candidate.reindex(validation)})
    _require(np.isfinite(paired.to_numpy(dtype=float)).all(),
             "Audited target completion overlap is incomplete")
    difference = (paired.canonical - paired.candidate).abs()
    worst = difference.nlargest(min(5, len(difference)))
    return {"max_abs_difference_eur_mwh": float(difference.max()),
            "mismatched_hours": int((difference > tolerance).sum()),
            "paired_hours": len(validation),
            "worst_hours": [{"timestamp_utc": stamp.isoformat(),
                "canonical_eur_mwh": float(paired.loc[stamp, "canonical"]),
                "candidate_eur_mwh": float(paired.loc[stamp, "candidate"]),
                "abs_difference_eur_mwh": float(delta)}
                for stamp, delta in worst.items()]}


def _detail(comparison: dict) -> str:
    hours = "; ".join(
        f"{row['timestamp_utc']} canonical={row['canonical_eur_mwh']:.12g} "
        f"candidate={row['candidate_eur_mwh']:.12g} "
        f"difference={row['abs_difference_eur_mwh']:.12g}"
        for row in comparison["worst_hours"])
    return (f"{comparison['mismatched_hours']}/{comparison['paired_hours']} "
            f"mismatched overlap hours; max={comparison['max_abs_difference_eur_mwh']:.12g} "
            f"EUR/MWh; worst UTC hours: {hours}")


def _native_hourly_candidate(client: Any, primary_name: str,
                              hours: pd.DatetimeIndex,
                              revision: pd.Timestamp) -> tuple[pd.Series, dict]:
    _require(hours[0] >= pd.Timestamp("2025-10-01", tz="Europe/Paris").tz_convert("UTC"),
             f"Native primary {primary_name}: outside its audited 2025-10-01 price window")
    quarters = pd.date_range(hours[0], hours[-1] + pd.Timedelta(minutes=45), freq="15min")
    try:
        raw = client.get(primary_name, from_value_date=quarters[0],
            to_value_date=quarters[-1], revision_date=revision, nocache=True)
    except Exception as error:
        # Do not retry a latest-vintage request if this client/source cannot
        # serve the required revision, and do not expose connection material.
        raise ValueError(f"Native primary {primary_name}: same-cutoff raw request "
            f"failed at {revision.isoformat()} ({type(error).__name__}); "
            "no latest or alternate revision requested") from None
    _require(isinstance(raw, pd.Series) and isinstance(raw.index, pd.DatetimeIndex)
             and len(raw) > 0 and raw.index.tz is not None and not raw.index.hasnans,
             f"Native primary {primary_name}: nonempty raw timezone-aware quarter prices required")
    index = raw.index.tz_convert("UTC")
    _require(index.is_unique, f"Native primary {primary_name}: duplicate physical quarters rejected")
    _require(bool(((index.minute.isin([0, 15, 30, 45])) & (index.second == 0)
                   & (index.microsecond == 0) & (index.nanosecond == 0)).all()),
             f"Native primary {primary_name}: raw prices must be on the exact 15-minute UTC grid")
    _require(not np.iscomplexobj(raw.to_numpy()),
             f"Native primary {primary_name}: complex prices rejected")
    values = pd.Series(pd.to_numeric(raw.to_numpy(), errors="coerce"),
                       index=index, dtype=float).sort_index()
    selected = values.reindex(quarters)
    finite = np.isfinite(selected.to_numpy(dtype=float))
    _require(bool(finite.all()),
             f"Native primary {primary_name}: {int((~finite).sum())} missing/nonfinite "
             f"quarters at cutoff {revision.isoformat()}; first UTC: "
             + ", ".join(stamp.isoformat() for stamp in quarters[~finite][:5]))
    # The exact reindexed grid proves 00,15,30,45 all exist exactly once.
    # No partial-hour mean, timestamp rounding, repeated value or fill is used.
    hourly = pd.Series(selected.to_numpy(dtype=float).reshape(len(hours), 4).mean(axis=1),
                       index=hours, name=primary_name)
    digest = hashlib.sha256()
    for stamp, price in selected.items():
        digest.update((stamp.isoformat() + " " + float(price).hex() + "\n").encode("ascii"))
    evidence = {
        "completion_method": "native_quarter_hour_mean_v1",
        "native_price_contract_commit": PRIMARY_CONTRACT_COMMIT,
        "native_price_contract_path": PRIMARY_CONTRACT_PATH,
        "native_price_contract_audit_day": "2026-09-16",
        "native_resolution_minutes": 15,
        "aggregation": "arithmetic_mean_of_four_complete_utc_quarters",
        "raw_native_timezone_aware_verified": True,
        "raw_native_unique_timestamps_verified": True,
        "raw_native_quarters_complete_verified": True,
        "native_quarters": len(quarters),
        "native_outside_quarters_excluded": int((~index.isin(quarters)).sum()),
        "native_first_quarter_utc": quarters[0].isoformat(),
        "native_last_quarter_utc": quarters[-1].isoformat(),
        "native_quarter_values_sha256": digest.hexdigest(),
        "native_quarter_values_hash_encoding": "UTC_iso8601_space_float_hex_LF",
    }
    return hourly, evidence


def fetch_complete_target(client: Any, series_name: str,
                          expected: pd.DatetimeIndex, revision_utc: pd.Timestamp,
                          *, fetcher: Callable | None = None
                          ) -> tuple[pd.Series, dict | None]:
    """Return the exact requested UTC grid and any admitted completion evidence.

    Both responses are queried freshly at the identical revision. An empty
    canonical response, an unproven overlap, a divergent equivalent, an
    ambiguous DST fold, or any remaining missing price stays blocking.
    """
    _require(isinstance(expected, pd.DatetimeIndex) and len(expected) > 0
             and str(expected.tz) == "UTC" and expected.is_unique
             and expected.is_monotonic_increasing and not expected.hasnans,
             "Annual target completion: nonempty increasing unique UTC grid required")
    _require(expected.equals(pd.date_range(expected[0], periods=len(expected), freq="h"))
             and expected[0] == expected[0].floor("h"),
             "Annual target completion: continuous physical-hour UTC grid required")
    revision = pd.Timestamp(revision_utc)
    _require(not pd.isna(revision) and revision.tzinfo is not None,
             "Annual target completion: timezone-aware revision required")
    revision = revision.tz_convert("UTC")
    fetch = fetch_saturn_series_from_client if fetcher is None else fetcher
    canonical_raw = fetch(
        client, series_name, expected[0], expected[-1], "UTC",
        revision_date=revision, naive_timezone="UTC",
        incomplete_dst_policy="raise", nocache=True,
    )
    canonical = _utc_series(canonical_raw, series_name).reindex(expected)
    finite = np.isfinite(canonical.to_numpy(dtype=float))
    _require(bool(finite.any()),
             f"{series_name}: canonical response has no finite requested price")
    if bool(finite.all()):
        return canonical, None

    missing = expected[~finite]
    admitted = AUDITED_EQUIVALENT_TARGET_FALLBACKS.get(series_name)
    _require(admitted is not None,
             f"{series_name}: {len(missing)} missing prices; no centrally audited completion source")
    guard_hours = int(admitted.get("validation_hours", 0))
    tolerance = float(admitted.get("atol_eur_mwh", float("nan")))
    _require(guard_hours >= 168 and np.isfinite(tolerance)
             and 0. <= tolerance <= 1e-6,
             f"{series_name}: invalid audited completion validation contract")
    guard = pd.date_range(end=missing[0] - pd.Timedelta(hours=1),
                          periods=guard_hours, freq="h")
    guard_prices = canonical.reindex(guard)
    _require(np.isfinite(guard_prices.to_numpy(dtype=float)).all(),
             f"{series_name}: audited completion requires {guard_hours} fresh finite "
             "canonical hours immediately before its first missing hour")

    # Fetch only the compatibility guard and repair span. Fetching an entire
    # multi-year local-civil history would unnecessarily encounter old folds.
    span = pd.date_range(guard[0], missing[-1], freq="h")
    canonical_span = canonical.reindex(span)
    validation = span[np.isfinite(canonical_span.to_numpy(dtype=float))]
    fallback_name = str(admitted.get("series", "")).strip()
    _require(bool(fallback_name), f"{series_name}: audited completion series absent")
    fallback_raw = fetch(
        client, fallback_name, span[0], span[-1], "UTC",
        revision_date=revision, naive_timezone=str(admitted["naive_timezone"]),
        incomplete_dst_policy="raise", nocache=True,
        request_padding_hours=int(admitted.get("request_padding_hours", 0)),
    )
    fallback = _utc_series(fallback_raw, fallback_name)
    validated_prices = fallback.reindex(validation)
    _require(np.isfinite(validated_prices.to_numpy(dtype=float)).all(),
             f"{series_name}: audited completion overlap is incomplete")
    comparison = _comparison(canonical, fallback, validation, tolerance)
    maximum = comparison["max_abs_difference_eur_mwh"]
    native_evidence = {}
    selected_naive_timezone = str(admitted["naive_timezone"])
    selected_padding = int(admitted.get("request_padding_hours", 0))
    if maximum > tolerance:
        rejected_name, rejected_comparison = fallback_name, comparison
        native_name = AUDITED_NATIVE_PRICE_PRIMARIES.get(series_name)
        rejection = (f"{series_name}: audited completion {rejected_name} differs from canonical "
                     f"prices at cutoff {revision.isoformat()} "
                     f"(max={maximum:.12g} EUR/MWh > {tolerance:.12g}); "
                     + _detail(rejected_comparison))
        _require(native_name is not None, rejection + "; no audited raw native primary")
        try:
            native, native_evidence = _native_hourly_candidate(client, native_name, span, revision)
            comparison = _comparison(canonical, native, validation, tolerance)
            maximum = comparison["max_abs_difference_eur_mwh"]
            _require(maximum <= tolerance,
                     f"Native primary {native_name}: four-quarter hourly means differ from canonical "
                     f"prices at cutoff {revision.isoformat()} "
                     f"(max={maximum:.12g} EUR/MWh > {tolerance:.12g}); "
                     + _detail(comparison))
        except ValueError as error:
            raise ValueError(rejection + "; independent raw native candidate failed: "
                             + str(error)) from None
        fallback, fallback_name = native, native_name
        selected_naive_timezone, selected_padding = "UTC", 0
        native_evidence["rejected_hourly_equivalent"] = {
            "series": rejected_name, "revision_utc": revision.isoformat(),
            "reason": "fresh_canonical_overlap_mismatch", **rejected_comparison}
    supplement = fallback.reindex(missing)
    _require(np.isfinite(supplement.to_numpy(dtype=float)).all(),
             f"{series_name}: audited completion source is incomplete on missing hours")
    completed = canonical.copy()
    completed.loc[missing] = supplement.to_numpy(dtype=float)
    _require(completed.index.equals(expected)
             and np.isfinite(completed.to_numpy(dtype=float)).all()
             and completed.loc[expected[finite]].equals(canonical.loc[expected[finite]]),
             f"{series_name}: completion changed a canonical hour or left a missing price")
    evidence = {
        "protocol": COMPLETION_PROTOCOL,
        "canonical_series": series_name, "completion_series": fallback_name,
        "revision_utc": revision.isoformat(),
        "completion_naive_timezone": selected_naive_timezone,
        "completion_request_padding_hours": selected_padding,
        "incomplete_dst_policy": "raise", "nocache": True,
        "validation_preceding_hours": guard_hours,
        "validation_paired_hours": len(validation),
        "validation_max_abs_difference_eur_mwh": maximum,
        "validation_tolerance_eur_mwh": tolerance,
        "validation_first_hour_utc": validation[0].isoformat(),
        "validation_last_hour_utc": validation[-1].isoformat(),
        "completed_hours": len(missing),
        "completed_value_times_utc": [stamp.isoformat() for stamp in missing],
        "canonical_values_preserved": True, "actual_values_imputed": False,
        "provider_publication_timestamp_verified": False,
        "availability_basis": ("Audited native EPEX quarter-hour primary queried at the original "
            "canonical Saturn revision_date; four-quarter means validated on fresh canonical overlap"
            if native_evidence else
            "Centrally audited equivalent queried freshly at the canonical Saturn revision_date"),
        **native_evidence,
    }
    return completed, evidence
