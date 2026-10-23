"""Render existing annual outputs with the team's established Plotly report.

The annual final forecast and the historical CPU baseline are two different
models.  They are deliberately displayed in separately identified sections.
No estimator is loaded, fitted or called by this module.
"""
from __future__ import annotations

from datetime import timedelta
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd


TIMEZONES = {"FR": "Europe/Paris", "DE": "Europe/Berlin",
             "BE": "Europe/Brussels", "NL": "Europe/Amsterdam"}
BASELINE_LABEL = "NYX"
FINAL_LABEL = "NYX"
INTERVAL_LABEL = (
    "Intervalle de prévision indicatif (80 % nominal) — enveloppe de référence ; "
    "non calibré pour la prévision finale"
)


def _same_instants(left: pd.DatetimeIndex, right: pd.DatetimeIndex) -> bool:
    """Compare physical timestamps without treating s/ns dtype as identity."""
    return len(left) == len(right) and all(a == b for a, b in zip(left, right))


def _utc_frame(value: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name}: a pandas DataFrame is required")
    result = value.copy(deep=True)
    if "delivery_start_utc" in result:
        raw = result.pop("delivery_start_utc")
    elif "timestamp" in result:
        raw = result.pop("timestamp")
    elif isinstance(result.index, pd.DatetimeIndex):
        raw = result.index
    else:
        raise ValueError(f"{name}: an explicit physical-hour timestamp is required")
    stamps = [pd.Timestamp(item) for item in raw]
    if any(pd.isna(item) or item.tzinfo is None for item in stamps):
        raise ValueError(f"{name}: UTC-aware timestamps are required")
    index = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True), name="delivery_start_utc")
    if index.has_duplicates or not _same_instants(index, index.floor("h")):
        raise ValueError(f"{name}: duplicated or off-hour physical timestamps")
    if result.columns.has_duplicates:
        raise ValueError(f"{name}: duplicated columns")
    result.index = index
    return result.sort_index()


def _utc_series(value: pd.Series | pd.DataFrame | None, name: str) -> pd.Series:
    if value is None:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name=name)
    if isinstance(value, pd.Series):
        frame = value.rename("value").to_frame()
        column = "value"
    elif isinstance(value, pd.DataFrame):
        frame = value
        columns = [item for item in ("actual", "observed", "storm", "value", "price_eur_mwh",
                                    "storm_dashboard_official__q50") if item in frame]
        if len(columns) != 1:
            raise ValueError(f"{name}: supply a Series or one unambiguous price column")
        column = columns[0]
    else:
        raise TypeError(f"{name}: pandas Series or DataFrame required")
    frame = _utc_frame(frame, name)
    values = pd.to_numeric(frame[column], errors="raise").astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError(f"{name}: infinite prices")
    return values.rename(name)


def _grid(day: str, timezone: str) -> pd.DatetimeIndex:
    parsed = pd.Timestamp(day)
    if parsed.tzinfo is not None or parsed.isoformat()[:10] != day or len(day) != 10:
        raise ValueError("day must use YYYY-MM-DD")
    start = pd.Timestamp(parsed.date(), tz=timezone)
    stop = pd.Timestamp(parsed.date() + timedelta(days=1), tz=timezone)
    return pd.date_range(start, stop, freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")


def _baseline_prediction(raw: pd.DataFrame, actual: pd.Series, timezone: str,
                         *, issued_origin: Any = None,
                         allow_origin_fallback: bool = True) -> pd.DataFrame:
    """Expand genuine CPU quantiles for the existing display-only renderer."""
    low, middle, high = [pd.to_numeric(raw[f"nyx__q{level}"], errors="raise").to_numpy(float)
                         for level in (10, 50, 90)]
    values = np.column_stack([low, middle, high])
    if not np.isfinite(values).all() or (np.diff(values, axis=1) < 0).any():
        raise ValueError("CPU baseline quantiles are missing, non-finite or crossed")
    local = raw.index.tz_convert(timezone)
    # An explicit stored origin is preferable; otherwise the annual contract
    # is the previous civil day's 08:00, including on DST transition days.
    origin_name = next((name for name in ("nyx_forecast_origin_utc", "forecast_origin_utc")
                        if name in raw and raw[name].notna().any()), None)
    if issued_origin is not None:
        cutoff = pd.Timestamp(issued_origin)
        if pd.isna(cutoff) or cutoff.tzinfo is None:
            raise ValueError("An explicit timezone-aware forecast origin is required")
        cutoff = cutoff.tz_convert(timezone)
        origin = pd.DatetimeIndex([cutoff] * len(raw))
        if origin_name:
            stored = pd.DatetimeIndex(pd.to_datetime(raw[origin_name], utc=True, errors="raise"))
            expected_origin = origin.tz_convert("UTC")
            if stored.hasnans or not _same_instants(stored, expected_origin):
                raise ValueError("Stored CPU future origins differ from the common forecast cutoff")
    elif origin_name:
        origin = pd.DatetimeIndex(pd.to_datetime(raw[origin_name], utc=True, errors="raise")).tz_convert(timezone)
        if origin.hasnans:
            raise ValueError("CPU baseline forecast origins are missing")
    else:
        if not allow_origin_fallback:
            raise ValueError("A two-day forecast requires an explicit common issued origin")
        origin = pd.DatetimeIndex([pd.Timestamp(stamp.date() - timedelta(days=1), tz=timezone)
                                  + pd.Timedelta(hours=8) for stamp in local])
    result = pd.DataFrame({"timestamp": local, "origin_timestamp": origin,
                           "actual": actual.reindex(raw.index).to_numpy(float), "point": middle})
    for level in range(10, 100, 10):
        if level <= 50:
            result[f"q{level}"] = low + ((level - 10) / 40.) * (middle - low)
        else:
            result[f"q{level}"] = middle + ((level - 50) / 40.) * (high - middle)
    result["horizon_step"] = result.groupby(result["timestamp"].dt.date, sort=False).cumcount() + 1
    return result


def _number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "—"


def _plot(fig: Any, identifier: str) -> str:
    """Wait for the shared renderer's embedded Plotly before drawing above it."""
    raw = fig.to_json().replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return (f'<div id="{identifier}" class="js-plotly-plot" style="width:100%;min-height:340px"></div>'
            f'<script id="{identifier}-data" type="application/json">{raw}</script>'
            '<script>(()=>{'
            f'const id="{identifier}";'
            'function theme(){const dark=document.documentElement.dataset.theme==="dark";'
            'return {paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",'
            '"font.color":dark?"#e7edf6":"#18212b",'
            '"xaxis.gridcolor":dark?"#314158":"#dfe5ea",'
            '"yaxis.gridcolor":dark?"#314158":"#dfe5ea",'
            '"yaxis2.gridcolor":dark?"#314158":"#dfe5ea"};}'
            'function draw(){const fig=JSON.parse(document.getElementById(id+"-data").textContent);'
            'for(const [key,value] of Object.entries(theme())){'
            'if(key.includes(".")){const [group,property]=key.split(".");'
            'fig.layout[group]={...(fig.layout[group]||{}),[property]:value};}'
            'else fig.layout[key]=value;}'
            'Plotly.newPlot(id,fig.data,fig.layout,{responsive:true,displaylogo:false,scrollZoom:true});}'
            'if(window.Plotly)draw();else window.addEventListener("load",draw,{once:true});'
            'window.addEventListener("chronos2-theme-change",()=>{'
            'const p=document.getElementById(id);if(window.Plotly&&p.layout)Plotly.relayout(p,theme());});'
            '})();</script>')


def _comparator_contract(audit: Mapping[str, Any], zone: str,
                         comparator: pd.Series) -> tuple[dict[str, Any], bool]:
    """Recognize verified official reporting provenance, never infer identity.

    The caller's source adapters have already fetched/materialized or verified
    these snapshots. Their identity and recorded coverage must still agree
    with the actual series supplied to this renderer.
    """
    generic = {"id": "supplied_reporting_comparator", "report_label": "Comparateur fourni (identité non certifiée)",
               "official_dashboard_metric": False, "used_for_prediction": False,
               "report_note": "Comparaison sur les seules heures physiques communes ; identité officielle non certifiée."}
    sources = audit.get("zone_sources", {})
    declared = sources.get("storm", {}) if isinstance(sources, Mapping) else {}
    if not isinstance(declared, Mapping):
        return generic, False
    materialization = declared.get("comparator", declared.get("materialization", {}))
    if not isinstance(materialization, Mapping):
        return generic, False
    source = materialization.get("source", declared.get("source", {}))
    if not isinstance(source, Mapping):
        return generic, False
    series = f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm.da.cache"
    source_matches = (
        source.get("kind") == "saturn_storm_day_ahead_cache_with_native_gap_fallback"
        and source.get("zone") == zone and source.get("series") == series
        and source.get("primary_series") == series and source.get("used_for_prediction") is False
        and source.get("fallback_series") in (None, f"power.price.{zone.lower()}.euromwh.h.fcst.3mv.storm")
    )
    contract_matches = (
        declared.get("status") in ("complete", "partial") and declared.get("used_for_prediction") is False
        and materialization.get("role") == "evaluation_only_dashboard_comparator"
        and materialization.get("series") == series
        and materialization.get("column") == "storm_dashboard_official__q50"
        and materialization.get("used_for_prediction") is False
        and materialization.get("used_for_live_forecast") is False
    )
    if not source_matches or not contract_matches:
        return generic, False
    extracted = pd.Timestamp(source.get("extracted_at_utc"))
    if pd.isna(extracted) or extracted.tzinfo is None:
        return generic, False
    count = int(np.isfinite(comparator.to_numpy(float)).sum())
    if (materialization.get("expected_hours") != len(comparator)
            or materialization.get("available_hours") != count
            or materialization.get("missing_hours") != len(comparator) - count):
        raise ValueError("Official Storm reporting coverage differs from its source audit")
    dst = materialization.get("dst", {})
    if not isinstance(dst, Mapping) or dst.get("interpolation") is not False or dst.get("strict_08_fallback") is not False:
        return generic, False
    return {"id": "storm_official_dashboard", "label": "Storm officiel dashboard",
            "report_label": "Storm officiel dashboard", "series": series,
            "official_dashboard_metric": True, "used_for_prediction": False,
            "source_materialization": dict(materialization),
            "source_snapshot": dict(declared)}, True


def _attach_comparison(result: Any, comparator: pd.Series, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Keep all candidate prices and audit gaps in the renderer's exact window."""
    from nyx_prod_templates.report import _prepare_statistics_frame, _latest_statistics_window

    source = _latest_statistics_window(_prepare_statistics_frame(result, result.statistics_candidate))
    expected = pd.DatetimeIndex(pd.to_datetime(source["_timestamp_utc"], utc=True))
    aligned = comparator.reindex(expected).to_numpy(float)
    source["_benchmark_q50"] = aligned
    actual_finite = np.isfinite(pd.to_numeric(source.actual, errors="coerce").to_numpy(float))
    candidate_finite = np.isfinite(pd.to_numeric(source.q50, errors="coerce").to_numpy(float))
    benchmark_finite = np.isfinite(aligned)
    paired = actual_finite & candidate_finite & benchmark_finite
    contract, official = _comparator_contract(audit, result.zone, comparator)
    missing = expected[~benchmark_finite]
    coverage = {"status": "complete", "role": "verified_official_storm_pairing" if official else "supplied_comparator_pairing",
                "zone": result.zone, "used_for_prediction": False, "expected_hours": len(source),
                "observed_hours": int(actual_finite.sum()), "candidate_available_hours": int(candidate_finite.sum()),
                "benchmark_available_hours": int(benchmark_finite.sum()), "matched_hours": int(paired.sum()),
                "unmatched_observed_hours": int((actual_finite & ~paired).sum()),
                "forecast_placeholder_hours": int((~actual_finite).sum()),
                "missing_benchmark_utc": [stamp.isoformat() for stamp in missing],
                "missing_observation_utc": [stamp.isoformat() for stamp in expected[~actual_finite]],
                "window_start_utc": expected[0].isoformat(), "window_end_utc": expected[-1].isoformat(),
                "official_dashboard_metric": official, "candidate_prices_or_actuals_masked": False,
                "comparison_rule": "same finite (EPEX actual, CPU baseline point, supplied comparator) triplets"}
    note = (f"Comparaison sur {coverage['matched_hours']} heures physiques communes ; "
            f"{coverage['unmatched_observed_hours']} heures observées sans comparaison et "
            f"{coverage['forecast_placeholder_hours']} heures sans observation. "
            "Les prix de l’historique de référence et les observations restent visibles quand le comparateur manque. "
            "Aucun benchmark manquant n'est interpolé ni remplacé.")
    contract.update(report_note=note, materialization_audit=coverage)
    result.hourly_comparison_source = source
    result.hourly_comparison_contract = contract
    result.statistics_benchmark_label = contract["report_label"]
    result.statistics_benchmark_contract = contract
    result.forecast_benchmark_label = contract["report_label"]
    result.statistics_scope_note += " " + note
    # The pinned renderer accepts unequal official coverage only with this
    # exact, source-backed gap inventory. An uncertified partial comparator
    # instead has its own paired-hour section, leaving all candidate Statistics
    # untouched and never manufacturing official Storm provenance.
    same_observed_coverage = np.array_equal(candidate_finite[actual_finite], benchmark_finite[actual_finite])
    if benchmark_finite.any() and (official or same_observed_coverage):
        result.statistics_benchmark = pd.DataFrame({"timestamp": source["_timestamp_local"].to_numpy(),
                                                   "actual": source.actual.to_numpy(float), "q50": aligned})
        result.statistics_benchmark_label = contract["report_label"]
        result.statistics_benchmark_contract = contract
        if official:
            result.statistics_pairing_audit = coverage
    live = pd.DatetimeIndex(pd.to_datetime(result.forecast_native.timestamp, utc=True))
    live_values = comparator.reindex(live).to_numpy(float)
    if np.isfinite(live_values).any():
        result.forecast_benchmark = pd.DataFrame({"timestamp": result.forecast_native.timestamp.to_numpy(), "q50": live_values})
        result.forecast_benchmark_label = contract["report_label"]
    coverage["benchmark_label"] = contract["report_label"]
    coverage["comparative_statistics_attached"] = hasattr(result, "statistics_benchmark")
    if paired.any():
        actual = source.loc[paired, "actual"].to_numpy(float)
        candidate = source.loc[paired, "q50"].to_numpy(float)
        benchmark = source.loc[paired, "_benchmark_q50"].to_numpy(float)
        coverage["paired_metrics"] = {
            "candidate_mae": float(np.abs(candidate - actual).mean()),
            "candidate_rmse": float(np.sqrt(np.square(candidate - actual).mean())),
            "benchmark_mae": float(np.abs(benchmark - actual).mean()),
            "benchmark_rmse": float(np.sqrt(np.square(benchmark - actual).mean())),
        }
    return coverage


def _comparison_availability(info: Mapping[str, Any]) -> str:
    label = html.escape(str(info["benchmark_label"]))
    note = ("L'identité officielle du comparateur n'est pas certifiée. Les Statistics de l’historique de référence restent complètes ; "
            "la comparaison séparée utilise uniquement les heures communes."
            if not info["official_dashboard_metric"] else
            "Le snapshot officiel est audité ; les trous du benchmark restent manquants dans Statistics.")
    metrics = info.get("paired_metrics")
    table = ""
    if metrics and not info["comparative_statistics_attached"]:
        table = ('<div class="table-wrap"><table><thead><tr><th>Prévision</th><th>Heures communes</th>'
                 '<th>MAE · EUR/MWh</th><th>RMSE · EUR/MWh</th></tr></thead><tbody>'
                 f'<tr><td>NYX</td><td>{info["matched_hours"]}</td><td>{metrics["candidate_mae"]:.3f}</td><td>{metrics["candidate_rmse"]:.3f}</td></tr>'
                 f'<tr><td>{label}</td><td>{info["matched_hours"]}</td><td>{metrics["benchmark_mae"]:.3f}</td><td>{metrics["benchmark_rmse"]:.3f}</td></tr>'
                 '</tbody></table></div>')
    return ('<section data-report-section="benchmark-availability"><h2>Disponibilité de la comparaison historique</h2>'
            f'<p>Comparateur : {label}. {info["matched_hours"]} heures communes / {info["observed_hours"]} heures observées ; '
            f'{info["unmatched_observed_hours"]} heures observées sans comparaison ; '
            f'{info["forecast_placeholder_hours"]} heures sans observation. Fenêtre : '
            + html.escape(str(info["window_start_utc"])) + ' → ' + html.escape(str(info["window_end_utc"])) + '.</p>'
            '<p>' + html.escape(note) + ' Aucun prix, observation ou benchmark manquant n’est remplacé par zéro.</p>'
            + table + '</section>')


def _final_section(zone: str, day: str, final: pd.DataFrame, baseline: pd.DataFrame,
                   actual: pd.Series, storm: pd.Series, timezone: str,
                   comparator_label: str = "Storm — comparateur") -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    local = final.index.tz_convert(timezone)
    point = final["price_eur_mwh"].to_numpy(float)
    probability = final["p_negative"].to_numpy(float)
    lower = final["interval_lower_eur_mwh"].to_numpy(float)
    upper = final["interval_upper_eur_mwh"].to_numpy(float)
    reference = actual.reindex(final.index).to_numpy(float)
    benchmark = storm.reindex(final.index).to_numpy(float)
    last_day = str(local[-1].date())
    day_label = day if last_day == day else day + " → " + last_day
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=local, y=upper, name="Borne supérieure indicative",
                               mode="lines", line={"width": 0}, showlegend=False,
                               hoverinfo="skip", legendgroup="annual-interval"))
    chart.add_trace(go.Scatter(x=local, y=lower, name="Intervalle indicatif · 80 % nominal",
                               mode="lines", line={"width": 0}, fill="tonexty",
                               fillcolor="rgba(228,139,35,0.18)", legendgroup="annual-interval",
                               customdata=np.column_stack([lower, upper]),
                               hovertemplate="%{x}<br>Borne inférieure : %{customdata[0]:.2f} EUR/MWh"
                                             "<br>Borne supérieure : %{customdata[1]:.2f} EUR/MWh"
                                             "<extra>Enveloppe non calibrée des experts finaux</extra>"))
    if np.isfinite(reference).any():
        chart.add_trace(go.Scatter(x=local, y=reference, name="Observé EPEX", mode="lines+markers"))
    if np.isfinite(benchmark).any():
        chart.add_trace(go.Scatter(x=local, y=benchmark, name=comparator_label, mode="lines+markers"))
    chart.add_trace(go.Scatter(x=local, y=point, name="NYX", mode="lines+markers"))
    chart.update_layout(title=f"{zone} — prévision finale émise pour le {day_label}",
                        xaxis_title="Livraison locale", yaxis_title="EUR/MWh",
                        margin={"l": 50, "r": 30, "t": 65, "b": 45}, hovermode="x unified",
                        legend={"orientation": "h", "y": 1.02}, height=410)
    risk = make_subplots(specs=[[{"secondary_y": True}]])
    risk.add_trace(go.Bar(x=local, y=probability * 100., name="P(prix < 0)"), secondary_y=False)
    risk.add_trace(go.Scatter(x=local, y=point, name="NYX", mode="lines"), secondary_y=True)
    risk.update_yaxes(title_text="Probabilité (%)", range=[0, 100], secondary_y=False)
    risk.update_yaxes(title_text="Prix (EUR/MWh)", secondary_y=True)
    risk.update_layout(title="NYX — probabilité de prix négatif", hovermode="x unified",
                       margin={"l": 50, "r": 60, "t": 65, "b": 45}, height=340,
                       legend={"orientation": "h", "y": 1.02})
    rows = []
    for i, stamp in enumerate(local):
        rows.append('<tr><td title="' + html.escape(final.index[i].isoformat()) + '">'
                    + html.escape(stamp.strftime("%d/%m/%Y %H:%M %Z %z")) + '</td>'
                    + ''.join(f'<td>{_number(value)}</td>' for value in
                              (point[i], lower[i], upper[i], probability[i] * 100.,
                               benchmark[i], reference[i])) + '</tr>')
    identifier = uuid4().hex
    return (
        '<section id="annual-final" data-report-section="annual-final-forecast" '
        'data-model-role="annual-final" data-final-oof-history="unavailable">'
        f'<h2>{html.escape(zone)} — {FINAL_LABEL}</h2>'
        f'<p class="muted">Livraison du {html.escape(day_label)} · {len(final)} heures physiques · '
        f'{html.escape(timezone)}. Les modèles et ajustements de cette prévision sont décrits dans la traçabilité du rapport.</p>'
        '<div class="metric-grid">'
        f'<div class="metric-card"><div class="metric-label">Prix moyen final</div><div class="metric-value">{point.mean():.2f}</div><div class="metric-unit">EUR/MWh</div></div>'
        f'<div class="metric-card"><div class="metric-label">Minimum / maximum final</div><div class="metric-value">{point.min():.2f} / {point.max():.2f}</div><div class="metric-unit">EUR/MWh</div></div>'
        f'<div class="metric-card"><div class="metric-label">Probabilité négative maximale</div><div class="metric-value">{probability.max():.1%}</div></div>'
        f'<div class="metric-card"><div class="metric-label">Heures avec probabilité ≥ 50 %</div><div class="metric-value">{int((probability >= .5).sum())} / {len(final)}</div></div>'
        '</div><p>Le prix final est une prévision ponctuelle. La probabilité de prix négatif '
        'provient du classificateur final.</p>'
        f'<p data-interval-method="baseline_envelope_uncalibrated"><strong>{html.escape(INTERVAL_LABEL)}.</strong> '
        'La borne basse est le minimum entre le P10 de référence et le prix final ; la borne haute '
        'est le maximum entre le P90 de référence et le prix final. L’enveloppe conserve tout '
        'l’intervalle de référence et contient le prix final. Ce sont des bornes indicatives, '
        'pas les P10/P90 d’une distribution des experts finaux. La couverture à 80 % '
        'des experts finaux n’est pas garantie.</p>'
        '<p>L’historique hors échantillon de la prévision finale n’est pas disponible dans cette sortie : '
        'ses MAE/RMSE sur 365 jours ne sont pas affichées. Les scores et Statistics plus bas évaluent '
        'exclusivement l’historique de référence ; ils ne mesurent pas la performance de la prévision finale.</p>'
        + _plot(chart, "annual-final-price-" + identifier)
        + _plot(risk, "annual-final-negative-" + identifier)
        + '<details><summary>Détail horaire des prévisions finales</summary><div class="table-wrap"><table>'
        '<thead><tr><th>Heure locale</th><th>NYX · EUR/MWh</th>'
        '<th>Borne indicative basse</th><th>Borne indicative haute</th><th>P(prix &lt; 0) · %</th>'
        f'<th>{html.escape(comparator_label)}</th><th>Observé EPEX</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></details></section>'
    )


def _remove_input_display(document: str, zone: str) -> str:
    """Remove only the pinned renderer's final input block in this country.

    HTML parsing keeps script text and nested attribution elements from being
    mistaken for section boundaries. The exact heading and containing country
    section must both match before any markup is removed.
    """
    offsets = [0]
    for line in document.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    class InputDisplay(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.depth = 0
            self.target_depth = None
            self.country_sections = 0
            self.heading = None
            self.heading_text = []
            self.start = None
            self.end = None
            self.matches = 0

        def position(self):
            line, column = self.getpos()
            return offsets[line - 1] + column

        def handle_starttag(self, tag, attrs):
            if tag == "section":
                self.depth += 1
                if dict(attrs).get("id") == zone.lower():
                    self.country_sections += 1
                    self.target_depth = self.depth
            elif tag == "h3" and self.target_depth == self.depth:
                self.heading = self.position()
                self.heading_text = []

        def handle_data(self, text):
            if self.heading is not None:
                self.heading_text.append(text)

        def handle_endtag(self, tag):
            if tag == "h3" and self.heading is not None:
                if "".join(self.heading_text).strip() in ("Variables d’entrée", "Variables d'entrée"):
                    self.start = self.heading
                    self.matches += 1
                self.heading = None
            elif tag == "section":
                if self.target_depth == self.depth:
                    self.end = self.position()
                    self.target_depth = None
                self.depth -= 1

    parser = InputDisplay()
    parser.feed(document)
    parser.close()
    if (parser.country_sections != 1 or parser.matches != 1 or parser.start is None
            or parser.end is None or parser.start >= parser.end):
        raise ValueError("Pinned report input display section is missing or ambiguous")
    return document[:parser.start] + document[parser.end:]


def render_details(zone: str, day: str, final_forecast: pd.DataFrame,
                   baseline: pd.DataFrame, covariates: pd.DataFrame,
                   observed: pd.Series | pd.DataFrame,
                   storm: pd.Series | pd.DataFrame | None,
                   output_path: str | Path, audit: Mapping[str, Any]) -> Path:
    """Write a final-current + baseline-history report using existing UI.

    Baseline ``nyx__q10/q50/q90`` must cover the preceding 365 civil days
    and one/two delivery days according to audit ``horizon_days``. Both days
    of a two-day forecast require the same explicit issued forecast origin.
    ``observed`` is an already audited report-only EPEX
    snapshot. Missing historical refresh values use the frozen baseline's
    EPEX actuals; current-day actuals are never borrowed from training data.
    ``storm`` is an optional audited, possibly incomplete reporting snapshot.
    """
    from nyx_prod_templates.common import ZoneData, ZoneRunResult
    from nyx_prod_templates.metrics import compute_metrics, metric_breakdowns
    from nyx_prod_templates.report import write_html_report

    zone = str(zone).upper()
    if zone not in TIMEZONES:
        raise ValueError("Unknown CWE country")
    timezone = TIMEZONES[zone]
    horizon_days = audit.get("horizon_days", 1)
    if type(horizon_days) is not int or horizon_days not in (1, 2):
        raise ValueError("The report forecast horizon must be one or two civil days")
    current = _grid(day, timezone)
    if horizon_days == 2:
        next_day = str(pd.Timestamp(day).date() + timedelta(days=1))
        current = current.append(_grid(next_day, timezone))
    start = pd.Timestamp(pd.Timestamp(day).date() - timedelta(days=365), tz=timezone)
    history = pd.date_range(start, pd.Timestamp(day, tz=timezone), freq="h", inclusive="left").tz_convert("UTC").rename("delivery_start_utc")
    final = _utc_frame(final_forecast, "final_forecast")
    if not _same_instants(final.index, current):
        raise ValueError("Final annual forecast must match the exact one/two-day delivery UTC grid")
    final_columns = ("price_eur_mwh", "p_negative", "interval_lower_eur_mwh", "interval_upper_eur_mwh")
    if set(final_columns).difference(final):
        raise ValueError("Final annual price, probability and indicative interval columns are required")
    for column in final_columns:
        final[column] = pd.to_numeric(final[column], errors="raise").astype(float)
    if not np.isfinite(final[list(final_columns)].to_numpy()).all():
        raise ValueError("Non-finite final annual forecasts")
    if not final["p_negative"].between(0., 1.).all():
        raise ValueError("Final negative-price probabilities must lie in [0,1]")
    base = _utc_frame(baseline, "baseline")
    expected = history.append(current)
    if len(expected.difference(base.index)):
        raise ValueError("The CPU baseline must contain all FINAL365 and delivery hours")
    base = base.reindex(expected)
    missing = {"nyx__q10", "nyx__q50", "nyx__q90", "actual"}.difference(base)
    if missing:
        raise ValueError(f"CPU baseline columns missing: {sorted(missing)}")
    current_base = base.loc[current]
    source_lower = pd.to_numeric(current_base["nyx__q10"], errors="raise").to_numpy(float)
    source_upper = pd.to_numeric(current_base["nyx__q90"], errors="raise").to_numpy(float)
    point = final["price_eur_mwh"].to_numpy(float)
    lower = final["interval_lower_eur_mwh"].to_numpy(float)
    upper = final["interval_upper_eur_mwh"].to_numpy(float)
    if not ((lower <= point).all() and (point <= upper).all()
            and (lower <= source_lower).all() and (source_upper <= upper).all()):
        raise ValueError("Final indicative intervals must contain the point and the entire CPU source interval")
    if not (np.array_equal(lower, np.minimum(source_lower, point))
            and np.array_equal(upper, np.maximum(source_upper, point))):
        raise ValueError("Final indicative intervals differ from baseline_envelope_uncalibrated")
    refreshed = _utc_series(observed, "observed")
    comparator = _utc_series(storm, "storm")
    frozen_actual = pd.to_numeric(base.loc[history, "actual"], errors="raise").astype(float)
    if not np.isfinite(frozen_actual.to_numpy()).all():
        raise ValueError("The frozen CPU baseline FINAL365 actuals are incomplete")
    historical_refresh = refreshed.reindex(history)
    historical_actual = historical_refresh.where(historical_refresh.notna(), frozen_actual)
    actual = pd.concat([historical_actual, refreshed.reindex(current)])
    if not np.isfinite(historical_actual.to_numpy()).all():
        raise ValueError("FINAL365 EPEX observations must be complete")
    native = _baseline_prediction(base.loc[history], actual, timezone)
    supplied_origin = next((audit[name] for name in ("forecast_origin_utc", "forecast_cutoff_utc", "cutoff_utc")
                            if audit.get(name) is not None), None)
    future = _baseline_prediction(base.loc[current], actual, timezone,
                                  issued_origin=supplied_origin if horizon_days == 2 else None,
                                  allow_origin_fallback=horizon_days == 1)
    if horizon_days == 2:
        future_origins = pd.DatetimeIndex(pd.to_datetime(future["origin_timestamp"], utc=True))
        if future_origins.nunique() != 1 or not (future_origins < current[0]).all():
            raise ValueError("Both forecast days must share one cutoff preceding the first delivery")
    statistics = pd.concat([native, future], ignore_index=True)
    by_horizon, by_hour = metric_breakdowns(native, None, 150.)
    cov = _utc_frame(covariates, "covariates").apply(pd.to_numeric, errors="raise")
    if np.isinf(cov.to_numpy(float)).any():
        raise ValueError("Infinite report covariates")
    cov.index = cov.index.tz_convert(timezone)
    target = actual.copy()
    target.index = target.index.tz_convert(timezone)
    # The shared live graph's context ends before delivery even if a later
    # report-only snapshot now contains observed prices for that delivery.
    target_history = target.loc[target.index < current[0].tz_convert(timezone)]
    coverage = pd.DataFrame({"alias": list(cov), "coverage": cov.notna().mean().to_numpy(),
                             "coverage_after_fill": cov.notna().mean().to_numpy()})
    supplied_manifest = audit.get("input_manifest")
    manifest = (pd.DataFrame(supplied_manifest) if isinstance(supplied_manifest, list)
                else pd.DataFrame({"alias": list(cov), "source": "annual frozen snapshot"}))
    zone_data = ZoneData(zone=zone, timezone=timezone, frequency="h", target=target_history,
                         covariates=cov, model_context_covariates=cov.copy(),
                         known_future_columns=list(cov), coverage=coverage, input_manifest=manifest,
                         diagnostics={"report_only": True, "model_role": "cpu_baseline"})
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = ZoneRunResult(zone=zone, metrics_native=compute_metrics(native, 150.),
                           metrics_baseline=None, backtest_native=native, backtest_baseline=None,
                           metrics_by_horizon=by_horizon, metrics_by_hour=by_hour,
                           forecast_native=future[["timestamp", *[f"q{level}" for level in range(10, 100, 10)], "point"]],
                           forecast_baseline=None, zone_data=zone_data, output_dir=destination.parent)
    result.statistics_candidate = statistics
    result.statistics_candidate_label = BASELINE_LABEL
    result.statistics_scope_note = (
        "Statistiques de l’historique de référence. "
        "L’historique hors échantillon de la prévision finale n’est pas disponible."
    )
    comparison_audit = _attach_comparison(result, comparator, audit)
    report_audit = {**dict(audit), "model_roles": {"current": "annual_final_point_forecast",
                     "historical_statistics": "cpu_baseline_quantiles"},
                    "final_annual_oof_history_available": False, "training_performed": False,
                    "final_price_quantiles_available": False,
                    "final_prediction_interval_available": True,
                    "final_interval_method": "baseline_envelope_uncalibrated",
                    "final_interval_calibration_status": "uncalibrated_for_final_experts",
                    "final_interval_nominal_probability": .8,
                    "final_interval_coverage_guaranteed": False,
                    "horizon_days": horizon_days,
                    "forecast_first_delivery_day": day,
                    "forecast_last_delivery_day": str(current[-1].tz_convert(timezone).date()),
                    "forecast_physical_hours": len(current),
                    "forecast_origin_utc": (str(pd.Timestamp(future["origin_timestamp"].iloc[0]).tz_convert("UTC"))
                                             if horizon_days == 2 else dict(audit).get("forecast_origin_utc")),
                    "baseline_intermediate_deciles": "linear_display_interpolation_of_real_q10_q50_q90",
                    "statistics_pairing": comparison_audit,
                    "historical_frozen_epex_hours_used": int(historical_refresh.isna().sum()),
                    "evaluation_start_day": str(start.date()), "evaluation_end_day": str(pd.Timestamp(day).date() - timedelta(days=1)),
                    "evaluation_physical_hours": len(history)}
    write_html_report([result], {"report": {"title": f"{zone} — NYX",
                                             "forecast_history_hours": 168}}, destination)
    document = destination.read_text(encoding="utf-8")
    if horizon_days == 2:
        document = document.replace("(24 h)", f"({len(current)} h)")
    for old, new in (
        ("Chronos-2 + covariables P50", BASELINE_LABEL + " P50"),
        ("Covariables natives", BASELINE_LABEL), ("Covariables P50", BASELINE_LABEL + " P50"),
        ("MAE covariables", "MAE NYX"), ("Couverture covariables", "Couverture NYX"),
        ("Biais covariables", "Biais NYX"), ("Erreur Chronos-2", "Erreur NYX"),
        ("Performance du modèle", "Performance de l’historique de référence"),
        ("Prévision Day-Ahead réelle", "NYX — courbe de référence P10/P50/P90"),
        ("Backtest et probabilités", "Historique de référence et intervalles de prix"),
        ("Prix moyen du jour — modèle", "Prix moyen du jour — NYX"),
        ("<h2>Statistics</h2>", "<h2>Statistics — historique de référence</h2>"),
        ('<div class="statistics-title-pill">Statistics</div>',
         '<div class="statistics-title-pill">Statistiques de l’historique de référence</div>'),
        ("Notre modèle", "NYX"),
        ("annual frozen snapshot", "instantané historique de référence"),
        ("Pour Kalman, l’attribution amont et l’ajustement final sont distingués", "L’attribution amont et l’ajustement final sont distingués"),
        ("Chronos-2", "NYX"),
    ):
        document = document.replace(old, new)
    document = re.sub(r'<h3>Comparaison au modèle prix seul</h3>\s*(?:<div class="table-wrap">)?<table\b.*?</table>(?:</div>)?',
                      "", document, count=1, flags=re.DOTALL)
    document = _remove_input_display(document, zone)
    # Scientific provenance remains machine-readable without creating another
    # visible report section or copying audit text into the report navigation.
    audit_json = json.dumps(report_audit, ensure_ascii=False, indent=2, default=str)
    audit_json = audit_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    provenance = '<script id="nyx-report-audit" type="application/json">' + audit_json + '</script>'
    inserted = (_final_section(zone, day, final, base.loc[current], actual, comparator, timezone,
                               comparator_label=comparison_audit["benchmark_label"])
                + provenance)
    document = document.replace("<main>", "<main>" + inserted, 1)
    document = document.replace('<a href="#statistics">Statistics</a>',
                                '<a href="#annual-final">NYX</a><a href="#statistics">Statistics · historique de référence</a>', 1)
    document = document.replace(f'<h2>{zone}</h2>', f'<h2>{zone} — historique de référence</h2>', 1)
    document = document.replace(f'<section id="{zone.lower()}">',
                                f'<section id="{zone.lower()}" data-model-role="cpu-baseline">', 1)
    destination.write_text(document, encoding="utf-8", newline="\n")
    return destination


__all__ = ["render_details"]
