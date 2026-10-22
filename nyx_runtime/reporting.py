"""Validate prediction CSVs and render standalone daily and J+1/J+2 reports.

The CSV schema is shared with the forecasting engine. Dates describe physical
hours in Europe/Paris, including 23-hour and 25-hour delivery days. This module
does not import estimators, contact providers or refresh observed prices.

HTML layout and tab interactions live in templates/report.html. Plotly is
embedded from a local asset so each generated report can be opened offline.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
from string import Template
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo


CSV_COLUMNS = (
    "delivery_day", "zone", "delivery_start_utc", "delivery_start_local",
    "forecast_origin_utc", "horizon_day", "prediction_eur_mwh",
    "lower_bound_eur_mwh", "upper_bound_eur_mwh", "storm_eur_mwh",
    "observed_eur_mwh", "interval_method", "nominal_interval_level",
    "interval_calibrated", "source_html_sha256", "source_generated_at_utc",
    "source_html_name",
)
VALUE_COLUMNS = (
    "prediction_eur_mwh", "lower_bound_eur_mwh", "upper_bound_eur_mwh",
    "storm_eur_mwh", "observed_eur_mwh",
)
ZONES = ("BE", "DE", "FR", "NL")
LOCAL_TZ = ZoneInfo("Europe/Paris")
INTERVAL_METHOD = "baseline_envelope_uncalibrated"
PredictionRow = dict[str, Any]


class ReportFormatError(ValueError):
    """Prediction values or their timeline violate the report contract."""


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _datetime(value: Any, label: str) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ReportFormatError(f"{label}: horodatage ISO invalide: {value!r}") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ReportFormatError(f"{label}: fuseau obligatoire.")
    return result


def _utc(value: Any, label: str = "timestamp") -> datetime:
    return _datetime(value, label).astimezone(timezone.utc)


def physical_day_hours(delivery_day: str | date) -> list[datetime]:
    """Return the UTC start of each physical hour in a Paris delivery day."""
    day = date.fromisoformat(str(delivery_day))
    first = datetime.combine(day, time(), LOCAL_TZ).astimezone(timezone.utc)
    last = datetime.combine(day + timedelta(days=1), time(), LOCAL_TZ).astimezone(timezone.utc)
    hour_count = int((last - first).total_seconds() / 3600)
    return [first + timedelta(hours=index) for index in range(hour_count)]


def _finite(value: Any, label: str) -> float | None:
    """Accept a finite JSON number or a genuinely missing value."""
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ReportFormatError(f"{label}: nombre JSON fini ou null attendu.")
    return float(value)


def _csv_number(value: Any, label: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ReportFormatError(f"{label}: nombre invalide.") from error
    return _finite(number, label)


def _row_timeline(row: Mapping[str, Any]) -> tuple[str, datetime, str, datetime]:
    """Validate one row's country, physical hour, shared origin and horizon."""
    zone = row["zone"]
    if zone not in ZONES:
        raise ReportFormatError(f"Zone inconnue: {zone}")
    start = _utc(row["delivery_start_utc"])
    if _datetime(row["delivery_start_utc"], "delivery_start_utc").utcoffset() != timedelta(0):
        raise ReportFormatError("delivery_start_utc doit être exprimé en UTC.")

    local = start.astimezone(LOCAL_TZ)
    day = str(row["delivery_day"])
    if local.date().isoformat() != day:
        raise ReportFormatError("Jour de livraison incohérent avec le timestamp UTC.")
    supplied_local = _datetime(row["delivery_start_local"], "delivery_start_local")
    if supplied_local.isoformat() != local.isoformat():
        raise ReportFormatError("Heure locale/offset incohérents avec Europe/Paris.")

    origin = _utc(row["forecast_origin_utc"], "forecast_origin_utc")
    if _datetime(row["forecast_origin_utc"], "forecast_origin_utc").utcoffset() != timedelta(0):
        raise ReportFormatError("forecast_origin_utc doit être exprimé en UTC.")
    origin_local = origin.astimezone(LOCAL_TZ)
    origin_time = (
        origin_local.hour, origin_local.minute, origin_local.second, origin_local.microsecond,
    )
    if origin_time != (8, 0, 0, 0):
        raise ReportFormatError("Le cutoff doit correspondre à 08:00 Europe/Paris.")
    horizon = (local.date() - origin_local.date()).days
    if horizon not in (1, 2) or int(row["horizon_day"]) not in (1, 2):
        raise ReportFormatError("horizon_day doit rester dans le périmètre J+1/J+2 (1 ou 2).")
    if int(row["horizon_day"]) != horizon:
        raise ReportFormatError("horizon_day incohérent avec le cutoff commun.")
    return zone, start, day, origin


def _normalize_values(row: PredictionRow, zone: str, start: datetime) -> None:
    """Convert CSV numbers and enforce the indicative envelope's meaning."""
    for column in VALUE_COLUMNS:
        row[column] = _csv_number(row[column], column)
    for column in ("prediction_eur_mwh", "lower_bound_eur_mwh", "upper_bound_eur_mwh"):
        if row[column] is None:
            raise ReportFormatError(f"{zone} {start.isoformat()}: {column} obligatoire et fini.")

    point = row["prediction_eur_mwh"]
    low, high = row["lower_bound_eur_mwh"], row["upper_bound_eur_mwh"]
    if low > high:
        raise ReportFormatError("Bornes indicatives inversées.")
    if point < low or point > high:
        raise ReportFormatError("L'enveloppe indicative doit contenir le prix final.")
    if row["interval_method"] != INTERVAL_METHOD or str(row["interval_calibrated"]).lower() != "false":
        raise ReportFormatError("L'enveloppe source doit rester indicative et non calibrée.")
    if float(row["nominal_interval_level"]) != 0.8:
        raise ReportFormatError("Niveau nominal attendu: 0,8.")


def validate_rows(rows: list[PredictionRow]) -> list[str]:
    """Validate complete CWE delivery days and normalize numeric fields in place.

    One daily report may contain J+1 or J+2; a combined report must contain both.
    Every included delivery day requires all four countries and its full ordered
    UTC grid. Missing Storm or observed prices remain missing.
    """
    if not rows:
        raise ReportFormatError("Aucune ligne de prévision.")
    keys: set[tuple[str, datetime]] = set()
    groups: dict[tuple[str, str], list[datetime]] = {}
    origins: set[datetime] = set()
    for row in rows:
        zone, start, day, origin = _row_timeline(row)
        origins.add(origin)
        key = zone, start
        if key in keys:
            raise ReportFormatError(f"Heure dupliquée: {zone} {start.isoformat()}")
        keys.add(key)
        groups.setdefault((day, zone), []).append(start)
        _normalize_values(row, zone, start)

    if len(origins) != 1:
        raise ReportFormatError("Les lignes doivent conserver un cutoff commun.")
    days = sorted({day for day, _ in groups})
    if len(days) > 2:
        raise ReportFormatError("Au plus deux jours de livraison J+1/J+2 sont autorisés.")
    if len(days) == 2:
        origin_day = next(iter(origins)).astimezone(LOCAL_TZ).date()
        horizons = [(date.fromisoformat(day) - origin_day).days for day in days]
        if horizons != [1, 2]:
            raise ReportFormatError("Un rapport combiné doit couvrir J+1 et J+2 consécutifs au cutoff commun.")
    for day in days:
        for zone in ZONES:
            if groups.get((day, zone)) != physical_day_hours(day):
                raise ReportFormatError(f"Journée physique incomplète ou désordonnée: {day} {zone}.")
    if any((date.fromisoformat(b) - date.fromisoformat(a)).days != 1 for a, b in zip(days, days[1:])):
        raise ReportFormatError("Jours de livraison non contigus.")
    return days


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the canonical CSV order: delivery day, country, physical hour."""
    return row["delivery_day"], row["zone"], row["delivery_start_utc"]


def write_predictions(path: str | Path, rows: Sequence[PredictionRow]) -> None:
    """Write canonical columns without rounding or replacing missing values."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_predictions(path: str | Path) -> list[PredictionRow]:
    """Read, sort and validate a canonical prediction CSV."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        # Fresh forecasts may omit the optional historical HTML provenance name.
        missing = (set(CSV_COLUMNS) - {"source_html_name"}) - set(reader.fieldnames or [])
        if missing:
            raise ReportFormatError(f"Colonnes CSV absentes: {', '.join(sorted(missing))}")
        rows = list(reader)
    for row in rows:
        row.setdefault("source_html_name", "")
        row["horizon_day"] = int(row["horizon_day"])
        row["nominal_interval_level"] = float(row["nominal_interval_level"])
    rows.sort(key=row_key)
    validate_rows(rows)
    return rows


def _find_plotly(csv_path: str | Path) -> str:
    """Prefer a report's adjacent local asset, then the installed Plotly asset."""
    candidates = [
        Path(csv_path).parent / "assets" / "plotly.min.js",
        Path(__file__).resolve().parents[1] / "assets" / "plotly.min.js",
    ]
    for path in candidates:
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            if "plotly.js v" in content[:400] and "Licensed under the MIT license" in content[:500]:
                return content
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as error:
        raise FileNotFoundError(
            "Plotly local absent : installer plotly ou fournir assets/plotly.min.js."
        ) from error
    return get_plotlyjs()


def _display(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _complete_mean(rows: Sequence[PredictionRow], column: str) -> float | None:
    values = [row[column] for row in rows]
    return statistics.fmean(values) if values and all(value is not None for value in values) else None


def _local_labels(rows: Sequence[PredictionRow]) -> list[str]:
    return [
        _utc(row["delivery_start_utc"]).astimezone(LOCAL_TZ).strftime("%Y-%m-%d · %H:%M %Z")
        for row in rows
    ]


def _zone_layout(zone_index: int, labels: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Place one country's price panel and error panel in the four-column chart."""
    suffix = "" if zone_index == 0 else str(zone_index + 1)
    axis, y_axis = "x" + suffix, "y" + suffix
    error_suffix = str(zone_index + 5)
    error_axis, error_y = "x" + error_suffix, "y" + error_suffix
    left, right = zone_index / 4 + 0.015, (zone_index + 1) / 4 - 0.015
    ticks = list(range(len(labels)))[::max(1, len(labels) // 5)]
    price_x = {
        "domain": [left, right], "anchor": y_axis, "tickmode": "array",
        "tickvals": ticks, "ticktext": [labels[index].split(" · ")[-1] for index in ticks],
        "gridcolor": "#343b40",
    }
    layout = {
        "xaxis" + suffix: price_x,
        "yaxis" + suffix: {
            "domain": [.37, 1], "anchor": axis,
            "title": "€/MWh" if zone_index == 0 else "", "gridcolor": "#343b40",
        },
        "xaxis" + error_suffix: {**price_x, "anchor": error_y},
        "yaxis" + error_suffix: {
            "domain": [0, .24], "anchor": error_axis,
            "gridcolor": "#343b40", "zerolinecolor": "#888",
        },
    }
    annotation = {
        "x": (left + right) / 2, "y": 1.05, "xref": "paper", "yref": "paper",
        "text": "<b>" + ZONES[zone_index] + "</b>", "showarrow": False,
    }
    return layout, annotation


def _zone_traces(
    rows: Sequence[PredictionRow], zone_index: int, labels: list[str],
) -> list[dict[str, Any]]:
    """Build forecast/comparator lines, the indicative band, and error lines."""
    suffix = "" if zone_index == 0 else str(zone_index + 1)
    axis, y_axis = "x" + suffix, "y" + suffix
    x = list(range(len(rows)))
    traces: list[dict[str, Any]] = []
    for bound, column in (("lower", "lower_bound_eur_mwh"), ("upper", "upper_bound_eur_mwh")):
        traces.append({
            "type": "scatter", "mode": "lines", "x": x, "y": [row[column] for row in rows],
            "xaxis": axis, "yaxis": y_axis, "name": "Indicative envelope (80% nominal)",
            "showlegend": zone_index == 0 and bound == "upper", "legendgroup": "envelope",
            "line": {"color": "rgba(228,139,35,.35)", "width": .6},
            "fill": "tonexty" if bound == "upper" else "none",
            "fillcolor": "rgba(228,139,35,.18)", "meta": {"role": "model_interval", "bound": bound},
            "hoverinfo": "skip", "connectgaps": False,
        })
    series = (
        ("Observed (EPEX)", "observed_eur_mwh", "#c084fc"),
        ("Storm", "storm_eur_mwh", "#00a8ff"),
        ("NYX", "prediction_eur_mwh", "#e48b23"),
    )
    for name, column, color in series:
        traces.append({
            "type": "scatter", "mode": "lines", "x": x, "y": [row[column] for row in rows],
            "xaxis": axis, "yaxis": y_axis, "name": name, "legendgroup": name,
            "showlegend": zone_index == 0, "line": {"color": color, "width": 2},
            "customdata": labels,
            "hovertemplate": "%{customdata}<br>" + name + ": %{y:.2f} €/MWh<extra></extra>",
            "connectgaps": False,
        })
        if name != "Observed (EPEX)":
            errors = [
                None if row[column] is None or row["observed_eur_mwh"] is None
                else row[column] - row["observed_eur_mwh"]
                for row in rows
            ]
            traces.append({
                "type": "scatter", "mode": "lines", "x": x, "y": errors,
                "xaxis": "x" + str(zone_index + 5), "yaxis": "y" + str(zone_index + 5),
                "name": name, "legendgroup": name, "showlegend": False,
                "line": {"color": color, "width": 1.5}, "customdata": labels,
                "hovertemplate": "%{customdata}<br>Error: %{y:.2f} €/MWh<extra></extra>",
                "connectgaps": False,
            })
    return traces


def _build_figure(rows: Sequence[PredictionRow]) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    layout: dict[str, Any] = {}
    for zone_index, zone in enumerate(ZONES):
        group = [row for row in rows if row["zone"] == zone]
        labels = _local_labels(group)
        zone_layout, annotation = _zone_layout(zone_index, labels)
        layout.update(zone_layout)
        annotations.append(annotation)
        traces.extend(_zone_traces(group, zone_index, labels))
    layout.update({
        "height": 650, "paper_bgcolor": "#1c2226", "plot_bgcolor": "#1c2226",
        "font": {"color": "#e7e9ea", "family": "Arial"},
        "margin": {"t": 55, "l": 45, "r": 20, "b": 50}, "hovermode": "closest",
        "annotations": annotations, "legend": {"orientation": "h", "y": 1.17},
        "uirevision": "csv-snapshot",
    })
    return {"data": traces, "layout": layout}


def _summary_cards(rows: Sequence[PredictionRow], days: list[str]) -> str:
    cards = []
    for zone in ZONES:
        for day in days:
            daily = [row for row in rows if row["zone"] == zone and row["delivery_day"] == day]
            cards.append(
                f'<article><h3>{zone} · {html.escape(day)}</h3>'
                f'<p>Observed (EPEX) <strong>{_display(_complete_mean(daily, "observed_eur_mwh"))}</strong></p>'
                f'<p>Storm <strong>{_display(_complete_mean(daily, "storm_eur_mwh"))}</strong></p>'
                f'<p>NYX <strong>{_display(_complete_mean(daily, "prediction_eur_mwh"))}</strong></p>'
                f'<small>{len(daily)} physical hours · €/MWh</small></article>'
            )
    return "".join(cards)


def _hourly_table(rows: Sequence[PredictionRow]) -> str:
    table_rows = []
    for zone in ZONES:
        group = [row for row in rows if row["zone"] == zone]
        for row, label in zip(group, _local_labels(group)):
            values = [
                row["observed_eur_mwh"], row["storm_eur_mwh"], row["prediction_eur_mwh"],
                row["lower_bound_eur_mwh"], row["upper_bound_eur_mwh"],
            ]
            values.extend(
                None if row[column] is None or row["observed_eur_mwh"] is None
                else row[column] - row["observed_eur_mwh"]
                for column in ("storm_eur_mwh", "prediction_eur_mwh")
            )
            timestamp = html.escape(row["delivery_start_utc"], quote=True)
            cells = "".join("<td>" + _display(value) + "</td>" for value in values)
            table_rows.append(
                f'<tr><th scope="row">{zone}</th><td title="{timestamp}">'
                + html.escape(label) + "</td>" + cells + "</tr>"
            )
    return "".join(table_rows)


def _render_html(rows: list[PredictionRow], plotly_script: str) -> str:
    days = sorted({row["delivery_day"] for row in rows})
    title = " → ".join([days[0], days[-1]]) if len(days) > 1 else days[0]
    figure_json = json.dumps(_build_figure(rows), ensure_ascii=False, allow_nan=False)
    # A JSON value must never terminate its containing HTML script element.
    figure_json = figure_json.replace("</", "<\\/")
    template = Template((Path(__file__).parent / "templates/report.html").read_text(encoding="utf-8"))
    return template.substitute(
        title=html.escape(title),
        forecast_origin=html.escape(rows[0]["forecast_origin_utc"]),
        generated_at=html.escape(datetime.now(timezone.utc).isoformat()),
        # These fragments are constructed above from escaped labels and validated numbers.
        summary_cards=_summary_cards(rows, days),
        hourly_rows=_hourly_table(rows),
        figure_json=figure_json,
        # Local Plotly code is intentionally embedded to keep the document standalone.
        plotly_script=plotly_script,
    )


def render_reports(csv_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Render one standalone HTML per delivery day and a combined J+1/J+2 view.

    Values and the original forecast cutoff come only from the validated CSV.
    Existing output reports and audit files are never overwritten. The footer
    records presentation time; the audit separately identifies the source CSV.
    """
    source = Path(csv_path).resolve()
    rows = read_predictions(source)
    plotly_script = _find_plotly(source)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    days = sorted({row["delivery_day"] for row in rows})
    batches = [
        (f"CWE_Model_Storm_{day}.html", [row for row in rows if row["delivery_day"] == day])
        for day in days
    ]
    if len(days) > 1:
        batches.append((f"CWE_Model_Storm_J1_J2_{days[0]}.html", rows))
    if any((output / name).exists() for name, _ in batches) or (output / "render_audit.json").exists():
        raise FileExistsError("Un rapport existe déjà dans cette destination.")

    records = []
    for name, group in batches:
        path = output / name
        path.write_text(_render_html(group, plotly_script), encoding="utf-8")
        records.append({
            "html_path": str(path), "html_sha256": _sha256(path), "rows": len(group),
            "delivery_days": sorted({row["delivery_day"] for row in group}),
        })
    audit = {
        "schema_version": 1, "operation": "csv_report_render", "new_prediction_computed": False,
        "source_csv": str(source), "source_csv_sha256": _sha256(source),
        "forecast_origin_utc": rows[0]["forecast_origin_utc"], "reports": records,
        "row_count": len(rows), "delivery_days": days,
        "interval_method": INTERVAL_METHOD, "interval_calibrated": False,
    }
    audit_path = output / "render_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        **audit, "audit_json": str(audit_path), "audit_path": str(audit_path),
        "predictions_csv": str(source), "html_files": [record["html_path"] for record in records],
        "output_dir": str(output),
    }
