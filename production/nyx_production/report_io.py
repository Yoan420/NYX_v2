"""Read and re-render NYX report snapshots without running a forecasting model.

Only JSON from ``script#chart-data`` is decoded. JavaScript is never evaluated
during import. Table cells supply the authoritative UTC timeline; the chart
supplies full-precision numbers. The orange envelope is explicitly indicative,
not calibrated quantiles of the final forecast. Runtime dependencies: stdlib.
"""
from __future__ import annotations

import csv
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import shutil
import statistics
from datetime import date, datetime, time, timedelta, timezone
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
_MISSING = {"", "—", "–", "-", "N/A", "null", "None"}


class ReportFormatError(ValueError):
    """A report/CSV violates its stated data or timeline contract."""


class _ReportParser(HTMLParser):
    """Collect real HTML cells and JSON while ignoring script-like HTML text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.script_attrs = None
        self.script_parts = []
        self.chart_blocks = []
        self.plotly_scripts = []
        self.visible_parts = []
        self.rows = []
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if self.script_attrs is not None:
            return
        if tag == "script":
            self.script_attrs, self.script_parts = attrs, []
        elif tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = {"tag": tag, "attrs": attrs, "parts": []}

    def handle_endtag(self, tag):
        if tag == "script" and self.script_attrs is not None:
            content = "".join(self.script_parts)
            if self.script_attrs.get("id") == "chart-data":
                if self.script_attrs.get("type") != "application/json":
                    raise ReportFormatError("chart-data doit être du JSON application/json.")
                self.chart_blocks.append(content)
            elif "plotly.js v" in content[:400] and "Licensed under the MIT license" in content[:500]:
                self.plotly_scripts.append(content)
            self.script_attrs, self.script_parts = None, []
            return
        if self.script_attrs is not None:
            return
        if tag in {"td", "th"} and self._cell is not None:
            self._cell["text"] = "".join(self._cell.pop("parts")).strip()
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self.script_attrs is not None:
            self.script_parts.append(data)
        else:
            self.visible_parts.append(data)
            if self._cell is not None:
                self._cell["parts"].append(data)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _datetime(value, label):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ReportFormatError(f"{label}: horodatage ISO invalide: {value!r}") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ReportFormatError(f"{label}: fuseau obligatoire.")
    return result


def _utc(value, label="timestamp"):
    return _datetime(value, label).astimezone(timezone.utc)


def physical_day_hours(delivery_day):
    """UTC starts of all physical hours in a Paris delivery day (23/24/25)."""
    day = date.fromisoformat(str(delivery_day))
    first = datetime.combine(day, time(), LOCAL_TZ).astimezone(timezone.utc)
    last = datetime.combine(day + timedelta(days=1), time(), LOCAL_TZ).astimezone(timezone.utc)
    return [first + timedelta(hours=i) for i in range(int((last-first).total_seconds()/3600))]


def _finite(value, label):
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ReportFormatError(f"{label}: nombre JSON fini ou null attendu.")
    return float(value)


def _csv_number(value, label):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportFormatError(f"{label}: nombre invalide.") from exc
    return _finite(number, label)


def _matches_display(displayed, value, label):
    if displayed in _MISSING:
        if value is not None:
            raise ReportFormatError(f"{label}: tableau manquant mais graphique renseigné.")
    elif value is None:
        raise ReportFormatError(f"{label}: graphique manquant mais tableau renseigné.")
    else:
        try:
            rounded = float(displayed.replace("\u202f", "").replace("\xa0", ""))
        except ValueError as exc:
            raise ReportFormatError(f"{label}: cellule numérique invalide.") from exc
        if not math.isfinite(rounded) or abs(rounded - value) > 0.005000001:
            raise ReportFormatError(f"{label}: tableau et graphique divergent.")


def _strict_json(raw):
    def reject_constant(value):
        raise ReportFormatError(f"Constante JSON non finie interdite: {value}")
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReportFormatError(f"Clé JSON répétée: {key}")
            result[key] = value
        return result
    try:
        return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ReportFormatError("chart-data contient un JSON invalide.") from exc


def _validate_rows(rows):
    if not rows:
        raise ReportFormatError("Aucune ligne de prévision.")
    keys = set()
    groups = {}
    origins = set()
    for row in rows:
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
        origins.add(origin)
        origin_local = origin.astimezone(LOCAL_TZ)
        if (origin_local.hour, origin_local.minute, origin_local.second, origin_local.microsecond) != (8, 0, 0, 0):
            raise ReportFormatError("Le cutoff doit correspondre à 08:00 Europe/Paris.")
        horizon = (local.date() - origin_local.date()).days
        if horizon not in (1, 2) or int(row["horizon_day"]) not in (1, 2):
            raise ReportFormatError("horizon_day doit rester dans le périmètre J+1/J+2 (1 ou 2).")
        if int(row["horizon_day"]) != horizon:
            raise ReportFormatError("horizon_day incohérent avec le cutoff commun.")
        key = zone, start
        if key in keys:
            raise ReportFormatError(f"Heure dupliquée: {zone} {start.isoformat()}")
        keys.add(key)
        groups.setdefault((day, zone), []).append(start)
        for column in VALUE_COLUMNS:
            row[column] = _csv_number(row[column], column)
        for column in ("prediction_eur_mwh", "lower_bound_eur_mwh", "upper_bound_eur_mwh"):
            if row[column] is None:
                raise ReportFormatError(f"{zone} {start.isoformat()}: {column} obligatoire et fini.")
        point, low, high = row["prediction_eur_mwh"], row["lower_bound_eur_mwh"], row["upper_bound_eur_mwh"]
        if low is not None and high is not None and low > high:
            raise ReportFormatError("Bornes indicatives inversées.")
        if point is not None and ((low is not None and point < low) or (high is not None and point > high)):
            raise ReportFormatError("L'enveloppe indicative doit contenir le prix final.")
        if row["interval_method"] != INTERVAL_METHOD or str(row["interval_calibrated"]).lower() != "false":
            raise ReportFormatError("L'enveloppe source doit rester indicative et non calibrée.")
        if float(row["nominal_interval_level"]) != 0.8:
            raise ReportFormatError("Niveau nominal attendu: 0,8.")
    if len(origins) != 1:
        raise ReportFormatError("Les lignes doivent conserver un cutoff commun.")
    days = sorted({day for day, _ in groups})
    if len(days) > 2:
        raise ReportFormatError("Au plus deux jours de livraison J+1/J+2 sont autorisés.")
    if len(days) == 2:
        origin_day = next(iter(origins)).astimezone(LOCAL_TZ).date()
        if [(date.fromisoformat(day)-origin_day).days for day in days] != [1, 2]:
            raise ReportFormatError("Un rapport combiné doit couvrir J+1 et J+2 consécutifs au cutoff commun.")
    for day in days:
        for zone in ZONES:
            expected = physical_day_hours(day)
            if groups.get((day, zone)) != expected:
                raise ReportFormatError(f"Journée physique incomplète ou désordonnée: {day} {zone}.")
    if any((date.fromisoformat(b)-date.fromisoformat(a)).days != 1 for a, b in zip(days, days[1:])):
        raise ReportFormatError("Jours de livraison non contigus.")
    return days


def parse_report(html_path):
    """Parse a reference snapshot; returns rows, metadata, and its Plotly asset."""
    path = Path(html_path)
    raw = path.read_bytes()
    try:
        source = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ReportFormatError("Rapport HTML UTF-8 requis.") from exc
    parser = _ReportParser()
    parser.feed(source)
    parser.close()
    if len(parser.chart_blocks) != 1:
        raise ReportFormatError("Un seul script chart-data est requis.")
    figure = _strict_json(parser.chart_blocks[0])
    if not isinstance(figure, dict) or not isinstance(figure.get("data"), list):
        raise ReportFormatError("Structure chart-data non reconnue.")
    visible = " ".join(parser.visible_parts)
    iso_pattern = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))"
    cutoffs = set(re.findall(r"original information cutoff\s+" + iso_pattern, visible))
    generated = set(re.findall(r"Generated\s+" + iso_pattern, visible))
    if len(cutoffs) != 1 or len(generated) != 1:
        raise ReportFormatError("Cutoff commun ou date de génération absent/ambigu.")
    origin = _utc(next(iter(cutoffs)), "original information cutoff")
    generated_at = _utc(next(iter(generated)), "Generated").isoformat()
    if _utc(generated_at) < origin:
        raise ReportFormatError("Génération du rapport antérieure à son cutoff.")
    if "coverage is not calibrated for the final forecast" not in visible:
        raise ReportFormatError("Mention explicite des bornes non calibrées absente.")
    table_rows = [row for row in parser.rows if len(row) == 9 and row[0]["text"] in ZONES and row[1]["attrs"].get("title")]
    if not table_rows:
        raise ReportFormatError("Tableau horaire avec timestamps UTC absent.")
    zone_order = list(dict.fromkeys(row[0]["text"] for row in table_rows))
    if set(zone_order) != set(ZONES):
        raise ReportFormatError("Les quatre zones CWE sont requises.")
    zone_annotations = []
    for ann in figure.get("layout", {}).get("annotations", []):
        label = re.sub(r"<[^>]*>", "", str(ann.get("text", ""))).strip()
        if label in ZONES:
            zone_annotations.append((float(ann.get("x", 0)), label))
    if zone_annotations and [z for _, z in sorted(zone_annotations)] != zone_order:
        raise ReportFormatError("Ordre des zones du graphique et du tableau divergent.")
    interval_axes = {}
    for trace in figure["data"]:
        if trace.get("meta", {}).get("role") == "model_interval":
            interval_axes.setdefault(trace.get("xaxis", "x"), []).append(trace)
    axes = sorted(interval_axes, key=lambda x: int(x[1:] or "1"))
    if len(axes) != len(zone_order):
        raise ReportFormatError("Une enveloppe par zone est requise.")
    digest = hashlib.sha256(raw).hexdigest()
    result_rows = []
    for zone, axis in zip(zone_order, axes):
        cells = [r for r in table_rows if r[0]["text"] == zone]
        traces = [t for t in figure["data"] if t.get("xaxis", "x") == axis]
        chosen = {}
        for name, column in (("NYX", "prediction_eur_mwh"), ("Storm", "storm_eur_mwh")):
            found = [t for t in traces if t.get("name") == name and t.get("meta", {}).get("role") != "model_interval"]
            if len(found) != 1:
                raise ReportFormatError(f"{zone}: trace {name} absente ou dupliquée.")
            chosen[column] = found[0]
        observed = [t for t in traces if str(t.get("name", "")).startswith("Observed")]
        if len(observed) != 1:
            raise ReportFormatError(f"{zone}: trace Observed absente ou dupliquée.")
        chosen["observed_eur_mwh"] = observed[0]
        bounds = interval_axes[axis]
        if len(bounds) != 2:
            raise ReportFormatError(f"{zone}: deux bornes requises.")
        lower = [t for t in bounds if t.get("meta", {}).get("quantile") in {"p10", "lower_bound"} or t.get("meta", {}).get("bound") == "lower"]
        upper = [t for t in bounds if t.get("meta", {}).get("quantile") in {"p90", "upper_bound"} or t.get("meta", {}).get("bound") == "upper"]
        if len(lower) != 1 or len(upper) != 1:
            raise ReportFormatError(f"{zone}: identités des bornes ambiguës.")
        chosen["lower_bound_eur_mwh"], chosen["upper_bound_eur_mwh"] = lower[0], upper[0]
        expected_x = list(range(len(cells)))
        for column, trace in chosen.items():
            if trace.get("x") != expected_x or len(trace.get("y", [])) != len(cells):
                raise ReportFormatError(f"{zone} {column}: axe graphique incompatible avec le tableau.")
        for index, cell in enumerate(cells):
            start = _utc(cell[1]["attrs"]["title"], "table title")
            if _datetime(cell[1]["attrs"]["title"], "table title").utcoffset() != timedelta(0):
                raise ReportFormatError("Le title de la table doit être en UTC.")
            local = start.astimezone(LOCAL_TZ)
            local_label = local.strftime("%H:%M %Z")
            if not cell[1]["text"].endswith(local_label):
                raise ReportFormatError("Libellé horaire local incompatible avec le timestamp UTC.")
            values = {column: _finite(trace["y"][index], f"{zone} {column} {index}") for column, trace in chosen.items()}
            for col_index, column in ((2, "observed_eur_mwh"), (3, "storm_eur_mwh"), (4, "prediction_eur_mwh"), (5, "lower_bound_eur_mwh"), (6, "upper_bound_eur_mwh")):
                _matches_display(cell[col_index]["text"], values[column], f"{zone} {start} {column}")
            for col_index, column in ((7, "storm_eur_mwh"), (8, "prediction_eur_mwh")):
                error = None if values[column] is None or values["observed_eur_mwh"] is None else values[column] - values["observed_eur_mwh"]
                _matches_display(cell[col_index]["text"], error, f"{zone} {start} erreur")
            result_rows.append({
                "delivery_day": local.date().isoformat(), "zone": zone,
                "delivery_start_utc": start.isoformat(), "delivery_start_local": local.isoformat(),
                "forecast_origin_utc": origin.isoformat(), "horizon_day": (local.date()-origin.astimezone(LOCAL_TZ).date()).days,
                **values, "interval_method": INTERVAL_METHOD, "nominal_interval_level": 0.8,
                "interval_calibrated": "false", "source_html_sha256": digest,
                "source_generated_at_utc": generated_at, "source_html_name": path.name,
            })
    days = _validate_rows(result_rows)
    return {"path": str(path.resolve()), "name": path.name, "sha256": digest,
            "delivery_days": days, "forecast_origin_utc": origin.isoformat(),
            "generated_at_utc": generated_at, "rows": result_rows,
            "plotly_script": parser.plotly_scripts[0] if parser.plotly_scripts else None}


def _row_key(row):
    return row["delivery_day"], row["zone"], row["delivery_start_utc"]


def _write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def import_reports(html_paths, output_dir):
    """Validate snapshots, copy exact HTML, export each CSV and a deduplicated union.

    Overlapping reports must contain exactly the same full-precision numbers and
    forecast origin. Source generation dates may differ. No model is run.
    """
    paths = [Path(p) for p in html_paths]
    if not paths or len({p.name.casefold() for p in paths}) != len(paths):
        raise ReportFormatError("Rapports absents ou noms de fichiers dupliqués.")
    reports = [parse_report(p) for p in paths]
    if len({r["forecast_origin_utc"] for r in reports}) != 1:
        raise ReportFormatError("Cutoffs différents entre rapports.")
    combined = {}
    overlaps = 0
    compare_columns = VALUE_COLUMNS + ("forecast_origin_utc", "delivery_start_local", "horizon_day", "interval_method", "nominal_interval_level", "interval_calibrated")
    # Prefer daily sources for provenance in the union, independent of input order.
    for report in sorted(reports, key=lambda r: (len(r["delivery_days"]), r["name"])):
        for row in report["rows"]:
            key = _row_key(row)
            if key in combined:
                overlaps += 1
                if any(combined[key][column] != row[column] for column in compare_columns):
                    raise ReportFormatError(f"Rapports incohérents pour {key}.")
            else:
                combined[key] = row
    all_rows = [combined[key] for key in sorted(combined)]
    all_days = _validate_rows(all_rows)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    planned = [r["name"] for r in reports] + [Path(r["name"]).with_suffix(".csv").name for r in reports] + ["predictions_combined.csv", "import_audit.json"]
    if len(set(planned)) != len(planned) or any((output/name).exists() for name in planned):
        raise FileExistsError("La destination contient déjà un fichier à publier.")
    records = []
    for report in reports:
        destination = output / report["name"]
        shutil.copyfile(report["path"], destination)
        if _sha256(destination) != report["sha256"]:
            raise ReportFormatError("Copie HTML divergente.")
        csv_path = destination.with_suffix(".csv")
        _write_csv(csv_path, report["rows"])
        records.append({"html_path": str(destination), "csv_path": str(csv_path), "html_sha256": report["sha256"],
                        "csv_sha256": _sha256(csv_path), "rows": len(report["rows"]), "delivery_days": report["delivery_days"],
                        "source_generated_at_utc": report["generated_at_utc"]})
    combined_path = output / "predictions_combined.csv"
    _write_csv(combined_path, all_rows)
    asset = next((r["plotly_script"] for r in reports if r["plotly_script"]), None)
    asset_path = None
    if asset:
        packaged_asset = Path(__file__).resolve().parents[1] / "assets" / "plotly.min.js"
        asset_digest = hashlib.sha256(asset.encode("utf-8")).hexdigest()
        if packaged_asset.is_file() and _sha256(packaged_asset) == asset_digest:
            # Avoid duplicating 4.85 MB in every run when the package already
            # ships the exact source asset. Standalone rendered HTML embeds it.
            asset_path = packaged_asset
        else:
            asset_dir = output / "assets"
            asset_dir.mkdir(exist_ok=True)
            asset_path = asset_dir / "plotly.min.js"
            if asset_path.exists() and _sha256(asset_path) != asset_digest:
                raise FileExistsError("Un asset Plotly différent existe déjà.")
            if not asset_path.exists():
                asset_path.write_text(asset, encoding="utf-8")
    audit_path = output / "import_audit.json"
    audit = {"schema_version": 1, "operation": "reference_report_import", "new_prediction_computed": False,
             "forecast_origin_utc": reports[0]["forecast_origin_utc"], "delivery_days": all_days,
             "row_count": len(all_rows), "overlapping_rows_verified": overlaps,
             "numeric_source": "chart-data JSON (full precision)", "timeline_source": "real TABLE cells, UTC title attributes",
             "interval_method": INTERVAL_METHOD, "nominal_interval_level": 0.8, "interval_calibrated": False,
             "reports": records, "combined_csv": str(combined_path), "combined_csv_sha256": _sha256(combined_path),
             "plotly_asset": str(asset_path) if asset_path else None,
             "missing_observed_by_delivery_day": {d: sum(r["observed_eur_mwh"] is None for r in all_rows if r["delivery_day"] == d) for d in all_days}}
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return {**audit, "audit_json": str(audit_path), "audit_path": str(audit_path),
            "predictions_csv": str(combined_path), "html_files": [r["html_path"] for r in records],
            "csv_files": [r["csv_path"] for r in records] + [str(combined_path)], "output_dir": str(output)}


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        # source_html_name is optional provenance metadata; the canonical
        # prediction schema also accepts fresh model CSVs without that field.
        missing = (set(CSV_COLUMNS) - {"source_html_name"}) - set(reader.fieldnames or [])
        if missing:
            raise ReportFormatError(f"Colonnes CSV absentes: {', '.join(sorted(missing))}")
        rows = list(reader)
    for row in rows:
        row.setdefault("source_html_name", "")
        row["horizon_day"] = int(row["horizon_day"])
        row["nominal_interval_level"] = float(row["nominal_interval_level"])
    rows.sort(key=_row_key)
    _validate_rows(rows)
    return rows


def _find_plotly(csv_path):
    candidates = [Path(csv_path).parent / "assets" / "plotly.min.js",
                  Path(__file__).resolve().parents[1] / "assets" / "plotly.min.js"]
    for path in candidates:
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            if "plotly.js v" in content[:400] and "Licensed under the MIT license" in content[:500]:
                return content
    # The cloud image already installs Plotly. Embed its local JavaScript so
    # generated reports remain standalone without bundling a private report.
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as error:
        raise FileNotFoundError("Plotly local absent : installer plotly ou fournir assets/plotly.min.js.") from error
    return get_plotlyjs()


def _display(value):
    return "—" if value is None else f"{value:.2f}"


def _complete_mean(rows, column):
    values = [r[column] for r in rows]
    return statistics.fmean(values) if values and all(v is not None for v in values) else None


def _render_html(rows, plotly_script):
    days = sorted({r["delivery_day"] for r in rows})
    title = " → ".join([days[0], days[-1]]) if len(days)>1 else days[0]
    origin = rows[0]["forecast_origin_utc"]
    traces, annotations, layout, cards = [], [], {}, []
    table_lines = []
    for zi, zone in enumerate(ZONES):
        group = [r for r in rows if r["zone"] == zone]
        x = list(range(len(group)))
        labels = [_utc(r["delivery_start_utc"]).astimezone(LOCAL_TZ).strftime("%Y-%m-%d · %H:%M %Z") for r in group]
        suffix = "" if zi == 0 else str(zi+1)
        axis, yax = "x"+suffix, "y"+suffix
        left, right = zi/4+0.015, (zi+1)/4-0.015
        layout["xaxis"+suffix] = {"domain": [left, right], "anchor": yax, "tickmode": "array", "tickvals": x[::max(1,len(x)//5)], "ticktext": [labels[i].split(" · ")[-1] for i in x[::max(1,len(x)//5)]], "gridcolor": "#343b40"}
        layout["yaxis"+suffix] = {"domain": [.37, 1], "anchor": axis, "title": "€/MWh" if zi==0 else "", "gridcolor": "#343b40"}
        error_axis = "x"+str(zi+5)
        error_y = "y"+str(zi+5)
        layout["xaxis"+str(zi+5)] = {**layout["xaxis"+suffix], "anchor": error_y}
        layout["yaxis"+str(zi+5)] = {"domain": [0,.24], "anchor": error_axis, "gridcolor": "#343b40", "zerolinecolor": "#888"}
        annotations.append({"x": (left+right)/2, "y": 1.05, "xref": "paper", "yref": "paper", "text": "<b>"+zone+"</b>", "showarrow": False})
        for bound, col in (("lower","lower_bound_eur_mwh"),("upper","upper_bound_eur_mwh")):
            traces.append({"type":"scatter", "mode":"lines", "x":x, "y":[r[col] for r in group], "xaxis":axis, "yaxis":yax,
                           "name":"Indicative envelope (80% nominal)", "showlegend": zi==0 and bound=="upper", "legendgroup":"envelope",
                           "line":{"color":"rgba(228,139,35,.35)","width":.6}, "fill":"tonexty" if bound=="upper" else "none", "fillcolor":"rgba(228,139,35,.18)",
                           "meta":{"role":"model_interval", "bound":bound}, "hoverinfo":"skip", "connectgaps":False})
        for name,col,color in (("Observed (EPEX)","observed_eur_mwh","#c084fc"),("Storm","storm_eur_mwh","#00a8ff"),("NYX","prediction_eur_mwh","#e48b23")):
            traces.append({"type":"scatter", "mode":"lines", "x":x, "y":[r[col] for r in group], "xaxis":axis, "yaxis":yax,
                           "name":name, "legendgroup":name, "showlegend":zi==0, "line":{"color":color,"width":2}, "customdata":labels,
                           "hovertemplate":"%{customdata}<br>"+name+": %{y:.2f} €/MWh<extra></extra>", "connectgaps":False})
            if name != "Observed (EPEX)":
                errors = [None if r[col] is None or r["observed_eur_mwh"] is None else r[col]-r["observed_eur_mwh"] for r in group]
                traces.append({"type":"scatter","mode":"lines","x":x,"y":errors,"xaxis":error_axis,"yaxis":error_y,"name":name,"legendgroup":name,"showlegend":False,"line":{"color":color,"width":1.5},"customdata":labels,"hovertemplate":"%{customdata}<br>Error: %{y:.2f} €/MWh<extra></extra>","connectgaps":False})
        for day in days:
            daily = [r for r in group if r["delivery_day"]==day]
            cards.append(f'<article><h3>{zone} · {html.escape(day)}</h3><p>Observed (EPEX) <strong>{_display(_complete_mean(daily,"observed_eur_mwh"))}</strong></p><p>Storm <strong>{_display(_complete_mean(daily,"storm_eur_mwh"))}</strong></p><p>NYX <strong>{_display(_complete_mean(daily,"prediction_eur_mwh"))}</strong></p><small>{len(daily)} physical hours · €/MWh</small></article>')
        for row,label in zip(group,labels):
            vals=[row["observed_eur_mwh"], row["storm_eur_mwh"], row["prediction_eur_mwh"], row["lower_bound_eur_mwh"], row["upper_bound_eur_mwh"]]
            vals += [None if row[c] is None or row["observed_eur_mwh"] is None else row[c]-row["observed_eur_mwh"] for c in ("storm_eur_mwh","prediction_eur_mwh")]
            table_lines.append('<tr><th scope="row">'+zone+'</th><td title="'+html.escape(row["delivery_start_utc"],quote=True)+'">'+html.escape(label)+'</td>'+''.join('<td>'+_display(v)+'</td>' for v in vals)+'</tr>')
    layout.update({"height":650,"paper_bgcolor":"#1c2226","plot_bgcolor":"#1c2226","font":{"color":"#e7e9ea","family":"Arial"},"margin":{"t":55,"l":45,"r":20,"b":50},"hovermode":"closest","annotations":annotations,"legend":{"orientation":"h","y":1.17},"uirevision":"csv-snapshot"})
    figure = json.dumps({"data":traces,"layout":layout},ensure_ascii=False,allow_nan=False).replace("</", "<\\/")
    generated = datetime.now(timezone.utc).isoformat()
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CWE · NYX &amp; Storm · {html.escape(title)}</title>
<style>body{{margin:0;background:#171d21;color:#e7e9ea;font:15px Arial,sans-serif}}main{{max-width:1500px;margin:auto;padding:26px}}h1{{font-size:27px}}.date{{color:#e6ad58}}.note,small{{color:#adb6bb;line-height:1.6}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}article{{border:1px solid #384148;border-radius:10px;padding:16px;background:#20272c}}article p{{display:flex;justify-content:space-between}}button{{background:#273137;color:#ddd;border:1px solid #657078;padding:9px 20px;margin:8px 4px 12px 0;cursor:pointer}}button.active{{color:#e6ad58;border-color:#e6ad58}}[hidden]{{display:none!important}}.scroll{{overflow:auto;max-height:680px}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:8px;border-bottom:1px solid #384148;text-align:right}}th:first-child{{text-align:left}}thead{{position:sticky;top:0;background:#242c31}}footer{{padding-top:28px;color:#97a2a9;font-size:12px}}@media(max-width:800px){{.cards{{grid-template-columns:repeat(2,1fr)}}main{{padding:12px}}}}</style>
<script>{plotly_script}</script></head><body><main><h1>CWE Day-Ahead · Daily Models Performance <span class="date">{html.escape(title)}</span></h1>
<p class="note">NYX · delivery {html.escape(title)} · original information cutoff {html.escape(origin)}. EPEX observations and Storm are report-only comparators. Recreated from the exported CSV; no model or new prediction was computed.</p>
<h2>Daily Snapshot</h2><section class="cards">{''.join(cards)}</section><h2>Hourly Forecasts per Zone</h2><nav><button class="active" data-tab="graph">GRAPH</button><button data-tab="table">TABLE</button><button data-tab="info">INFO</button></nav>
<section id="graph"><div id="forecast-chart"></div></section><section id="table" hidden><div class="scroll"><table><thead><tr><th>Zone</th><th>Hour</th><th>Observed (EPEX)</th><th>Storm</th><th>NYX</th><th>Lower bound</th><th>Upper bound</th><th>Storm error</th><th>NYX error</th></tr></thead><tbody>{''.join(table_lines)}</tbody></table></div></section>
<section id="info" hidden><h3>NYX, Storm and observed prices</h3><p>The NYX line is a point forecast. The orange band is an indicative prediction envelope derived from the reference distribution and widened to contain the final price. Its nominal level is 80%; coverage is not calibrated for the final forecast. The bounds are not calibrated NYX quantiles.</p><p>Errors are forecast − observed. Daily means require every physical hour of the delivery day. Missing values remain missing. Local delivery times distinguish repeated daylight-saving hours by their timezone.</p><p>This report reproduces CSV values; it does not refit NYX or refresh observations.</p></section>
<footer>NYX · Storm · Observed — CWE day-ahead<br>Generated {generated} · standalone HTML</footer></main><script id="chart-data" type="application/json">{figure}</script>
<script>const figure=JSON.parse(document.getElementById('chart-data').textContent);Plotly.newPlot('forecast-chart',figure.data,figure.layout,{{responsive:true,displaylogo:false}});document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x===b));['graph','table','info'].forEach(id=>document.getElementById(id).hidden=id!==b.dataset.tab);if(b.dataset.tab==='graph')Plotly.Plots.resize('forecast-chart');}});</script></body></html>'''


def render_reports(csv_path, output_dir):
    """Recreate daily + combined standalone HTML from an exported CSV only.

    A local Plotly asset is embedded. The CSV's values and original forecast
    cutoff remain unchanged; the footer is the new presentation generation time.
    """
    source = Path(csv_path).resolve()
    rows = _read_csv(source)
    plotly_script = _find_plotly(source)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    days = sorted({r["delivery_day"] for r in rows})
    batches = [(f"CWE_Model_Storm_{day}.html", [r for r in rows if r["delivery_day"]==day]) for day in days]
    if len(days)>1:
        batches.append((f"CWE_Model_Storm_J1_J2_{days[0]}.html", rows))
    if any((output/name).exists() for name, _ in batches) or (output/"render_audit.json").exists():
        raise FileExistsError("Un rapport existe déjà dans cette destination.")
    records = []
    for name, group in batches:
        path = output/name
        path.write_text(_render_html(group,plotly_script),encoding="utf-8")
        records.append({"html_path":str(path),"html_sha256":_sha256(path),"rows":len(group),"delivery_days":sorted({r["delivery_day"] for r in group})})
    audit = {"schema_version":1,"operation":"csv_report_render","new_prediction_computed":False,"source_csv":str(source),"source_csv_sha256":_sha256(source),"forecast_origin_utc":rows[0]["forecast_origin_utc"],"reports":records,"row_count":len(rows),"delivery_days":days,"interval_method":INTERVAL_METHOD,"interval_calibrated":False}
    audit_path=output/"render_audit.json"
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return {**audit,"audit_json":str(audit_path),"audit_path":str(audit_path),
            "predictions_csv":str(source),"html_files":[r["html_path"] for r in records],"output_dir":str(output)}
