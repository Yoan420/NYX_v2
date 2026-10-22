"""Oracle de test : relire les valeurs exactes des rapports HTML produits.

Seul le JSON chart-data est décodé ; aucun JavaScript n'est exécuté. Le tableau
fournit les heures UTC et le graphique les valeurs sans arrondi d'affichage.
"""
from __future__ import annotations

from datetime import timedelta
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import math
import re
import shutil
from typing import Any, Iterable, TypedDict

from nyx_runtime.reporting import (
    INTERVAL_METHOD, LOCAL_TZ, VALUE_COLUMNS, ZONES, PredictionRow,
    ReportFormatError, _datetime, _finite, _sha256, _utc,
    row_key, validate_rows, write_predictions,
)

_MISSING = {"", "—", "–", "-", "N/A", "null", "None"}


class ParsedReport(TypedDict):
    """Source identity plus exact values recovered from one HTML report."""

    path: str
    name: str
    sha256: str
    delivery_days: list[str]
    forecast_origin_utc: str
    generated_at_utc: str
    rows: list[PredictionRow]
    plotly_script: str | None


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


def _strict_json(raw: str) -> Any:
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


def parse_report(html_path: str | Path) -> ParsedReport:
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
    table_rows = [
        row for row in parser.rows
        if len(row) == 9 and row[0]["text"] in ZONES and row[1]["attrs"].get("title")
    ]
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
            found = [
                trace for trace in traces
                if trace.get("name") == name and trace.get("meta", {}).get("role") != "model_interval"
            ]
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
        lower = [
            trace for trace in bounds
            if trace.get("meta", {}).get("quantile") in {"p10", "lower_bound"}
            or trace.get("meta", {}).get("bound") == "lower"
        ]
        upper = [
            trace for trace in bounds
            if trace.get("meta", {}).get("quantile") in {"p90", "upper_bound"}
            or trace.get("meta", {}).get("bound") == "upper"
        ]
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
            values = {
                column: _finite(trace["y"][index], f"{zone} {column} {index}")
                for column, trace in chosen.items()
            }
            table_columns = (
                (2, "observed_eur_mwh"), (3, "storm_eur_mwh"), (4, "prediction_eur_mwh"),
                (5, "lower_bound_eur_mwh"), (6, "upper_bound_eur_mwh"),
            )
            for col_index, column in table_columns:
                _matches_display(cell[col_index]["text"], values[column], f"{zone} {start} {column}")
            for col_index, column in ((7, "storm_eur_mwh"), (8, "prediction_eur_mwh")):
                error = (
                    None if values[column] is None or values["observed_eur_mwh"] is None
                    else values[column] - values["observed_eur_mwh"]
                )
                _matches_display(cell[col_index]["text"], error, f"{zone} {start} erreur")
            result_rows.append({
                "delivery_day": local.date().isoformat(), "zone": zone,
                "delivery_start_utc": start.isoformat(), "delivery_start_local": local.isoformat(),
                "forecast_origin_utc": origin.isoformat(),
                "horizon_day": (local.date() - origin.astimezone(LOCAL_TZ).date()).days,
                **values, "interval_method": INTERVAL_METHOD, "nominal_interval_level": 0.8,
                "interval_calibrated": "false", "source_html_sha256": digest,
                "source_generated_at_utc": generated_at, "source_html_name": path.name,
            })
    days = validate_rows(result_rows)
    return {
        "path": str(path.resolve()), "name": path.name, "sha256": digest,
        "delivery_days": days, "forecast_origin_utc": origin.isoformat(),
        "generated_at_utc": generated_at, "rows": result_rows,
        "plotly_script": parser.plotly_scripts[0] if parser.plotly_scripts else None,
    }


def import_reports(
    html_paths: Iterable[str | Path], output_dir: str | Path,
) -> dict[str, Any]:
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
    compare_columns = VALUE_COLUMNS + (
        "forecast_origin_utc", "delivery_start_local", "horizon_day",
        "interval_method", "nominal_interval_level", "interval_calibrated",
    )
    # Prefer daily sources for provenance in the union, independent of input order.
    for report in sorted(reports, key=lambda r: (len(r["delivery_days"]), r["name"])):
        for row in report["rows"]:
            key = row_key(row)
            if key in combined:
                overlaps += 1
                if any(combined[key][column] != row[column] for column in compare_columns):
                    raise ReportFormatError(f"Rapports incohérents pour {key}.")
            else:
                combined[key] = row
    all_rows = [combined[key] for key in sorted(combined)]
    all_days = validate_rows(all_rows)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    planned = (
        [report["name"] for report in reports]
        + [Path(report["name"]).with_suffix(".csv").name for report in reports]
        + ["predictions_combined.csv", "import_audit.json"]
    )
    if len(set(planned)) != len(planned) or any((output / name).exists() for name in planned):
        raise FileExistsError("La destination contient déjà un fichier à publier.")
    records = []
    for report in reports:
        destination = output / report["name"]
        shutil.copyfile(report["path"], destination)
        if _sha256(destination) != report["sha256"]:
            raise ReportFormatError("Copie HTML divergente.")
        csv_path = destination.with_suffix(".csv")
        write_predictions(csv_path, report["rows"])
        records.append({
            "html_path": str(destination), "csv_path": str(csv_path),
            "html_sha256": report["sha256"], "csv_sha256": _sha256(csv_path),
            "rows": len(report["rows"]), "delivery_days": report["delivery_days"],
            "source_generated_at_utc": report["generated_at_utc"],
        })
    combined_path = output / "predictions_combined.csv"
    write_predictions(combined_path, all_rows)
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
    audit = {
        "schema_version": 1, "operation": "reference_report_import", "new_prediction_computed": False,
        "forecast_origin_utc": reports[0]["forecast_origin_utc"], "delivery_days": all_days,
        "row_count": len(all_rows), "overlapping_rows_verified": overlaps,
        "numeric_source": "chart-data JSON (full precision)",
        "timeline_source": "real TABLE cells, UTC title attributes",
        "interval_method": INTERVAL_METHOD, "nominal_interval_level": 0.8, "interval_calibrated": False,
        "reports": records, "combined_csv": str(combined_path), "combined_csv_sha256": _sha256(combined_path),
        "plotly_asset": str(asset_path) if asset_path else None,
        "missing_observed_by_delivery_day": {
            day: sum(row["observed_eur_mwh"] is None for row in all_rows if row["delivery_day"] == day)
            for day in all_days
        },
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        **audit, "audit_json": str(audit_path), "audit_path": str(audit_path),
        "predictions_csv": str(combined_path), "html_files": [record["html_path"] for record in records],
        "csv_files": [record["csv_path"] for record in records] + [str(combined_path)],
        "output_dir": str(output),
    }
