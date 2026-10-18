"""Past-only research comparison of NYX OOF price experts. Never edits production.

The input must contain independently generated daily predictions, not predictions
made retrospectively with today's models. Each price label has an explicit
availability timestamp. Storm is opened only by the scoring functions.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROTOCOL = "nyx_past_only_adaptive_research_v1"
TZ = "Europe/Paris"
COUNTRIES = ("FR", "DE", "BE", "NL")
COMPONENTS = ("reference_price", "expert_fr_residual", "expert_cwe_residual", "expert_cwe_absolute")
REQUIRED = ("country", "delivery_utc", "forecast_origin_utc", "horizon_day", "target_price",
            "target_available_utc", "reference_price")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc(series: pd.Series, name: str) -> pd.Series:
    # Do not silently interpret timezone-naive timestamps as UTC.
    values = series.astype(str)
    require(values.str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all(),
            f"{name}: explicit UTC offset required")
    result = pd.to_datetime(values, utc=True, errors="raise")
    require(result.notna().all(), f"{name}: missing timestamp")
    return result


def frozen_current(frame: pd.DataFrame) -> np.ndarray:
    country = frame["country"].to_numpy()
    reference = frame["reference_price"].to_numpy(float)
    proposed = np.where(country == "FR", frame["expert_fr_residual"],
                        (frame["expert_cwe_residual"] + frame["expert_cwe_absolute"]) / 2.)
    return np.where((country == "NL") | (np.abs(proposed - reference) >= 20.), proposed, reference)


def validate_input(frame: pd.DataFrame, component_columns: tuple[str, ...] = COMPONENTS,
                   control_column: str = "nyx_current", scope: str = "final_experts") -> pd.DataFrame:
    require(scope in ("final_experts", "reference_component_pilot"), "Unknown research scope")
    require(2 <= len(component_columns) <= 6 and len(set(component_columns)) == len(component_columns),
            "Two to six distinct forecast components required")
    required = set((*REQUIRED, *component_columns, control_column))
    require(required <= set(frame.columns), "Missing columns: " + ", ".join(sorted(required - set(frame.columns))))
    frame = frame.copy()
    require(frame["country"].isin(COUNTRIES).all(), "Only FR/DE/BE/NL are supported")
    for name in ("delivery_utc", "forecast_origin_utc", "target_available_utc"):
        frame[name] = _utc(frame[name], name)
    horizon = pd.to_numeric(frame["horizon_day"], errors="raise")
    require(horizon.isin((1, 2)).all(), "horizon_day must be 1 or 2")
    frame["horizon_day"] = horizon.astype(int)
    for name in ("target_price", *component_columns, "reference_price", control_column):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        require(np.isfinite(frame[name]).all(), f"{name}: finite values required")
    frame["nyx_current"] = frame[control_column]
    if "storm_price" not in frame:
        frame["storm_price"] = np.nan
    frame["storm_price"] = pd.to_numeric(frame["storm_price"], errors="raise")
    require(not np.isinf(frame["storm_price"]).any(), "storm_price cannot be infinite")
    require(not frame.duplicated(["country", "horizon_day", "delivery_utc"]).any(),
            "Duplicate delivery hour for a country/horizon")
    require((frame["forecast_origin_utc"] < frame["delivery_utc"]).all(),
            "OOF origin must precede delivery")
    require((frame["target_available_utc"] > frame["forecast_origin_utc"]).all(),
            "A scored target was already available at its own forecast origin")
    local = frame["delivery_utc"].dt.tz_convert(TZ)
    frame["delivery_day"] = local.dt.date
    frame["delivery_hour"] = local.dt.hour
    origin_days = frame["forecast_origin_utc"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    expected = (origin_days + pd.to_timedelta(frame["horizon_day"], unit="D")).dt.date
    require((expected == frame["delivery_day"]).all(), "Horizon does not match Paris calendar dates")
    for key, group in frame.groupby(["country", "horizon_day", "delivery_day"], sort=False):
        require(group["forecast_origin_utc"].nunique() == 1, f"{key}: mixed forecast origins")
        start = pd.Timestamp(key[2]).tz_localize(TZ).tz_convert("UTC")
        stop = (pd.Timestamp(key[2]) + pd.Timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
        grid = pd.date_range(start, stop, freq="h", inclusive="left")
        require(pd.DatetimeIndex(group["delivery_utc"].sort_values()).equals(grid),
                f"{key}: incomplete physical-hour grid (23/24/25 expected)")
    if scope == "final_experts":
        require(component_columns == COMPONENTS and control_column == "nyx_current",
                "Final-expert study requires the frozen production component schema")
        require(np.allclose(frame["nyx_current"], frozen_current(frame), atol=1e-7, rtol=1e-10),
                "nyx_current does not reproduce the frozen production composition")
    # Cross-country features use forecasts at the identical timestamp and origin.
    # Missing peer countries are explicit; neither labels nor Storm enter here.
    frame["_peer_point"] = ((frame["expert_cwe_residual"] + frame["expert_cwe_absolute"]) / 2.
                            if scope == "final_experts" else frame[list(component_columns)].mean(axis=1))
    peers = frame.groupby(["horizon_day", "forecast_origin_utc", "delivery_utc"])["_peer_point"].agg(["mean", "count"])
    peers.columns = ["_cwe_point", "_cwe_count"]
    frame = frame.merge(peers, left_on=["horizon_day", "forecast_origin_utc", "delivery_utc"],
                        right_index=True, validate="many_to_one")
    frame = frame.sort_values(["country", "horizon_day", "forecast_origin_utc", "delivery_utc"]).reset_index(drop=True)
    groups = ["country", "horizon_day", "forecast_origin_utc"]
    frame["_own_ramp"] = frame.groupby(groups)["_peer_point"].diff().fillna(0.)
    frame["_cwe_ramp"] = frame.groupby(groups)["_cwe_point"].diff().fillna(0.)
    return frame


def prior_weights(country: str) -> np.ndarray:
    return np.array([.5, .5, 0., 0.]) if country == "FR" else np.array(
        [0., 0., .5, .5] if country == "NL" else [.5, 0., .25, .25])


def convex_weights(x: np.ndarray, y: np.ndarray, prior: np.ndarray) -> np.ndarray:
    # Enumerate the <=63 faces of the small simplex. A strictly positive ridge
    # makes each face solution unique and avoids numerical SLSQP failure on spikes.
    n = len(y)
    gram = x.T @ x / n
    cross = x.T @ y / n
    penalty = 25.  # Fixed shrinkage: five EUR/MWh squared. Not tuned on holdout.
    q = gram + penalty * np.eye(len(prior))
    b = cross + penalty * prior
    best, objective = prior, float("inf")
    for mask in range(1, 1 << len(prior)):
        active = np.array([i for i in range(len(prior)) if mask & (1 << i)])
        count = len(active)
        kkt = np.zeros((count + 1, count + 1))
        kkt[:count, :count] = q[np.ix_(active, active)]
        kkt[count, :count], kkt[:count, count] = 1., 1.
        solution = np.linalg.solve(kkt, np.r_[b[active], 1.])[:count]
        if np.any(solution < -1e-9):
            continue
        weights = np.zeros(len(prior))
        weights[active] = np.maximum(solution, 0.)
        weights /= weights.sum()
        value = float(weights @ q @ weights - 2. * b @ weights)
        if value < objective:
            best, objective = weights, value
    require(np.isfinite(objective) and np.isfinite(best).all(), "Convex solver failed")
    return best


def bias_features(frame: pd.DataFrame, point: np.ndarray) -> np.ndarray:
    hour = frame["delivery_hour"].to_numpy(float)
    return np.column_stack([np.ones(len(frame)), np.sin(2. * np.pi * hour / 24.),
                            np.cos(2. * np.pi * hour / 24.), point,
                            point - frame["reference_price"].to_numpy(float),
                            frame["_own_ramp"], frame["_cwe_point"], frame["_cwe_ramp"],
                            frame["_cwe_count"]])


def conditional_bias(x: np.ndarray, residual: np.ndarray, future: np.ndarray) -> np.ndarray:
    centre = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.
    centre[0], scale[0] = 0., 1.
    train = (x - centre) / scale
    target = (future - centre) / scale
    penalty = np.eye(x.shape[1]) * 20.
    penalty[0, 0] = 5.
    beta = np.linalg.solve(train.T @ train + penalty, train.T @ residual)
    return np.clip(target @ beta, -20., 20.)


def interval_radius(errors: np.ndarray, alpha: float, minimum: int = 168) -> float:
    values = np.abs(np.asarray(errors, dtype=float))
    values = values[np.isfinite(values)]
    if len(values) < minimum:
        return float("nan")
    rank = min(len(values), int(math.ceil((len(values) + 1) * (1. - alpha))))
    return float(np.partition(values, rank - 1)[rank - 1])


def recent_days(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    available = sorted(frame["delivery_day"].unique())
    return frame[frame["delivery_day"].isin(available[-days:])]


def generate_predictions(frame: pd.DataFrame, windows: tuple[int, ...] = (30, 60, 90),
                         minimum_days: int = 30, alpha: float = .2,
                         component_columns: tuple[str, ...] = COMPONENTS) -> tuple[pd.DataFrame, list[dict]]:
    """No Storm access. Past residuals always use their original prequential point."""
    result = frame.copy()
    algorithms = ["current"] + [f"adaptive_{days}" for days in windows] + [f"adaptive_bias_{days}" for days in windows]
    for algorithm in algorithms:
        for prefix in ("prediction_", "lower_", "upper_"):
            result[prefix + algorithm] = np.nan
    result["prediction_current"] = result["nyx_current"]
    result["adaptive_ready"] = False
    audit: list[dict] = []
    for (country, horizon), country_rows in result.groupby(["country", "horizon_day"], sort=False):
        country_indices = country_rows.index.to_numpy()
        for origin, current_rows in country_rows.groupby("forecast_origin_utc", sort=True):
            current_indices = current_rows.index.to_numpy()
            day = current_rows["delivery_day"].iloc[0]
            # Origin and target availability are independent gates. Later-released
            # revisions cannot enter merely because their delivery date is old.
            eligible = result.loc[country_indices]
            eligible = eligible[(eligible["forecast_origin_utc"] < origin) &
                                (eligible["delivery_day"] < day) &
                                (eligible["target_available_utc"] <= origin)]
            expected_counts = country_rows.groupby("delivery_day").size()
            counts = eligible.groupby("delivery_day").size()
            complete_days = [value for value, count in counts.items() if count == expected_counts[value]]
            eligible = eligible[eligible["delivery_day"].isin(complete_days)]
            record: dict[str, Any] = {"country": country, "horizon_day": int(horizon),
                                     "forecast_origin_utc": origin.isoformat(), "delivery_day": str(day),
                                     "available_history_days": int(eligible["delivery_day"].nunique()),
                                     "windows": {}}
            for days in windows:
                history = recent_days(eligible, days)
                ready = history["delivery_day"].nunique() >= minimum_days
                weights = (prior_weights(country) if component_columns == COMPONENTS
                           else np.ones(len(component_columns)) / len(component_columns))
                if ready:
                    weights = convex_weights(history[list(component_columns)].to_numpy(float),
                                             history["target_price"].to_numpy(float), weights)
                base_name, bias_name = f"adaptive_{days}", f"adaptive_bias_{days}"
                point = current_rows[list(component_columns)].to_numpy(float) @ weights
                result.loc[current_indices, "prediction_" + base_name] = point
                saved_history = history[np.isfinite(history["prediction_" + base_name])]
                bias = np.zeros(len(point))
                bias_ready = ready and saved_history["delivery_day"].nunique() >= minimum_days
                if bias_ready:
                    original_points = saved_history["prediction_" + base_name].to_numpy(float)
                    bias = conditional_bias(bias_features(saved_history, original_points),
                                            saved_history["target_price"].to_numpy(float) - original_points,
                                            bias_features(current_rows, point))
                result.loc[current_indices, "prediction_" + bias_name] = point + bias
                record["windows"][str(days)] = {"fit_days": int(history["delivery_day"].nunique()),
                    "fit_rows": len(history), "weights": weights.tolist(), "ready": bool(ready),
                    "bias_ready": bool(bias_ready),
                    "fit_first_delivery_day": str(history["delivery_day"].min()) if len(history) else None,
                    "fit_last_delivery_day": str(history["delivery_day"].max()) if len(history) else None,
                    "latest_used_target_availability_utc": history["target_available_utc"].max().isoformat() if len(history) else None}
                if ready:
                    result.loc[current_indices, "adaptive_ready"] = True
                for name in (base_name, bias_name):
                    radius = interval_radius(history["target_price"] - history["prediction_" + name], alpha)
                    result.loc[current_indices, "lower_" + name] = result.loc[current_indices, "prediction_" + name] - radius
                    result.loc[current_indices, "upper_" + name] = result.loc[current_indices, "prediction_" + name] + radius
            interval_history = recent_days(eligible, max(windows))
            radius = interval_radius(interval_history["target_price"] - interval_history["prediction_current"], alpha)
            result.loc[current_indices, "lower_current"] = current_rows["nyx_current"] - radius
            result.loc[current_indices, "upper_current"] = current_rows["nyx_current"] + radius
            audit.append(record)
    return result, audit


def score(frame: pd.DataFrame, name: str, matched_storm: bool) -> dict[str, Any]:
    finite = np.isfinite(frame["target_price"]) & np.isfinite(frame["prediction_" + name])
    if matched_storm:
        finite &= np.isfinite(frame["storm_price"])
    rows = frame[finite]
    if not len(rows):
        return {"hours": 0, "days": 0, "rmse": None, "mae": None, "bias": None,
                "storm_rmse": None, "storm_mae": None, "win_rate": None, "tie_rate": None,
                "interval_hours": 0, "interval_coverage": None, "interval_mean_width": None}
    error = rows["prediction_" + name] - rows["target_price"]
    values: dict[str, Any] = {"hours": len(rows), "days": int(rows["delivery_day"].nunique()),
        "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)), "storm_rmse": None, "storm_mae": None,
        "win_rate": None, "tie_rate": None}
    common = rows[np.isfinite(rows["storm_price"])]
    if len(common):
        model_error = np.abs(common["prediction_" + name] - common["target_price"])
        storm_error = np.abs(common["storm_price"] - common["target_price"])
        values.update(storm_rmse=float(np.sqrt(np.mean(storm_error ** 2))),
                      storm_mae=float(storm_error.mean()),
                      win_rate=float((model_error < storm_error).mean()),
                      tie_rate=float((model_error == storm_error).mean()))
    lower, upper = rows["lower_" + name], rows["upper_" + name]
    available = np.isfinite(lower) & np.isfinite(upper)
    values["interval_hours"] = int(available.sum())
    values["interval_coverage"] = float(((rows.loc[available, "target_price"] >= lower[available]) &
        (rows.loc[available, "target_price"] <= upper[available])).mean()) if available.any() else None
    values["interval_mean_width"] = float((upper[available] - lower[available]).mean()) if available.any() else None
    return values


def evaluate(frame: pd.DataFrame, windows: tuple[int, ...], holdout_days: int, warmup_days: int) -> tuple[pd.DataFrame, dict]:
    algorithms = ["current"] + [f"adaptive_{days}" for days in windows] + [f"adaptive_bias_{days}" for days in windows]
    metrics, selections = [], []
    for (country, horizon), rows in frame.groupby(["country", "horizon_day"], sort=False):
        first, last = rows["delivery_day"].min(), rows["delivery_day"].max()
        holdout_start = last - timedelta(days=holdout_days - 1)
        selection_cutoff = rows.loc[rows["delivery_day"] >= holdout_start, "forecast_origin_utc"].min()
        usable = rows[(rows["delivery_day"] >= first + timedelta(days=warmup_days)) & rows["adaptive_ready"]]
        tuning = usable[(usable["delivery_day"] < holdout_start) &
                        (usable["forecast_origin_utc"] < selection_cutoff) &
                        (usable["target_available_utc"] <= selection_cutoff)]
        require(tuning["delivery_day"].nunique() >= 14,
                f"{country}/J+{horizon}: need at least 14 tuning days after {warmup_days} warmup days before holdout")
        require(usable[usable["delivery_day"] >= holdout_start]["delivery_day"].nunique() >= 14,
                f"{country}/J+{horizon}: fewer than 14 usable holdout days; label availability may be too late")
        # Only pre-holdout tuning metrics select window and conditional-bias rule.
        matched = bool(np.isfinite(tuning["storm_price"]).any())
        base = score(tuning, "current", matched)
        tuning_scores = {name: score(tuning, name, matched) for name in algorithms}
        accepted = [name for name in algorithms if tuning_scores[name]["mae"] <= base["mae"] + 1e-10
                    and (base["win_rate"] is None or tuning_scores[name]["win_rate"] >= base["win_rate"] - 1e-10)]
        selected = min(accepted, key=lambda name: tuning_scores[name]["rmse"])
        selections.append({"country": country, "horizon_day": int(horizon), "selected": selected,
            "holdout_start": str(holdout_start), "holdout_end": str(last),
            "selection_cutoff_utc": selection_cutoff.isoformat(),
            "tuning_start": str(tuning["delivery_day"].min()), "tuning_end": str(tuning["delivery_day"].max()),
            "tuning_hours": len(tuning), "selection_uses_storm_metrics": matched,
            "selection_rule": "minimum RMSE subject to no worse MAE and strict hourly win rate versus frozen current on identical tuning hours",
            "tuning_scores": tuning_scores})
        for period in ("holdout", "365", "90", "30"):
            start = holdout_start if period == "holdout" else last - timedelta(days=int(period) - 1)
            period_rows = usable[usable["delivery_day"] >= start]
            requested_days = holdout_days if period == "holdout" else int(period)
            requested_start = pd.Timestamp(start).tz_localize(TZ).tz_convert("UTC")
            requested_stop = pd.Timestamp(last + timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
            expected_hours = int((requested_stop - requested_start).total_seconds() // 3600)
            for matched_storm in (False, True):
                for name in algorithms:
                    values = score(period_rows, name, matched_storm)
                    metrics.append({"country": country, "horizon_day": int(horizon), "period": period,
                        "algorithm": name, "selected_before_holdout": name == selected,
                        "matched_storm_only": matched_storm, "window_start": str(start), "window_end": str(last),
                        "requested_days": requested_days, "expected_hours": expected_hours,
                        "complete_requested_days": values["days"] == requested_days and values["hours"] == expected_hours,
                        **values})
    return pd.DataFrame(metrics), {"selections": selections}


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_report(output: Path, metrics: pd.DataFrame, study: dict) -> None:
    selected = metrics[(metrics["period"] == "holdout") & metrics["matched_storm_only"] &
                       ((metrics["algorithm"] == "current") | metrics["selected_before_holdout"])]
    table = selected[["country", "horizon_day", "algorithm", "hours", "rmse", "mae", "storm_rmse", "win_rate",
                      "interval_coverage", "interval_mean_width", "complete_requested_days"]].to_html(index=False, float_format=lambda x: f"{x:.4f}")
    caveats = "".join("<li>" + html.escape(item) + "</li>" for item in study["caveats"])
    scope_title = ("pilote des composants de référence" if study["scope"] == "reference_component_pilot"
                   else "comparaison chronologique des experts finaux")
    body = f"""<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX — tests chronologiques</title>
<style>body{{font:15px system-ui;margin:32px;color:#172238;background:#f8fafc}}main{{max-width:1500px;margin:auto}}table{{border-collapse:collapse;background:white;font-size:13px}}td,th{{padding:9px;border:1px solid #dde3eb;text-align:right}}th{{background:#e7edf5}}li{{margin:8px 0}}.scroll{{overflow:auto}}</style>
<main><h1>NYX — {scope_title}</h1>
<p>Le tableau présente uniquement le bloc final réservé, après choix des règles sur la période antérieure. Les scores 30/90/365 jours et tous les candidats figurent dans <code>metrics.csv</code>.</p>
<div class="scroll">{table}</div><h2>Conditions d’interprétation</h2><ul>{caveats}</ul>
<p>Statut des données : <strong>{html.escape(str(study['input_provenance_status']))}</strong>. Protocole : {PROTOCOL}.</p>
<p>Aucun modèle ni paramètre de production n’a été modifié. Les intervalles visent 80 % de couverture nominale, à vérifier sur le bloc final.</p></main></html>"""
    (output / "report.html").write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Daily OOF replay CSV")
    parser.add_argument("--manifest", type=Path, help="Provenance JSON; default INPUT.manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", default="30,60,90")
    parser.add_argument("--holdout-days", type=int, default=60)
    parser.add_argument("--warmup-days", type=int, default=60)
    parser.add_argument("--minimum-days", type=int, default=30)
    parser.add_argument("--component-columns", default=",".join(COMPONENTS))
    parser.add_argument("--control-column", default="nyx_current")
    parser.add_argument("--scope", choices=("final_experts", "reference_component_pilot"), default="final_experts")
    args = parser.parse_args(argv)
    windows = tuple(sorted(set(int(value) for value in args.windows.split(","))))
    component_columns = tuple(value.strip() for value in args.component_columns.split(","))
    require(len(windows) > 0 and min(windows) >= args.minimum_days >= 7, "Invalid windows/minimum days")
    require(args.holdout_days >= 14 and args.warmup_days >= args.minimum_days, "Invalid holdout/warmup duration")
    require(not any((args.output / name).exists() for name in ("predictions.csv", "metrics.csv", "study.json", "report.html", "fit_audit.json")),
            "Output files already exist; choose a new research directory")
    manifest_path = args.manifest or args.input.with_suffix(".manifest.json")
    require(manifest_path.is_file(), f"Provenance manifest required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict), "Provenance manifest must be a JSON object")
    require(manifest.get("scope", "final_experts") == args.scope,
            "Command scope differs from the bound input manifest")
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    declared = manifest.get("oof_csv_sha256")
    require(isinstance(declared, str) and len(declared) == 64 and declared == input_hash,
            "Manifest must bind the OOF CSV with its exact oof_csv_sha256")
    print("[NYX tests] Validate daily OOF inputs", flush=True)
    frame = validate_input(pd.read_csv(args.input), component_columns, args.control_column, args.scope)
    print(f"[NYX tests] Calibrate {len(frame):,} physical hours; separate countries and horizons", flush=True)
    predictions, audit = generate_predictions(frame, windows, args.minimum_days, component_columns=component_columns)
    metrics, selection = evaluate(predictions, windows, args.holdout_days, args.warmup_days)
    require(hashlib.sha256(args.input.read_bytes()).hexdigest() == input_hash,
            "OOF CSV changed during the research run")
    caveats = [
        "Les résultats héritent des limites PIT déclarées par le replay ; ce script ne certifie pas rétroactivement les sources historiques.",
        "Chaque poids et correction utilise uniquement des labels dont la disponibilité précède l’origine ; les résidus utilisent les prédictions historiquement émises.",
        "Le bloc final ne sert jamais au choix de fenêtre ou de règle. Les mises à jour en ligne y utilisent néanmoins les labels antérieurs devenus disponibles.",
        "Les statistiques 30/90/365 jours sont descriptives ; seul le bloc final réservé mesure la règle choisie sans réutiliser ses scores pour le choix.",
        "Une fenêtre dont complete_requested_days vaut False ne constitue pas une qualification sur la durée demandée.",
        "Storm est un comparateur et un critère de sélection pré-holdout, jamais une variable explicative ni une cible d’entraînement.",
        "Les intervalles symétriques sont calibrés sur les erreurs finales antérieures de chaque candidat et horizon ; leur couverture future n’est pas garantie.",
        "La correction conditionnelle est un test léger de niveau prévu et rampes CWE ; elle n’active aucun changement de production."]
    if args.scope == "reference_component_pilot":
        caveats.insert(0, "Ce pilote compare les composants de la référence historique au contrôle indiqué. Il n’évalue pas les trois experts finaux réentraînés et ne peut qualifier la chaîne NYX finale.")
    extra = manifest.get("caveats", [])
    if isinstance(extra, list):
        caveats.extend(str(item) for item in extra)
    study = {"protocol": PROTOCOL, "input_csv": str(args.input.resolve()), "input_sha256": input_hash,
        "calibration_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_manifest": manifest, "input_provenance_status": manifest.get("provenance_status", "not_certified"),
        "scope": args.scope, "control_column": args.control_column,
        "windows": windows, "holdout_days": args.holdout_days, "warmup_days": args.warmup_days,
        "minimum_calibration_days": args.minimum_days, "nominal_interval_coverage": .8,
        "history_window_unit": "last N complete delivery days whose targets were available at forecast origin",
        "convex_weight_penalty": 25., "conditional_bias_ridge": 20., "conditional_bias_clip_eur_mwh": 20.,
        "production_modified": False, "storm_used_as_model_input": False,
        "point_component_order": component_columns, "caveats": caveats, **selection}
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.drop(columns=[name for name in predictions if name.startswith("_")]).to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    (args.output / "study.json").write_text(json.dumps(_safe(study), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "fit_audit.json").write_text(json.dumps(_safe(audit), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(args.output, metrics, study)
    print("[NYX tests] Complete: " + str((args.output / "report.html").resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
