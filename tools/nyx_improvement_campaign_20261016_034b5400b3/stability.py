"""Held-out paired uncertainty for a completed, immutable NYX research study.

The selected rule is read from study.json; this module never selects or fits a
model. Seven-day circular blocks preserve short-range dependence. Intervals are
descriptive research evidence, never an automatic production qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

PROTOCOL = "nyx_selected_holdout_paired_block_bootstrap_v1"
MIN_DAYS, BLOCK_DAYS, REPLICATES, SEED = 14, 7, 1000, 20261016
TZ = "Europe/Paris"
OUTPUTS = ("metrics.csv", "daily_paired_losses.csv", "strata_metrics.csv", "status.json", "report.html")
SUM_COLUMNS = ("hours", "selected_se_sum", "current_se_sum", "storm_se_sum",
               "selected_ae_sum", "current_ae_sum", "storm_ae_sum", "selected_win_sum",
               "current_win_sum", "selected_tie_sum", "selected_error_sum")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(safe(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verified_source(directory):
    directory = Path(directory).resolve()
    marker = directory / "complete.json"
    require(marker.is_file(), "Completed study required: complete.json is missing")
    completion = json.loads(marker.read_text(encoding="utf-8"))
    hashes = completion.get("output_sha256", {})
    require(isinstance(hashes, dict) and {"study.json", "predictions.csv"} <= set(hashes),
            "Study completion must bind study.json and predictions.csv")
    for name, digest in hashes.items():
        require(Path(name).name == name and name not in (".", ".."), "Unsafe study output name")
        path = directory / name
        require(path.is_file() and sha(path) == digest, "Completed study output changed: " + name)
    study = json.loads((directory / "study.json").read_text(encoding="utf-8"))
    require(isinstance(study.get("selections"), list), "Study selections are missing")
    return study, {"complete_sha256": sha(marker), "output_sha256": hashes}


def family(name):
    if name == "current":
        return "current_control"
    if name.startswith("component_"):
        return "individual_component"
    if name.startswith("convex_") and ("level" in name or "ridge" in name):
        return "ensemble_and_correction"
    if name.startswith("convex_"):
        return "continuous_ensemble"
    if name.startswith("level_"):
        return "level_correction"
    if name.startswith("ridge_"):
        return "ridge_correction"
    return "online_ensemble" if name == "online_blend" else "other_preselected_rule"


def expected_grid(day):
    start = pd.Timestamp(day).tz_localize(TZ)
    end = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TZ)
    return pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")


def sums(rows, name):
    point = rows["prediction_" + name].to_numpy(float)
    target = rows.target_price.to_numpy(float)
    current = rows.prediction_current.to_numpy(float)
    storm = rows.storm_price.to_numpy(float)
    e, c, s = point - target, current - target, storm - target
    return dict(zip(SUM_COLUMNS, (len(rows), np.sum(e*e), np.sum(c*c), np.sum(s*s),
        np.sum(abs(e)), np.sum(abs(c)), np.sum(abs(s)), np.sum(abs(e) < abs(s)),
        np.sum(abs(c) < abs(s)), np.sum(abs(e) == abs(s)), np.sum(e))))


def measures(total):
    n = total["hours"]
    if not n:
        return {name: None for name in ("rmse", "mae", "bias", "current_rmse", "current_mae",
            "storm_rmse", "storm_mae", "rmse_diff_storm", "rmse_diff_current", "mae_diff_storm",
            "mae_diff_current", "strict_win_rate", "current_strict_win_rate", "strict_win_rate_diff_current", "tie_rate")}
    rmse = np.sqrt(total["selected_se_sum"] / n)
    current_rmse, storm_rmse = np.sqrt(total["current_se_sum"] / n), np.sqrt(total["storm_se_sum"] / n)
    mae, current_mae, storm_mae = (total[key] / n for key in ("selected_ae_sum", "current_ae_sum", "storm_ae_sum"))
    win, current_win = total["selected_win_sum"] / n, total["current_win_sum"] / n
    return {"rmse": rmse, "mae": mae, "bias": total["selected_error_sum"] / n,
        "current_rmse": current_rmse, "current_mae": current_mae, "storm_rmse": storm_rmse, "storm_mae": storm_mae,
        "rmse_diff_storm": rmse-storm_rmse, "rmse_diff_current": rmse-current_rmse,
        "mae_diff_storm": mae-storm_mae, "mae_diff_current": mae-current_mae,
        "strict_win_rate": win, "current_strict_win_rate": current_win,
        "strict_win_rate_diff_current": win-current_win, "tie_rate": total["selected_tie_sum"] / n}


def bootstrap(daily, key):
    require(len(daily) >= MIN_DAYS, "Insufficient days for block intervals")
    dates = pd.to_datetime(daily.delivery_day)
    require((dates.diff().dropna() == pd.Timedelta(days=1)).all(), "Block intervals need consecutive complete days")
    digest = hashlib.sha256((str(key[0]) + ":" + str(key[1])).encode()).hexdigest()
    random = np.random.default_rng(SEED + int(digest[:8], 16))
    values = daily[list(SUM_COLUMNS)].to_numpy(float)
    names = ("rmse_diff_storm", "rmse_diff_current", "mae_diff_storm", "mae_diff_current",
             "strict_win_rate", "strict_win_rate_diff_current")
    samples = {name: [] for name in names}
    size = len(values)
    for _ in range(REPLICATES):
        starts = random.integers(size, size=int(np.ceil(size / BLOCK_DAYS)))
        positions = ((starts[:, None] + np.arange(BLOCK_DAYS)) % size).ravel()[:size]
        total = dict(zip(SUM_COLUMNS, values[positions].sum(axis=0)))
        scores = measures(total)
        for name in names:
            samples[name].append(scores[name])
    return {name + suffix: float(np.quantile(samples[name], quantile))
        for name in names for suffix, quantile in (("_ci_low", .025), ("_ci_high", .975))}


def analyse(study, predictions):
    required = {"country", "horizon_day", "delivery_utc", "forecast_origin_utc", "target_price",
                "storm_price", "prediction_current"}
    require(required <= set(predictions), "Missing prediction columns: " + str(sorted(required-set(predictions))))
    frame = predictions.copy()
    frame["country"] = frame.country.astype(str)
    frame["horizon_day"] = pd.to_numeric(frame.horizon_day, errors="raise").astype(int)
    require(frame.horizon_day.isin([1, 2]).all(), "Only separate J+1 and J+2 groups are supported")
    frame["delivery_utc"] = pd.to_datetime(frame.delivery_utc, utc=True, errors="raise")
    frame["forecast_origin_utc"] = pd.to_datetime(frame.forecast_origin_utc, utc=True, errors="raise")
    frame["delivery_day"] = frame.delivery_utc.dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    require(not frame.duplicated(["country", "horizon_day", "delivery_utc"]).any(), "Duplicate physical delivery hours")
    summaries, daily_rows, strata_rows, seen = [], [], [], set()
    for selection in study["selections"]:
        key = (str(selection["country"]), int(selection["horizon_day"]))
        require(key not in seen, "Duplicate country/horizon study selection")
        seen.add(key)
        selected = selection.get("selected", "current")
        require(isinstance(selected, str) and "prediction_"+selected in frame, "Selected forecast column is missing")
        identity = {"country": key[0], "horizon_day": key[1], "selected": selected,
                    "selection_family": family(selected), "selection_status": selection.get("selection_status"),
                    "production_qualification": False, "production_modified": False}
        groups = frame[(frame.country == key[0]) & (frame.horizon_day == key[1])]
        reasons = []
        start, end = selection.get("holdout_start"), selection.get("holdout_end")
        if not selection.get("ready") or not start or not end:
            summaries.append({**identity, "status": "NOT_READY", "reason": "No frozen ready holdout in source study",
                              "complete_paired_days": 0, "hours": 0})
            continue
        begin, finish = pd.Timestamp(start), pd.Timestamp(end)
        require(begin.tz is None and finish.tz is None and begin <= finish, "Invalid holdout boundaries")
        requested_days = int(selection.get("holdout_days", (finish-begin).days+1))
        require((finish-begin).days+1 == requested_days, "Holdout calendar length differs from frozen selection")
        cutoff = pd.Timestamp(selection["selection_cutoff_utc"])
        require(cutoff.tz is not None, "Selection cutoff must be timezone-aware")
        tuning_days = selection.get("tuning_days_list", [])
        require(all(pd.Timestamp(day) < begin for day in tuning_days), "Tuning overlaps frozen holdout")
        tuning = groups[groups.delivery_day.isin(tuning_days)]
        require((tuning.forecast_origin_utc < cutoff).all(), "Tuning origin is not before selection cutoff")
        if "target_available_utc" in tuning and len(tuning):
            require((pd.to_datetime(tuning.target_available_utc, utc=True) <= cutoff).all(), "Tuning label unavailable at selection cutoff")
        heldout = groups[(groups.delivery_day >= str(begin.date())) & (groups.delivery_day <= str(finish.date()))].copy()
        require(not len(heldout) or (heldout.forecast_origin_utc >= cutoff).all(), "Holdout forecast precedes frozen selection")
        if not selection.get("benchmark_selection_ready", False):
            reasons.append("Source study did not have sufficient pre-holdout Storm comparison for selection")
        columns = list(dict.fromkeys(("target_price", "storm_price", "prediction_current", "prediction_"+selected)))
        for column in columns:
            heldout[column] = pd.to_numeric(heldout[column], errors="coerce")
        complete = []
        for day in pd.date_range(begin, finish, freq="D").strftime("%Y-%m-%d"):
            part = heldout[heldout.delivery_day == day].sort_values("delivery_utc")
            if (part.forecast_origin_utc.nunique() == 1 and
                pd.DatetimeIndex(part.delivery_utc).equals(expected_grid(day)) and
                np.isfinite(part[columns].to_numpy(float)).all()):
                complete.append(part)
                total = sums(part, selected)
                daily_rows.append({**identity, "delivery_day": day, **total, **measures(total)})
        paired = pd.concat(complete, ignore_index=True) if complete else heldout.iloc[:0]
        count = len(complete)
        if count != requested_days:
            reasons.append("Incomplete common hourly coverage of requested holdout: missing Storm, labels or forecasts, physical hours, or days")
        if count < MIN_DAYS:
            reasons.append(f"Need at least {MIN_DAYS} complete paired days")
        total = sums(paired, selected)
        summary = {**identity, "status": "NOT_READY" if reasons else "HELDOUT_STABILITY_COMPLETE",
            "reason": "; ".join(reasons) or None, "holdout_start": str(begin.date()), "holdout_end": str(finish.date()),
            "requested_holdout_days": requested_days, "complete_paired_days": count,
            "excluded_holdout_hours": len(heldout)-len(paired), "hours": len(paired), **measures(total)}
        if not reasons:
            daily = pd.DataFrame([row for row in daily_rows if (row["country"], row["horizon_day"]) == key])
            summary.update(bootstrap(daily, key))
        summaries.append(summary)
        if len(paired):
            paired["local_hour"] = paired.delivery_utc.dt.tz_convert(TZ).dt.hour
            paired["local_month"] = paired.delivery_utc.dt.tz_convert(TZ).dt.strftime("%Y-%m")
            threshold = float(paired.target_price.quantile(.99))
            paired["target_tail"] = np.where(paired.target_price >= threshold, "target_top_1pct", "other_target_prices")
            for dimension, column in (("local_hour", "local_hour"), ("local_month", "local_month"), ("target_tail", "target_tail")):
                for bucket, part in paired.groupby(column, sort=True):
                    strata_rows.append({**identity, "stratum": dimension, "bucket": str(bucket),
                        "hours": len(part), "days": part.delivery_day.nunique(), "descriptive_only": True,
                        "target_top_1pct_threshold": threshold if dimension == "target_tail" else None,
                        **measures(sums(part, selected))})
    require(seen == set(zip(frame.country, frame.horizon_day)), "Prediction groups differ from frozen study selections")
    return summaries, daily_rows, strata_rows


def report(output, summaries, status):
    frame = pd.DataFrame(summaries)
    columns = [c for c in ("country", "horizon_day", "selected", "selection_family", "status", "complete_paired_days", "hours",
        "rmse", "mae", "rmse_diff_storm", "rmse_diff_storm_ci_low", "rmse_diff_storm_ci_high",
        "mae_diff_storm", "mae_diff_storm_ci_low", "mae_diff_storm_ci_high", "strict_win_rate",
        "strict_win_rate_ci_low", "strict_win_rate_ci_high") if c in frame]
    table = frame[columns].to_html(index=False, float_format=lambda x: f"{x:.5f}")
    explanations = "".join("<li>"+html.escape(f"{row['country']} J+{row['horizon_day']} : {row.get('reason') or 'bloc réservé complet'}")+"</li>" for row in summaries)
    (output/"report.html").write_text(f'''<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX — stabilité hors échantillon</title>
<style>body{{font:15px system-ui;color:#172238;background:#f6f8fb;margin:30px}}main{{max-width:1600px;margin:auto}}table{{border-collapse:collapse;background:white}}th,td{{border:1px solid #dce2eb;padding:8px;text-align:right}}.scroll{{overflow:auto}}</style>
<main><h1>NYX — stabilité sur le bloc réservé</h1><p><strong>{html.escape(status['status'])}</strong> — recherche ; qualification de production : NON.</p>
<p>La règle a été reprise de la sélection figée avant le bloc réservé ; aucune nouvelle sélection n'utilise ces résultats. Pays et horizons ne sont jamais regroupés.</p>
<div class="scroll">{table}</div><ul>{explanations}</ul>
<p>Écarts = erreur du candidat moins erreur du comparateur ; une valeur négative est favorable. Win rate strict : erreur absolue NYX inférieure à Storm ; les égalités ne gagnent pas.</p>
<p>Intervalles percentile à 95 %, {REPLICATES} répétitions de blocs circulaires de {BLOCK_DAYS} jours, graine figée. Rééchantillonnage apparié des mêmes jours et des mêmes heures physiques pour NYX, contrôle et Storm. Aucun intervalle si le bloc réservé n'est pas intégralement couvert ou compte moins de {MIN_DAYS} jours.</p>
<p>Les tableaux par heure, mois et prix réalisés les plus élevés (1 %) sont descriptifs uniquement. Le seuil de prix réalisé est calculé pour le scoring, jamais utilisé pour sélectionner ou entraîner le modèle. Aucun intervalle par sous-groupe n'est présenté.</p>
<p>Une série courte reste une preuve limitée ; ces intervalles ne corrigent ni l'incertitude des vintages, ni les comparaisons répétées, ni un changement de régime. Les limites PIT et le périmètre du rapport d'origine restent applicables. Les résultats d'un pilote de référence ne qualifient pas les experts finaux.</p>
<p>Détails : metrics.csv, daily_paired_losses.csv, strata_metrics.csv et status.json. Aucune activation automatique.</p></main></html>''', encoding="utf-8")


def run(study_dir, output):
    study_dir, output = Path(study_dir).resolve(), Path(output).resolve()
    require(output != study_dir, "Stability output must be separate from the source study")
    study, source = verified_source(study_dir)
    plan = {"protocol": PROTOCOL, "source_directory": str(study_dir), "source": source,
        "code_sha256": sha(Path(__file__)), "runtime": {"python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()},
        "block_days": BLOCK_DAYS, "replicates": REPLICATES, "seed": SEED, "minimum_complete_days": MIN_DAYS}
    output.mkdir(parents=True, exist_ok=True)
    plan_file, marker = output/"run_plan.json", output/"complete.json"
    if plan_file.exists():
        require(json.loads(plan_file.read_text(encoding="utf-8")) == plan,
                "Existing stability run is pinned to different inputs, code or runtime; choose a new output directory")
        if marker.exists():
            binding = json.loads(marker.read_text(encoding="utf-8")).get("output_sha256", {})
            require(set(binding) == set(OUTPUTS) and all((output/name).is_file() and sha(output/name) == digest for name,digest in binding.items()),
                    "Completed stability output changed; use a new output directory")
            print("[NYX stability] Reuse verified report: " + str(output/"report.html"), flush=True)
            return output/"report.html"
    else:
        require(not marker.exists() and not any((output/name).exists() for name in OUTPUTS), "Unpinned existing stability output")
        atomic_json(plan_file, plan)
    predictions = pd.read_csv(study_dir/"predictions.csv")
    summaries, daily, strata = analyse(study, predictions)
    completed = sum(row["status"] == "HELDOUT_STABILITY_COMPLETE" for row in summaries)
    status = {"protocol": PROTOCOL, "status": "HELDOUT_STABILITY_COMPLETE" if completed and completed == len(summaries)
        else "PARTIAL_STABILITY_NOT_READY" if completed else "NOT_READY", "production_qualification": False,
        "production_modified": False, "retuned_using_holdout": False, "source_scope": study.get("scope"),
        "source_provenance_status": study.get("input_provenance_status", "not_certified"),
        "country_horizon_groups": len(summaries), "completed_groups": completed,
        "bootstrap": {"method": "paired_circular_day_blocks", "block_days": BLOCK_DAYS,
            "replicates": REPLICATES, "confidence_level": .95, "seed": SEED}, "groups": summaries}
    pd.DataFrame(summaries).to_csv(output/"metrics.csv", index=False)
    pd.DataFrame(daily, columns=list(daily[0]) if daily else ["country","horizon_day","delivery_day",*SUM_COLUMNS]).to_csv(output/"daily_paired_losses.csv", index=False)
    pd.DataFrame(strata, columns=list(strata[0]) if strata else ["country","horizon_day","stratum","bucket","hours"]).to_csv(output/"strata_metrics.csv", index=False)
    atomic_json(output/"status.json", status)
    report(output, summaries, status)
    _, after = verified_source(study_dir)
    require(after == source, "Study changed during stability evaluation")
    atomic_json(marker, {"output_sha256": {name: sha(output/name) for name in OUTPUTS}})
    print("[NYX stability] " + status["status"] + ": " + str(output/"report.html"), flush=True)
    return output/"report.html"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run(args.study_dir, args.output)
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        print("[NYX stability] ERROR: " + str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
