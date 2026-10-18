"""Isolated, causal NYX improvement campaign; never changes production models.

Input is a CSV bound by INPUT.manifest.json (oof_csv_sha256). All final inputs
must reproduce calibration.frozen_current exactly. This is research, never an
automatic release. Python 3.11; numpy, pandas, scipy and sklearn required.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import html
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import sklearn

import calibration

PROTOCOL = "nyx_improvement_chronological_research_v1"
CAPS = {"FR": 18.5965, "BE": 21.2027, "NL": 20.5667}
REFERENCE = ("reference_price", "nyx_baseline", "test2_price")
BASE_ALGORITHMS = ("current", "convex_mse", "convex_mae", "convex_huber",
                   "level_14", "level_60", "ridge_current", "convex_level_14",
                   "convex_ridge", "online_blend")
OUTPUTS = ("predictions.csv", "metrics.csv", "study.json", "status.json", "report.html")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def convex(x: np.ndarray, y: np.ndarray, loss: str) -> np.ndarray:
    """Strongly regularized simplex fit; forecast spread centered for stability."""
    prior = np.full(x.shape[1], 1. / x.shape[1])
    centre = x.mean(axis=1)
    xx, yy = x - centre[:, None], y - centre
    # Scale the entire objective, including its regularizer. This preserves the
    # optimum and avoids large spikes confusing SLSQP's line search/constraints.
    objective_scale=max(1.,float(np.mean(xx**2)),float(np.mean(yy**2)))
    def objective(w):
        error = xx @ w - yy
        if loss == "mse":
            value, derivative = .5 * error ** 2, error
        elif loss == "mae":
            # Smooth absolute error with a 1 EUR/MWh smoothing scale.
            norm = np.sqrt(error ** 2 + 1.)
            value, derivative = 20. * norm, 20. * error / norm
        else:
            absolute = np.abs(error)
            value = np.where(absolute <= 20., .5 * error ** 2, 20. * (absolute - 10.))
            derivative = np.clip(error, -20., 20.)
        regularizer = 25. * np.sum((w - prior) ** 2) / 2.
        return (float(value.mean()+regularizer)/objective_scale,
                (xx.T@derivative/len(y)+25.*(w-prior))/objective_scale)
    solved = minimize(objective, prior, method="SLSQP", jac=True,
        bounds=[(0., 1.)] * len(prior), constraints={"type": "eq", "fun": lambda w: w.sum() - 1.,
        "jac": lambda w: np.ones_like(w)}, options={"maxiter": 500, "ftol": 1e-12})
    if not solved.success:
        # Deterministic second initialization at the best simplex vertex. A
        # remaining failure is explicit; the prior is never silently called fit.
        vertices=np.eye(x.shape[1])
        start=min(vertices,key=lambda w:objective(w)[0])
        solved=minimize(objective,start,method="SLSQP",jac=True,
            bounds=[(0.,1.)]*len(prior),constraints={"type":"eq","fun":lambda w:w.sum()-1.,
            "jac":lambda w:np.ones_like(w)},options={"maxiter":1000,"ftol":1e-12})
    calibration.require(solved.success and np.isfinite(solved.x).all(),
                        "Convex optimization failed: " + str(solved.message))
    result = np.maximum(solved.x, 0.)
    return result / result.sum()


def features(frame: pd.DataFrame, point: np.ndarray, components: tuple[str, ...]) -> np.ndarray:
    hour = frame.delivery_hour.to_numpy(float)
    day = pd.to_datetime(frame.delivery_day).dt.dayofweek.to_numpy(float)
    values = frame[list(components)].to_numpy(float)
    return np.column_stack((np.ones(len(frame)), np.sin(2*np.pi*hour/24),
        np.cos(2*np.pi*hour/24), np.sin(2*np.pi*day/7), np.cos(2*np.pi*day/7),
        point, frame.reference_price.to_numpy(float), values.max(axis=1)-values.min(axis=1),
        frame._own_ramp.to_numpy(float), frame._cwe_point.to_numpy(float)))


def ridge_correct(history, future, point_history, point_future, components):
    x = features(history, point_history, components)
    z = features(future, point_future, components)
    centre, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.
    centre[0], scale[0] = 0., 1.
    x, z = (x-centre)/scale, (z-centre)/scale
    penalty = np.eye(x.shape[1]) * 80.
    penalty[0, 0] = 24. * 14.  # Shrink the level intercept toward zero.
    beta = np.linalg.solve(x.T@x+penalty, x.T@(history.target_price.to_numpy(float)-point_history))
    return np.clip(z@beta, -40., 40.)


def eligible_history(rows: pd.DataFrame, origin, day) -> pd.DataFrame:
    prior = rows[(rows.forecast_origin_utc < origin) & (rows.delivery_day < day)
                 & (rows.target_available_utc <= origin)]
    expected = rows.groupby("delivery_day").size()
    counts = prior.groupby("delivery_day").size()
    complete = [d for d, n in counts.items() if n == expected[d]]
    return prior[prior.delivery_day.isin(complete)]


def recent(history, days=60):
    dates = sorted(history.delivery_day.unique())[-days:]
    return history[history.delivery_day.isin(dates)]


def level(history, name, days):
    history = recent(history, days)
    history = history[np.isfinite(history["prediction_"+name])]
    daily = (history.target_price-history["prediction_"+name]).groupby(history.delivery_day).mean()
    if len(daily) < 7:
        return 0.
    return float(np.clip(daily.mean() * len(daily)/(len(daily)+14.), -40., 40.))


def generate(frame: pd.DataFrame, components: tuple[str, ...], train_groups: set[tuple]) -> tuple[pd.DataFrame, list]:
    """Storm is deliberately removed before any model/feature computation."""
    result = frame.drop(columns=["storm_price"], errors="ignore").copy()
    algorithms = BASE_ALGORITHMS + tuple("component_"+c for c in components)
    for name in algorithms:
        result["prediction_"+name] = np.nan
    result.prediction_current = result.nyx_current
    for name in components:
        result["prediction_component_"+name] = result[name]
    audits = []
    for key, group in result.groupby(["country", "horizon_day"], sort=True):
        if key not in train_groups:
            print(f"[NYX campaign] {key[0]} J+{key[1]}: descriptive only, insufficient history", flush=True)
            continue
        indices = group.index
        for count, (origin, current) in enumerate(group.groupby("forecast_origin_utc", sort=True), 1):
            history = eligible_history(result.loc[indices], origin, current.delivery_day.iloc[0])
            history = recent(history)
            if history.delivery_day.nunique() < 30:
                continue
            pos = current.index
            current = result.loc[pos]
            x, y = history[list(components)].to_numpy(float), history.target_price.to_numpy(float)
            weights = {}
            for loss in ("mse", "mae", "huber"):
                weights[loss] = convex(x, y, loss)
                result.loc[pos, "prediction_convex_"+loss] = current[list(components)].to_numpy(float)@weights[loss]
            now = result.loc[pos]
            base = now.prediction_current.to_numpy(float)
            blended = now.prediction_convex_mse.to_numpy(float)
            result.loc[pos, "prediction_level_14"] = base + level(history, "current", 14)
            result.loc[pos, "prediction_level_60"] = base + level(history, "current", 60)
            result.loc[pos, "prediction_ridge_current"] = base + ridge_correct(history, now,
                history.prediction_current.to_numpy(float), base, components)
            valid = history[np.isfinite(history.prediction_convex_mse)]
            if valid.delivery_day.nunique() >= 14:
                result.loc[pos, "prediction_convex_level_14"] = blended+level(valid, "convex_mse", 14)
                result.loc[pos, "prediction_convex_ridge"] = blended+ridge_correct(valid, now,
                    valid.prediction_convex_mse.to_numpy(float), blended, components)
            else:
                result.loc[pos, "prediction_convex_level_14"] = blended
                result.loc[pos, "prediction_convex_ridge"] = blended
            online_names = ("current", "convex_mse", "convex_mae", "convex_huber", "level_14", "level_60", "ridge_current")
            columns = ["prediction_"+n for n in online_names]
            prior = recent(history, 30)
            prior = prior[np.isfinite(prior[columns]).all(axis=1)]
            losses = (np.abs(prior[columns].to_numpy(float)-prior.target_price.to_numpy(float)[:,None]).mean(axis=0)
                      if len(prior) else np.zeros(len(columns)))
            online = np.exp(-(losses-losses.min())/10.)
            online /= online.sum()
            result.loc[pos, "prediction_online_blend"] = result.loc[pos,columns].to_numpy(float)@online
            audits.append({"country": key[0], "horizon_day": int(key[1]), "origin": origin.isoformat(),
                "fit_days": int(history.delivery_day.nunique()), "latest_used_label_available_utc":
                history.target_available_utc.max().isoformat(), "convex_weights": {k:w.tolist() for k,w in weights.items()},
                "online_weights": online.tolist()})
            if count % 20 == 0:
                print(f"[NYX campaign] {key[0]} J+{key[1]}: {count}/{group.forecast_origin_utc.nunique()} origins", flush=True)
    result["storm_price"] = frame.storm_price
    return result, audits


def score(rows, name):
    point = rows["prediction_"+name].to_numpy(float)
    target = rows.target_price.to_numpy(float)
    valid = np.isfinite(point) & np.isfinite(target)
    rows, point, target = rows.loc[valid], point[valid], target[valid]
    if not len(rows):
        return {"hours": 0, "days": 0, "rmse": None, "mae": None, "bias": None,
                "storm_rmse": None, "storm_mae": None, "win_rate": None, "tie_rate": None}
    error = point-target
    result = {"hours":len(rows), "days":int(rows.delivery_day.nunique()),
        "rmse":float(np.sqrt(np.mean(error**2))), "mae":float(np.mean(abs(error))), "bias":float(error.mean()),
        "storm_rmse":None, "storm_mae":None, "win_rate":None, "tie_rate":None}
    storm = rows.storm_price.to_numpy(float)
    available = np.isfinite(storm)
    if available.any():
        se = storm[available]-target[available]
        ae = abs(error[available])
        result.update(storm_rmse=float(np.sqrt(np.mean(se**2))), storm_mae=float(abs(se).mean()),
            win_rate=float((ae<abs(se)).mean()), tie_rate=float((ae==abs(se)).mean()))
    return result


def gates(candidate, current, country, horizon):
    def less(a,b):
        return a is not None and b is not None and a < b
    cap = CAPS.get(country) if horizon == 1 else None
    result = {"rmse_below_current":less(candidate["rmse"],current["rmse"]),
        "mae_below_current":less(candidate["mae"],current["mae"]),
        "rmse_below_storm":less(candidate["rmse"],candidate["storm_rmse"]),
        "mae_below_storm":less(candidate["mae"],candidate["storm_mae"]),
        "strict_win_above_half_and_current":candidate["win_rate"] is not None and current["win_rate"] is not None
             and candidate["win_rate"] > max(.5,current["win_rate"]),
        "historical_j1_rmse_cap":cap, "historical_cap_passed":less(candidate["rmse"],cap) if cap is not None else None}
    result["all_gates_passed"] = bool(all(result[k] for k in ("rmse_below_current","mae_below_current",
        "rmse_below_storm","mae_below_storm","strict_win_above_half_and_current"))
        and (cap is None or result["historical_cap_passed"]))
    return result


def split_plan(rows, warmup=45, holdout=30):
    days = sorted(rows.delivery_day.unique())
    required=max(90,warmup+14+holdout)
    if len(days)<required:
        return {"ready":False, "reason":"Insufficient complete days for warmup, tuning and untouched requested holdout", 
                "available_days":len(days),"minimum_requested_days":required,"holdout_days":holdout,
                "warmup_days":warmup,"missing_days_before_availability_embargo":required-len(days)}
    start = days[-holdout]
    if (days[-1]-start).days != holdout-1:
        requested_start=days[-1]-timedelta(days=holdout-1)
        covered=[d for d in days if d>=requested_start]
        return {"ready":False,"reason":"Requested holdout has missing calendar days; no partial period is relabelled complete",
            "available_days":len(days),"holdout_days":holdout,"warmup_days":warmup,
            "holdout_start":str(requested_start),"holdout_end":str(days[-1]),
            "missing_holdout_calendar_days":holdout-len(covered)}
    cutoff = rows.loc[rows.delivery_day>=start,"forecast_origin_utc"].min()
    tuning = rows[(rows.delivery_day>=days[warmup]) & (rows.delivery_day<start)
                  & (rows.forecast_origin_utc<cutoff) & (rows.target_available_utc<=cutoff)]
    counts, expected = tuning.groupby("delivery_day").size(), rows.groupby("delivery_day").size()
    complete = [d for d,n in counts.items() if n==expected[d]]
    tuning=tuning[tuning.delivery_day.isin(complete)]
    return {"ready":len(complete)>=14, "available_days":len(days), "warmup_days":warmup,
        "holdout_days":holdout, "holdout_start":str(start), "holdout_end":str(days[-1]),
        "selection_cutoff_utc":cutoff.isoformat(), "tuning_days":len(complete),
        "tuning_days_list":[str(d) for d in complete],
        "reason":None if len(complete)>=14 else "Need 14 complete tuning days after warmup and availability embargo"}


def evaluate(predictions, plans):
    algorithms=[c.removeprefix("prediction_") for c in predictions if c.startswith("prediction_")]
    metrics, selections=[] ,[]
    for key,rows in predictions.groupby(["country","horizon_day"],sort=True):
        plan=plans[key]
        available_names=[n for n in algorithms if np.isfinite(rows["prediction_"+n]).any()]
        if not plan["ready"]:
            periods={"available_descriptive":rows}
            selected="current"
            selection={"country":key[0],"horizon_day":int(key[1]),"selected":"current",
                       "selection_status":"NOT_READY",**plan}
        else:
            tuning=rows[rows.delivery_day.astype(str).isin(plan["tuning_days_list"])]
            # One common physical-hour sample for every candidate and comparator.
            tuning=tuning[np.isfinite(tuning.storm_price)&np.isfinite(tuning[
                ["prediction_"+n for n in available_names]]).all(axis=1)]
            control=score(tuning,"current")
            tuned={name:score(tuning,name) for name in available_names}
            facts={name:gates(values,control,key[0],key[1]) for name,values in tuned.items()}
            enough_benchmark = tuning.delivery_day.nunique() >= 14 and len(tuning) >= 300
            accepted=[n for n in available_names if enough_benchmark and facts[n]["all_gates_passed"]]
            selected=min(accepted,key=lambda n:(tuned[n]["mae"],tuned[n]["rmse"],n)) if accepted else "current"
            selection={"country":key[0],"horizon_day":int(key[1]),"selected":selected,
                "selection_status":"RESEARCH_CANDIDATE_SELECTED" if accepted else "NO_CANDIDATE_PASSED_TUNING_GATES",
                **plan,"matched_tuning_hours":len(tuning),"matched_tuning_days":int(tuning.delivery_day.nunique()),
                "benchmark_selection_ready":bool(enough_benchmark),"tuning_scores":tuned,"tuning_gates":facts,
                "selection_uses_storm_metrics":True,"storm_used_as_model_input":False}
            holdout=rows[rows.delivery_day.astype(str)>=plan["holdout_start"]]
            periods={f"holdout_{plan['holdout_days']}":holdout,"all_post_warmup_descriptive":rows[rows.delivery_day>=sorted(rows.delivery_day.unique())[45]]}
        selections.append(selection)
        for period,part in periods.items():
            # Matched rows fixed before score; missing Storm may not change candidate sample.
            for matched in (False,True):
                common=part[np.isfinite(part[["prediction_"+n for n in available_names]]).all(axis=1)]
                if matched: common=common[np.isfinite(common.storm_price)]
                base=score(common,"current")
                for name in available_names:
                    values=score(common,name)
                    gate=gates(values,base,key[0],key[1]) if matched else {}
                    metrics.append({"country":key[0],"horizon_day":int(key[1]),"period":period,
                        "algorithm":name,"selected_before_holdout":name==selected,"matched_storm_only":matched,
                        "first_observed_day":str(common.delivery_day.min()) if len(common) else None,
                        "last_observed_day":str(common.delivery_day.max()) if len(common) else None,
                        "requested_holdout_days":plan["holdout_days"] if period.startswith("holdout_") else None,
                        "complete_requested_holdout":bool(period.startswith("holdout_") and common.delivery_day.nunique()==plan["holdout_days"]
                            and len(common)==len(part)),**values,**gate})
    return pd.DataFrame(metrics),selections


def report(output,metrics,study,status):
    preferred=metrics[(metrics.matched_storm_only)&((metrics.algorithm=="current")|metrics.selected_before_holdout)]
    columns=[c for c in ("country","horizon_day","period","algorithm","hours","days","rmse","mae",
        "storm_rmse","storm_mae","win_rate","all_gates_passed") if c in preferred]
    title="NYX — programme de tests des modèles"
    caveats="".join("<li>"+html.escape(s)+"</li>" for s in study["caveats"])
    table=preferred[columns].to_html(index=False,float_format=lambda x:f"{x:.4f}")
    selection="".join("<li>"+html.escape(f"{s['country']} J+{s['horizon_day']} : {s['selection_status']} ; "
        f"{s['available_days']} jours complets disponibles ; règle retenue : {s['selected']}.")+"</li>" for s in study["selections"])
    (output/"report.html").write_text(f'''<!doctype html><html lang="fr"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:15px system-ui;background:#f6f8fb;color:#172238;margin:30px}}main{{max-width:1500px;margin:auto}}table{{border-collapse:collapse;background:white;font-size:13px}}th,td{{padding:8px;border:1px solid #dce2eb;text-align:right}}.scroll{{overflow:auto}}li{{margin:8px 0}}</style>
<main><h1>{title}</h1><p><strong>{html.escape(status['status'])}</strong> — recherche isolée ; aucune activation de production.</p>
<p>Périmètre : {html.escape(study['scope'])}. Les résultats du pilote de référence ne représentent pas les experts finaux.</p>
<ul>{selection}</ul><div class="scroll">{table}</div><h2>Lecture du test</h2><ul>{caveats}</ul>
<p>Les seuils J+1 historiques FR/BE/NL sont des ambitions liées à l’ancien protocole 08 h. Aucun seuil absolu Allemagne ou J+2 n’est inventé. Les fenêtres réellement observées figurent dans metrics.csv ; un historique partiel n’est jamais une validation annuelle.</p>
<p>Qualification de production : <strong>NON</strong>. Toutes les variantes sont dans metrics.csv ; prédictions causales dans predictions.csv ; sélection et traçabilité dans study.json.</p></main></html>''',encoding="utf-8")


def run(input_csv: Path, output: Path, manifest_path: Path|None=None, extras=(), holdout_days=30):
    input_csv,output=input_csv.resolve(),output.resolve()
    manifest_path=(manifest_path or input_csv.with_suffix(".manifest.json")).resolve()
    calibration.require(manifest_path.is_file(),"Bound manifest required: "+str(manifest_path))
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration.require(isinstance(manifest,dict),"Bound manifest must be a JSON object")
    input_hash=sha(input_csv)
    calibration.require(manifest.get("oof_csv_sha256")==input_hash,"Input CSV hash differs from manifest")
    scope=manifest.get("scope","final_experts")
    calibration.require(scope in ("final_experts","reference_component_pilot"),"Unknown manifest scope")
    calibration.require(holdout_days>=30,"At least 30 holdout days required")
    calibration.require(scope=="final_experts" or holdout_days==30,"Reference pilot uses 30-day holdout; annual assessment requires final-expert scope")
    raw=pd.read_csv(input_csv)
    components=calibration.COMPONENTS if scope=="final_experts" else REFERENCE
    control="nyx_current" if scope=="final_experts" else "nyx_baseline"
    frame=calibration.validate_input(raw,components,control,scope)
    extras=tuple(extras)
    calibration.require(len(components)+len(extras)<=9 and len(set(extras))==len(extras),"At most nine distinct components total")
    for column in extras:
        calibration.require(column in frame and column not in components and column not in calibration.REQUIRED
            and not any(word in column.lower() for word in ("storm","target","actual","available","origin","delivery","prediction"))
            and column not in ("country","horizon_day","nyx_current"),"Invalid challenger forecast column: "+column)
        frame[column]=pd.to_numeric(frame[column],errors="raise")
        calibration.require(np.isfinite(frame[column]).all(),"Challenger component must be finite: "+column)
    components=(*components,*extras)
    runtime={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
             "scipy":scipy.__version__,"sklearn":sklearn.__version__,"platform":platform.platform()}
    plan={"protocol":PROTOCOL,"input_sha256":input_hash,"manifest_sha256":sha(manifest_path),
        "code_sha256":sha(Path(__file__)),"calibration_code_sha256":sha(Path(calibration.__file__)),
        "runtime":runtime,"components":list(components),"scope":scope,"warmup_days":45,"holdout_days":holdout_days}
    output.mkdir(parents=True,exist_ok=True)
    plan_file=output/"run_plan.json"
    if plan_file.exists():
        calibration.require(json.loads(plan_file.read_text(encoding="utf-8"))==plan,
            "Existing campaign is pinned to different inputs, code or runtime; choose a new output directory")
        complete=output/"complete.json"
        if complete.exists():
            bound=json.loads(complete.read_text(encoding="utf-8"))
            calibration.require(all((output/n).is_file() and sha(output/n)==digest for n,digest in bound["output_sha256"].items()),
                "Completed campaign output changed; use a new output directory")
            print("[NYX campaign] Reuse verified completed study: "+str(output/"report.html"),flush=True)
            return output/"report.html"
    else:
        calibration.require(not any((output/n).exists() for n in OUTPUTS),"Unpinned existing output directory")
        atomic_json(plan_file,plan)
    plans={key:split_plan(rows,holdout=holdout_days) for key,rows in frame.groupby(["country","horizon_day"],sort=True)}
    predictions,audit=generate(frame,components,{key for key,p in plans.items() if p["ready"]})
    metrics,selections=evaluate(predictions,plans)
    ready=sum(bool(p["ready"]) for p in plans.values())
    status={"status":"RESEARCH_COMPLETE" if ready==len(plans) else "PARTIAL_RESEARCH_NOT_READY" if ready else "NOT_READY_DESCRIPTIVE_ONLY",
        "production_qualification":False,"production_modified":False,"trained_country_horizon_groups":ready,
        "country_horizon_groups":len(plans),"source_pit_certified":False,
        "completed_initial_state_replays":False}
    study={**plan,"input_csv":str(input_csv),"input_manifest":manifest,"input_provenance_status":manifest.get("provenance_status","not_certified"),
        "control_column":control,"production_qualification":False,"production_modified":False,
        "storm_used_as_model_input":False,"selected_rule_activation":False,"selections":selections,"fit_audit":audit,
        "regularization":{"convex_penalty":25,"huber_delta":20,"recent_level_shrink_days":14,"correction_clip_eur_mwh":40,
            "ridge_penalty":80,"ridge_intercept_penalty":336,"history_days":60},
        "candidate_names":list(BASE_ALGORITHMS)+["component_"+c for c in components],
        "caveats":["Les sources et révisions héritent de leurs limites PIT : aucune qualification automatique de production, même avec 365 jours.",
            "Storm sert uniquement au scoring et à la sélection avant le bloc réservé, jamais aux variables ni à l’apprentissage.",
            "Les poids et corrections utilisent uniquement des journées complètes dont tous les labels sont disponibles à l’origine.",
            f"La règle est choisie une seule fois avant les {holdout_days} derniers jours réservés, avec embargo sur les labels encore indisponibles ; ses paramètres peuvent ensuite se mettre à jour causalement.",
            "Les tableaux descriptifs de tous les candidats ne doivent pas servir à choisir une variante en regardant le bloc réservé.",
            "Le contrôle est conservé si aucun candidat ne passe toutes les contraintes préalables de RMSE, MAE, win rate et seuil historique applicable.",
            "La correction de niveau peut sortir de l’enveloppe convexe des composants. Aucun ajout fixe de 20 EUR/MWh n’est appliqué.",
            "Les challengers optionnels doivent être de vraies prévisions hors échantillon émises avant livraison ; un hash ne prouve pas à lui seul l’absence de fuite.",
            "Le calcul des métriques comparées utilise exactement les mêmes heures pour tous les candidats ; les heures Storm manquantes sont exclues de la comparaison appariée.",
            "Les variantes de pertes concernent l’assemblage des composants. Ce programme ne réentraîne pas des challengers LEAR/CatBoost et ne calibre pas d’intervalles probabilistes."]}
    calibration.require(sha(input_csv)==input_hash and sha(manifest_path)==plan["manifest_sha256"],"Input changed during study")
    predictions.drop(columns=[c for c in predictions if c.startswith("_")]).to_csv(output/"predictions.csv",index=False,float_format="%.17g")
    metrics.to_csv(output/"metrics.csv",index=False)
    atomic_json(output/"study.json",study);atomic_json(output/"status.json",status)
    report(output,metrics,study,status)
    atomic_json(output/"complete.json",{"output_sha256":{n:sha(output/n) for n in OUTPUTS}})
    print("[NYX campaign] "+status["status"]+": "+str(output/"report.html"),flush=True)
    return output/"report.html"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--extra-components",default="",help="Comma-separated OOF forecast columns in the hash-bound input CSV")
    parser.add_argument("--holdout-days",type=int,default=30,help="Untouched final block; use 365 for annual final-expert assessment (no relaxation if insufficient)")
    args=parser.parse_args(argv)
    try:
        run(args.input,args.output,args.manifest,tuple(c.strip() for c in args.extra_components.split(",") if c.strip()),args.holdout_days)
        return 0
    except (ValueError,OSError,KeyError,json.JSONDecodeError) as error:
        print("[NYX campaign] ERROR: "+str(error),file=sys.stderr,flush=True)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
