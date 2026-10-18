"""Causal empirical 80% intervals for a rule selected before its heldout block.

This isolated post-study experiment does not change point forecasts or production.
Temporal dependence/nonstationarity mean nominal coverage is never guaranteed.
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

import calibration

PROTOCOL="nyx_causal_signed_empirical_intervals_v1"
NOMINAL=.8
ALPHA=.2
MINIMUM_DAYS=30
ROLLING_DAYS=60
OUTPUTS=("interval_predictions.csv","metrics.csv","intervals.json","status.json","report.html")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def atomic_json(path,value):
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def validate(frame,document):
    scope=document["scope"]
    calibration.require(scope in ("final_experts","reference_component_pilot"),"Unknown study scope")
    components=calibration.COMPONENTS if scope=="final_experts" else ("reference_price","nyx_baseline","test2_price")
    control="nyx_current" if scope=="final_experts" else "nyx_baseline"
    return calibration.validate_input(frame,components,control,scope)


def empirical_offsets(errors):
    """Outward empirical order statistics; separate signed tails, no symmetry assumption."""
    errors=np.asarray(errors,dtype=float)
    calibration.require(len(errors)>0 and np.isfinite(errors).all(),"Finite prior errors required")
    # NumPy >=1.22 supports method=. These outward ranks avoid interpolation.
    lower=float(np.quantile(errors,ALPHA/2,method="lower"))
    upper=float(np.quantile(errors,1-ALPHA/2,method="higher"))
    return lower,upper


def generate(frame,selections):
    result=frame[["country","horizon_day","delivery_utc","forecast_origin_utc","delivery_day",
                  "target_price","target_available_utc"]].copy()
    for name in ("point","lower","upper","lower_residual_quantile","upper_residual_quantile"):
        result[name]=np.nan
    result["calibration_days"]=0
    result["calibration_hours"]=0
    result["latest_used_label_available_utc"]=""
    result["selected_rule"]=""
    result["heldout_block"]=False
    result["pending_reason"]="GROUP_NOT_READY"
    audit=[]
    indexed={}
    for selection in selections:
        key=(selection["country"],int(selection["horizon_day"]))
        calibration.require(key not in indexed,"Duplicate selection group")
        indexed[key]=selection
    actual=set(frame.groupby(["country","horizon_day"]).groups)
    calibration.require(set(indexed)==actual,"Study selections differ from prediction country/horizon groups")
    for key,group in frame.groupby(["country","horizon_day"],sort=True):
        selection=indexed[key]
        selected=selection["selected"]
        column="prediction_"+selected
        calibration.require(column in frame,"Selected point column missing: "+column)
        result.loc[group.index,"selected_rule"]=selected
        result.loc[group.index,"point"]=group[column]
        if not bool(selection.get("ready",False)):
            print(f"[NYX intervals] {key[0]} J+{key[1]}: NOT_READY; no interval calibration",flush=True)
            continue
        start=pd.Timestamp(selection["holdout_start"]).date()
        stop=pd.Timestamp(selection["holdout_end"]).date()
        block=(group.delivery_day>=start)&(group.delivery_day<=stop)
        result.loc[group.index[block],"heldout_block"]=True
        result.loc[group.index[~block],"pending_reason"]="OUTSIDE_DECLARED_HOLDOUT"
        expected=group.groupby("delivery_day").size()
        for number,(origin,current) in enumerate(group.groupby("forecast_origin_utc",sort=True),1):
            pos=current.index
            day=current.delivery_day.iloc[0]
            if day<start or day>stop:continue
            prior=group[(group.forecast_origin_utc<origin)&(group.delivery_day<day)
                &(group.target_available_utc<=origin)&np.isfinite(group[column])]
            counts=prior.groupby("delivery_day").size()
            complete=[d for d,n in counts.items() if n==expected[d]]
            complete=sorted(complete)[-ROLLING_DAYS:]
            prior=prior[prior.delivery_day.isin(complete)]
            result.loc[pos,"calibration_days"]=len(complete)
            result.loc[pos,"calibration_hours"]=len(prior)
            result.loc[pos,"pending_reason"]="INSUFFICIENT_PRIOR_COMPLETE_DAYS"
            if len(complete)<MINIMUM_DAYS:continue
            lower,upper=empirical_offsets(prior.target_price.to_numpy(float)-prior[column].to_numpy(float))
            finite=np.isfinite(current[column])
            ready_pos=pos[finite]
            result.loc[ready_pos,"lower"]=current.loc[ready_pos,column]+lower
            result.loc[ready_pos,"upper"]=current.loc[ready_pos,column]+upper
            result.loc[ready_pos,"lower_residual_quantile"]=lower
            result.loc[ready_pos,"upper_residual_quantile"]=upper
            result.loc[pos,"pending_reason"]="POINT_NOT_AVAILABLE"
            result.loc[ready_pos,"pending_reason"]=""
            latest=prior.target_available_utc.max().isoformat()
            result.loc[ready_pos,"latest_used_label_available_utc"]=latest
            audit.append({"country":key[0],"horizon_day":int(key[1]),"origin":origin.isoformat(),
                "selected_rule":selected,"calibration_days":len(complete),"calibration_hours":len(prior),
                "latest_used_label_available_utc":latest,"lower_error_quantile":lower,"upper_error_quantile":upper})
            if number%30==0:
                print(f"[NYX intervals] {key[0]} J+{key[1]}: {number}/{group.forecast_origin_utc.nunique()} origins",flush=True)
    return result,audit


def evaluate(frame,selections):
    records=[]
    for selection in selections:
        key=(selection["country"],int(selection["horizon_day"]))
        group=frame[(frame.country==key[0])&(frame.horizon_day==key[1])]
        requested=group[group.heldout_block]
        finite=np.isfinite(requested[["point","lower","upper","target_price"]]).all(axis=1)
        rows=requested.loc[finite]
        values={"country":key[0],"horizon_day":key[1],"selected_rule":selection["selected"],
            "period":"declared_holdout_only","requested_holdout_days":int(selection.get("holdout_days",30)),
            "heldout_hours":len(requested),"evaluated_hours":len(rows),
            "evaluated_days":int(rows.delivery_day.nunique()),"pending_heldout_hours":int((~finite).sum()),
            "nominal_coverage":NOMINAL,"coverage":None,"mean_width":None,"mean_interval_score":None,
            "point_mae_on_same_hours":None,"point_rmse_on_same_hours":None,
            "complete_requested_holdout":False,"status":"NOT_READY" if not selection.get("ready") else "PENDING_PRIOR_ERRORS"}
        if len(rows):
            target,low,high=rows.target_price.to_numpy(float),rows.lower.to_numpy(float),rows.upper.to_numpy(float)
            width=high-low
            score=width+(2/ALPHA)*np.maximum(low-target,0)+(2/ALPHA)*np.maximum(target-high,0)
            error=rows.point.to_numpy(float)-target
            values.update(coverage=float(((target>=low)&(target<=high)).mean()),mean_width=float(width.mean()),
                mean_interval_score=float(score.mean()),point_mae_on_same_hours=float(abs(error).mean()),
                point_rmse_on_same_hours=float(np.sqrt(np.mean(error**2))),status="EMPIRICAL_HELDOUT_SCORED",
                complete_requested_holdout=bool(len(rows)==len(requested) and rows.delivery_day.nunique()==values["requested_holdout_days"]))
        records.append(values)
    return pd.DataFrame(records)


def run(study_dir,output):
    study_dir,output=Path(study_dir).resolve(),Path(output).resolve()
    source={n:study_dir/n for n in ("study.json","predictions.csv","complete.json")}
    calibration.require(all(p.is_file() for p in source.values()),"A completed hash-bound new-study directory is required")
    complete=json.loads(source["complete.json"].read_text(encoding="utf-8"))
    for name in ("study.json","predictions.csv"):
        calibration.require(complete.get("output_sha256",{}).get(name)==sha(source[name]),"Completed study source changed: "+name)
    document=json.loads(source["study.json"].read_text(encoding="utf-8"))
    calibration.require(document.get("production_modified") is False and isinstance(document.get("selections"),list),"Unsupported source study")
    frame=validate(pd.read_csv(source["predictions.csv"]),document)
    plan={"protocol":PROTOCOL,"source_sha256":{n:sha(p) for n,p in source.items()},
        "code_sha256":sha(Path(__file__)),"validation_code_sha256":sha(Path(calibration.__file__)),
        "runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"platform":platform.platform()},
        "nominal_coverage":NOMINAL,"alpha":ALPHA,"minimum_prior_complete_days":MINIMUM_DAYS,"rolling_complete_days":ROLLING_DAYS}
    calibration.require(not output.is_relative_to(study_dir) or output!=study_dir,"Choose an isolated interval output directory")
    output.mkdir(parents=True,exist_ok=True)
    plan_file=output/"run_plan.json"
    if plan_file.exists():
        calibration.require(json.loads(plan_file.read_text(encoding="utf-8"))==plan,"Interval output is pinned to different sources, code or runtime; choose a new output directory")
        completed=output/"complete.json"
        if completed.exists():
            hashes=json.loads(completed.read_text(encoding="utf-8"))
            calibration.require(all((output/n).is_file() and sha(output/n)==h for n,h in hashes["output_sha256"].items()),"Completed interval output changed")
            print("[NYX intervals] Reuse verified completed report: "+str(output/"report.html"),flush=True)
            return output/"report.html"
    else:
        calibration.require(not any((output/n).exists() for n in OUTPUTS),"Unpinned existing interval outputs")
        atomic_json(plan_file,plan)
    predictions,audit=generate(frame,document["selections"])
    metrics=evaluate(predictions,document["selections"])
    calibrated=int(np.isfinite(predictions.lower).sum())
    scored=int(metrics.evaluated_hours.sum())
    status={"status":"EMPIRICAL_INTERVALS_SCORED" if scored else "NOT_READY_NO_CALIBRATED_HOLDOUT",
        "production_qualification":False,"production_modified":False,"calibrated_prediction_hours":calibrated,
        "heldout_evaluated_hours":scored,"nominal_coverage":NOMINAL,"formal_coverage_guarantee":False}
    caveats=["Les intervalles empiriques visent 80 % de couverture nominale, sans garantie formelle sous dépendance temporelle et changement de régime.",
        "La règle de point est celle choisie avant le bloc réservé. Aucun choix supplémentaire n’utilise les résultats de ce bloc.",
        "Les erreurs signées proviennent uniquement de prédictions historiques réellement émises, sur les 60 dernières journées complètes disponibles à l’origine ; minimum 30 journées.",
        "Les quantiles inférieur 10 % et supérieur 90 % sont estimés séparément : l’enveloppe peut être asymétrique et ne pas être centrée sur le point.",
        "Aucun résidu du jour courant, aucune donnée Storm, aucun mélange entre pays ou entre J+1 et J+2 n’entre dans la calibration.",
        "Seul le bloc réservé déclaré dans l’étude est évalué. Les erreurs de point, la couverture, la largeur et l’interval score utilisent exactement les mêmes heures.",
        "Un groupe non prêt dans l’étude initiale ne reçoit aucun intervalle ; les données et révisions restent non certifiées pour la production."]
    result={**plan,"study_dir":str(study_dir),"scope":document["scope"],"selections":document["selections"],
        "production_qualification":False,"production_modified":False,"formal_coverage_guarantee":False,
        "quantile_rule":"signed_error_10pct_lower_order_statistic_and_90pct_upper_order_statistic",
        "caveats":caveats,"fit_audit":audit}
    calibration.require(all(sha(p)==plan["source_sha256"][n] for n,p in source.items()),"Source study changed during interval calculation")
    predictions.to_csv(output/"interval_predictions.csv",index=False,float_format="%.17g")
    metrics.to_csv(output/"metrics.csv",index=False)
    atomic_json(output/"intervals.json",result);atomic_json(output/"status.json",status)
    table=metrics.to_html(index=False,float_format=lambda x:f"{x:.4f}")
    notes="".join("<li>"+html.escape(item)+"</li>" for item in caveats)
    (output/"report.html").write_text(f'''<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX — intervalles empiriques</title>
<style>body{{font:15px system-ui;background:#f6f8fb;color:#172238;margin:30px}}table{{border-collapse:collapse;background:white;font-size:13px}}td,th{{padding:8px;border:1px solid #dce2eb}}.scroll{{overflow:auto}}li{{margin:8px 0}}</style>
<h1>NYX — intervalles empiriques à 80 %</h1><p><strong>{status['status']}</strong>. Qualification de production : NON.</p>
<div class="scroll">{table}</div><ul>{notes}</ul></html>''',encoding="utf-8")
    atomic_json(output/"complete.json",{"output_sha256":{n:sha(output/n) for n in OUTPUTS}})
    print("[NYX intervals] "+status["status"]+": "+str(output/"report.html"),flush=True)
    return output/"report.html"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        run(args.study_dir,args.output)
        return 0
    except (ValueError,OSError,KeyError,json.JSONDecodeError) as error:
        print("[NYX intervals] ERROR: "+str(error),file=sys.stderr,flush=True)
        return 2


if __name__=="__main__":raise SystemExit(main())
