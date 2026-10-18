"""Attach verified reporting comparators and score the available final forecasts."""
from __future__ import annotations
import argparse
from datetime import timedelta
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import calibration
from reference_pilot import report_sources
from replay import atomic_json, read_json, require, seed_record, sha


def run(input_csv, output, annual_repo, app_repo=None, calibrate_when_ready=False):
    input_csv, output = Path(input_csv).resolve(), Path(output).resolve()
    require(not output.exists() or not any(output.iterdir()), 'Choose a new empty final-study directory')
    manifest_path = input_csv.with_suffix('.manifest.json')
    manifest = read_json(manifest_path)
    require(manifest.get('scope','final_experts') == 'final_experts', 'Final-expert input required')
    require(manifest.get('oof_csv_sha256') == sha(input_csv), 'Final OOF CSV differs from its manifest')
    seed = seed_record(annual_repo, manifest.get('seed_run'))
    actuals, storm, reporting = report_sources(seed, app_repo)
    frame = pd.read_csv(input_csv)
    require(set(calibration.REQUIRED) <= set(frame), 'Final OOF schema missing')
    require(len(frame)>0, 'No fully observed final forecasts yet; see pending_forecasts.csv')
    index = pd.to_datetime(frame.delivery_utc, utc=True)
    source_hashes = {str(input_csv):sha(input_csv),str(manifest_path):sha(manifest_path),
                     **reporting.get('consumed_artifact_sha256',{})}
    for zone in frame.country.unique():
        mask = frame.country.eq(zone).to_numpy()
        stamps = pd.DatetimeIndex(index[mask])
        if zone in actuals:
            values = actuals[zone].reindex(stamps).to_numpy(float)
            fresh = np.isfinite(values)
            rows = frame.index[mask]
            frame.loc[rows[fresh],'target_price'] = values[fresh]
            available = [pd.Timestamp(str(d+timedelta(days=1)),tz='Europe/Paris').tz_convert('UTC').isoformat()
                         for d in stamps.tz_convert('Europe/Paris').date]
            # Attaching a comparator must never make an explicitly delayed
            # label appear available earlier than the input provenance says.
            old_available = pd.to_datetime(frame.loc[rows,'target_available_utc'],utc=True)
            conservative = pd.Series(pd.to_datetime(available,utc=True),index=rows)
            retained = old_available.where(old_available>=conservative,conservative)
            frame.loc[rows[fresh],'target_available_utc'] = retained.loc[rows[fresh]].map(lambda x:x.isoformat())
            frame.loc[rows,'storm_price'] = storm[zone].reindex(stamps).to_numpy(float)
    valid = calibration.validate_input(frame)
    metrics=[]
    names=['nyx_current','reference_price','expert_fr_residual','expert_cwe_residual','expert_cwe_absolute']
    for (zone,horizon), group in valid.groupby(['country','horizon_day']):
        for name in names:
            finite=np.isfinite(group.storm_price)
            for common in (False,True):
                subset=group.loc[finite] if common else group
                if not len(subset):continue
                error=subset[name]-subset.target_price
                matched=subset.loc[np.isfinite(subset.storm_price)]
                metrics.append({'country':zone,'horizon_day':int(horizon),'component':name,
                    'matched_storm_only':common,'hours':len(subset),'days':subset.delivery_day.nunique(),
                    'mae':float(abs(error).mean()),'rmse':float(np.sqrt(np.mean(error**2))),
                    'bias':float(error.mean()),
                    'storm_rmse':float(np.sqrt(np.mean((matched.storm_price-matched.target_price)**2))) if len(matched) else None,
                    'strict_win_rate':float((abs(matched[name]-matched.target_price)<abs(matched.storm_price-matched.target_price)).mean()) if len(matched) else None})
    output.mkdir(parents=True,exist_ok=True)
    copied=output/'OOF.csv'
    frame.to_csv(copied,index=False,float_format='%.17g')
    result_manifest={**manifest,'oof_csv_sha256':sha(copied),'parent_oof_csv_sha256':source_hashes[str(input_csv)],
        'reporting_attachment':reporting,'reporting_attachment_sha256':source_hashes,
        'target_availability_policy':'conservative_proxy_delivery_day_end_not_provider_publication_attestation'}
    atomic_json(output/'OOF.manifest.json',result_manifest)
    metrics_frame=pd.DataFrame(metrics)
    metrics_frame.to_csv(output/'metrics.csv',index=False)
    states=[]
    if calibrate_when_ready:
        for horizon,part in frame.groupby('horizon_day'):
            counts=valid.loc[valid.horizon_day.eq(horizon)].groupby('country').delivery_day.nunique()
            ready=len(counts)==4 and counts.min()>=90
            status={'horizon_day':int(horizon),'days_per_country':{z:int(v) for z,v in counts.items()},
                    'minimum_requested_days':90,'adaptive_study_ready':bool(ready)}
            if ready:
                data_dir=output/f'horizon_{int(horizon)}';data_dir.mkdir()
                p=data_dir/'OOF.csv';part.to_csv(p,index=False,float_format='%.17g')
                atomic_json(data_dir/'OOF.manifest.json',{**result_manifest,'oof_csv_sha256':sha(p)})
                command=[sys.executable,str(Path(__file__).parent/'calibration.py'),'--input',str(p),
                         '--output',str(data_dir/'adaptive_study'),'--warmup-days','45','--holdout-days','30']
                subprocess.run(command,check=True)
                status['report']=str(data_dir/'adaptive_study/report.html')
            else:
                status['reason']='Insufficient final-expert OOF history; forecasts were scored, adaptive parameters were not inferred from a few days.'
            states.append(status)
    for path,digest in source_hashes.items():
        require(sha(path)==digest,'Research source changed during scoring')
    atomic_json(output/'status.json',{'status':'AVAILABLE_FINAL_FORECASTS_SCORED','adaptive_studies':states,
        'production_qualification':False,'production_modified':False})
    note='Prévisions finales disponibles, séparées par horizon. Le rejeu hérite des limites de ses sources historiques. Ces résultats ne constituent pas une qualification annuelle. Une étude adaptative exige au moins 90 jours par pays et horizon ; le bloc final réservé couvre 30 jours.'
    body='<h1>NYX — tests des prévisions finales disponibles</h1><p>'+html.escape(note)+'</p>'
    body+=metrics_frame.to_html(index=False,float_format=lambda x:f'{x:.4f}')
    body+='<h2>Couverture pour les tests adaptatifs</h2><pre>'+html.escape(json.dumps(states,ensure_ascii=False,indent=2))+'</pre>'
    (output/'report.html').write_text('<!doctype html><html lang="fr"><meta charset="utf-8"><title>NYX — tests finaux</title><style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}pre{white-space:pre-wrap}</style>'+body+'</html>',encoding='utf-8')
    print(json.dumps({'status':'AVAILABLE_FINAL_FORECASTS_SCORED','report':str(output/'report.html'),
                      'adaptive_studies':states},ensure_ascii=False),flush=True)
    return output/'report.html'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--annual-repo',type=Path,required=True)
    parser.add_argument('--app-repo',type=Path)
    parser.add_argument('--calibrate-when-ready',action='store_true')
    args=parser.parse_args(argv)
    try:
        run(args.input,args.output,args.annual_repo,args.app_repo,args.calibrate_when_ready)
        return 0
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as exc:
        print('[NYX tests] '+str(exc),file=sys.stderr,flush=True)
        return 2

if __name__=='__main__':raise SystemExit(main())
