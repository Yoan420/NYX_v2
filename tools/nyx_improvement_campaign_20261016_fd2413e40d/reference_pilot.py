"""Export an immediate, conditional reference-component experiment.

This does not fabricate final-expert OOF history. It reuses the three genuinely
saved reference signals, original baseline origins and receipt-bound targets.
No model, source collector, production file or foundation state is modified.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from replay import ZONES, atomic_json, grid, hourly, read_json, require, safe_bound, seed_record, sha

SCOPE = "reference_component_pilot"


def execution_roots(annual_repo, app_repo=None):
    roots = [Path(annual_repo).resolve() / 'runs/reports/nyx_annual_standard']
    if app_repo:
        app = Path(app_repo).resolve()
        config = app / 'config/experiment_console.json'
        if config.is_file():
            state = Path(read_json(config).get('state_root', 'runs/.experiment_console'))
            roots.append((state if state.is_absolute() else app / state) / 'executions')
    return roots


def report_sources(seed, app_repo=None):
    """Choose a completed matching report, then verify its reporting-only data."""
    candidates = []
    for root in execution_roots(seed['repo'], app_repo):
        if not root.is_dir():
            continue
        for path in root.glob('**/report_only_sources/reporting_sources/audit.json'):
            report = path.parent.parent.parent / 'report_audit.json'
            if not report.is_file():
                continue
            doc = read_json(report)
            if doc.get('status') != 'REPORTS_COMPLETE':
                continue
            if Path(doc.get('completed_run', '')).resolve() != seed['run_path'].parent.resolve():
                continue
            receipt = read_json(path)
            if receipt.get('protocol') != 'nyx_saved_horizon_reporting_sources_v1' or receipt.get('used_for_prediction') is not False or receipt.get('used_for_training') is not False:
                continue
            candidates.append((pd.Timestamp(receipt['extracted_at_utc']), str(path), path, receipt))
    if not candidates:
        return {}, {}, {'benchmark': 'unavailable', 'reason': 'No completed matching dashboard reporting snapshot found'}
    _, _, audit_path, receipt = max(candidates)
    root = audit_path.parent.parent
    hashes = receipt.get('artifact_sha256', {})
    report_audit = audit_path.parent.parent.parent / 'report_audit.json'
    actual, storm, inputs = {}, {}, {str(audit_path): sha(audit_path), str(report_audit): sha(report_audit)}
    for zone in ZONES:
        evidence = receipt.get('countries', {}).get(zone, {})
        actual_source = evidence.get('observed', {}).get('source', {})
        require(actual_source.get('origin') == 'EPEX' and actual_source.get('policy') == 'epex_only_v1'
                and actual_source.get('used_for_prediction') is False,
                zone + ': reporting target is not declared EPEX-only')
        comparator = evidence.get('storm', {}).get('comparator', {})
        require(comparator.get('column') == 'storm_dashboard_official__q50'
                and comparator.get('used_for_prediction') is False
                and evidence.get('storm', {}).get('source', {}).get('zone') == zone,
                zone + ': official Storm country/comparator identity differs')
        for kind, dest in [('observed', actual), ('storm', storm)]:
            relative = f'reporting_sources/{zone}/{kind}.parquet'
            path = safe_bound(root, relative, hashes)
            frame = hourly(pd.read_parquet(path), relative)
            require(len(frame.columns) == 1, 'One reporting series per country required')
            require(sha(path) == hashes[relative], 'Reporting data changed during read')
            dest[zone] = frame.iloc[:, 0].astype(float)
            inputs[str(path)] = hashes[relative]
    require(sha(audit_path) == inputs[str(audit_path)], 'Reporting receipt changed')
    finite_counts = {zone: int(np.isfinite(storm[zone]).sum()) for zone in ZONES}
    return actual, storm, {'benchmark': 'Storm officiel dashboard' if any(finite_counts.values()) else 'unavailable',
        'finite_source_hours': finite_counts,
        'strict_08_vintage_certified': False, 'consumed_artifact_sha256': inputs,
        'reporting_snapshot': str(audit_path), 'extracted_at_utc': receipt['extracted_at_utc']}


def export(seed, output, app_repo=None):
    output = Path(output).resolve()
    require(not output.exists() or not any(output.iterdir()),
            'Choose a new empty reference-pilot output directory')
    for protected in (seed['bundle'], seed['output'], seed['run_path'].parent):
        require(not output.is_relative_to(protected.resolve()) and not protected.resolve().is_relative_to(output),
                'Research output overlaps immutable annual inputs')
    refreshed, benchmarks, benchmark_info = report_sources(seed, app_repo)
    consumed = {str(seed['run_path']): sha(seed['run_path']), str(seed['receipt_path']): sha(seed['receipt_path'])}
    rows, counts, omitted = [], {}, {}
    bundle, hashes = seed['bundle'], seed['hashes']
    for zone in ZONES:
        frames = []
        for relative in (f'reference_oof/{zone}.parquet', f'baseline_history/{zone}.parquet'):
            path = safe_bound(bundle, relative, hashes)
            frames.append(hourly(pd.read_parquet(path), relative))
            consumed[str(path)] = hashes[relative]
            require(sha(path) == hashes[relative], 'Input changed during read')
        oof, baseline = frames
        names = ['ensemble__q50', 'nyx__q50', 'test2__q50']
        require(set(names) <= set(oof), zone + ': saved reference signals missing')
        require({'actual', 'forecast_origin_utc', 'nyx__q50'} <= set(baseline), 'Baseline origins/labels missing')
        require(oof.index.isin(baseline.index).all(), 'Reference signals outside baseline support')
        np.testing.assert_allclose(oof.nyx__q50, baseline.loc[oof.index, 'nyx__q50'], atol=0, rtol=0)
        target = baseline.actual.reindex(oof.index).astype(float)
        if zone in refreshed:
            fresh = refreshed[zone].reindex(oof.index)
            target = fresh.where(fresh.notna(), target)
        origin = pd.to_datetime(baseline.loc[oof.index, 'forecast_origin_utc'], utc=True)
        valid_days, excluded = [], []
        for day in sorted(set(oof.index.tz_convert('Europe/Paris').date)):
            index = grid(day, day + timedelta(days=1))
            cutoff = pd.Timestamp(f'{day-timedelta(days=1)} 08:00', tz='Europe/Paris').tz_convert('UTC')
            if not index.isin(oof.index).all():
                excluded.append({'day': str(day), 'reason': 'incomplete_physical_day'}); continue
            if not origin.loc[index].eq(cutoff).all():
                excluded.append({'day': str(day), 'reason': 'baseline_origin_differs_from_logical_cutoff'}); continue
            if not np.isfinite(target.loc[index]).all() or not np.isfinite(oof.loc[index, names].to_numpy(float)).all():
                excluded.append({'day': str(day), 'reason': 'missing_label_or_component'}); continue
            # A deliberately conservative availability proxy: after the whole
            # delivery day, rather than pretending a provider timestamp exists.
            available = pd.Timestamp(str(day + timedelta(days=1)), tz='Europe/Paris').tz_convert('UTC')
            for stamp in index:
                rows.append({'country': zone, 'delivery_utc': stamp.isoformat(),
                    'forecast_origin_utc': cutoff.isoformat(), 'horizon_day': 1,
                    'target_price': float(target.loc[stamp]), 'target_available_utc': available.isoformat(),
                    'reference_price': float(oof.loc[stamp, 'ensemble__q50']),
                    'nyx_baseline': float(oof.loc[stamp, 'nyx__q50']),
                    'test2_price': float(oof.loc[stamp, 'test2__q50']),
                    'storm_price': float(benchmarks[zone].get(stamp, np.nan)) if zone in benchmarks else np.nan})
            valid_days.append(str(day))
        counts[zone] = {'days': len(valid_days), 'first_day': valid_days[0] if valid_days else None,
                        'last_day': valid_days[-1] if valid_days else None}
        omitted[zone] = excluded
    frame = pd.DataFrame(rows)
    require(len(frame) and frame.country.nunique() == 4, 'No complete four-country saved reference history')
    for path, digest in {**consumed, **benchmark_info.get('consumed_artifact_sha256', {})}.items():
        require(sha(path) == digest, 'Consumed input changed during export')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'OOF.csv'
    frame.to_csv(path, index=False, float_format='%.17g')
    # A separate reporting-only label file lets the sealed final-forecast
    # collector score already published prices, including dates after the seed.
    label_rows = []
    for zone, values in refreshed.items():
        for stamp, target in values.items():
            if not np.isfinite(target):
                continue
            day = stamp.tz_convert('Europe/Paris').date()
            available = pd.Timestamp(str(day+timedelta(days=1)),tz='Europe/Paris').tz_convert('UTC')
            label_rows.append({'country':zone,'delivery_utc':stamp.isoformat(), 'target_price':float(target),
                'target_available_utc':available.isoformat(),
                'storm_price':float(benchmarks[zone].get(stamp,np.nan)), 'storm_origin_utc':''})
    labels = output/'report_labels.csv'
    pd.DataFrame(label_rows, columns=['country','delivery_utc','target_price','target_available_utc','storm_price','storm_origin_utc']).to_csv(labels,index=False,float_format='%.17g')
    manifest = {'protocol': 'nyx_saved_reference_component_experiment_v1', 'scope': SCOPE,
        'oof_csv_sha256': sha(path), 'provenance_status': 'conditional_saved_reference_components',
        'production_qualification': False, 'final_expert_oof': False,
        'historical_provider_vintages_certified': False, 'foundation_model_called': False,
        'model_fits_in_export': 0, 'component_columns': ['reference_price', 'nyx_baseline', 'test2_price'],
        'control_column': 'nyx_baseline', 'horizons': [1], 'seed_run': str(seed['run_path'].parent),
        'days_per_country': counts, 'excluded_days': omitted,
        'report_labels_csv_sha256': sha(labels),
        'matched_storm_hours_per_country': {zone:int(frame.loc[frame.country.eq(zone),'storm_price'].notna().sum()) for zone in ZONES},
        'consumed_artifact_sha256': consumed, 'benchmark': benchmark_info,
        'target_availability_policy': 'conservative_proxy_delivery_day_end_not_provider_publication_attestation',
        'caveats': ['Pilote J+1 sur les composants de référence sauvegardés : ce ne sont pas les trois experts prix finaux.',
            'Les prévisions et sources historiques sont conditionnelles aux données figées du run donneur ; les millésimes fournisseur ne sont pas certifiés.',
            'Les signaux HGB/Test2 historiques suivent les ajustements par blocs du run donneur ; ils ne sont pas des réentraînements quotidiens de toute la chaîne finale.',
            'Ce test évalue agrégation, correction conditionnelle et intervalles ; il ne qualifie pas les modèles en production.',
            'J+2 exige ses propres prévisions archivées au cutoff commun, jamais des anciens J+1 décalés.']}
    atomic_json(output / 'OOF.manifest.json', manifest)
    print(json.dumps({'status': 'REFERENCE_PILOT_EXPORTED', 'csv': str(path), 'days_per_country': counts,
        'storm': benchmark_info['benchmark'], 'production_qualification': False}, ensure_ascii=False), flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annual-repo', type=Path, required=True)
    parser.add_argument('--app-repo', type=Path)
    parser.add_argument('--seed-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        export(seed_record(args.annual_repo, args.seed_run), args.output, args.app_repo)
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print('[NYX tests] '+str(exc), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
