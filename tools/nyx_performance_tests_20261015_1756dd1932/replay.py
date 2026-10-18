"""Isolated, resumable NYX price research; never modifies or activates production.

A seed replay is CONDITIONAL on the donor's recovered historical profiles. It
re-fits the current 365-day price/reference recipes at each inner J+1 origin,
without rerunning Chronos. It never relabels J+1 history as J+2.
"""
from __future__ import annotations
import argparse
from datetime import date, timedelta
import hashlib
import importlib
from importlib.metadata import version
import platform
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import numpy as np
import pandas as pd

ZONES = ("FR", "DE", "BE", "NL")
FAMILIES = ("fr_residual_1000", "cwe_residual_2000", "cwe_absolute_2000")
PROTOCOL = "nyx_conditional_price_replay_v1"
COLUMNS = ("country", "delivery_utc", "forecast_origin_utc", "horizon_day",
           "target_price", "target_available_utc", "reference_price",
           "expert_fr_residual", "expert_cwe_residual", "expert_cwe_absolute",
           "nyx_current", "storm_price", "storm_origin_utc")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
            sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def grid(first, stop):
    return pd.date_range(str(first), str(stop), tz="Europe/Paris", freq="h",
                         inclusive="left").tz_convert("UTC")


def hourly(frame, label):
    index = frame.index
    require(isinstance(index, pd.DatetimeIndex) and str(index.tz) == "UTC"
        and len(index) and index.is_unique and index.is_monotonic_increasing
        and not index.hasnans and index.equals(index.floor("h")),
        label + ": ordered physical UTC hours required")
    return frame


def safe_bound(root, relative, hashes):
    require(isinstance(relative, str) and "\\" not in relative and relative in hashes
        and not Path(relative).is_absolute() and ":" not in relative and ".." not in Path(relative).parts,
        "Unbound or unsafe research input: " + str(relative))
    root = Path(root).resolve()
    path = root / relative
    require(path.is_file() and not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents) if p != root.parent) and path.resolve().is_relative_to(root)
        and sha(path) == hashes[relative], "Research input missing or changed: " + str(path))
    return path


def seed_record(annual_repo, seed_run=None):
    repo = Path(annual_repo).resolve()
    candidates = [Path(seed_run)] if seed_run else sorted((repo / "runs").glob("**/run.json"))
    found = []
    for candidate in candidates:
        path = candidate / "run.json" if candidate.is_dir() else candidate
        if not path.is_file():
            continue
        try:
            record = read_json(path)
        except (ValueError, OSError):
            if seed_run:
                raise
            continue
        if record.get("state") != "REPORTS_COMPLETE":
            continue
        day = record.get("operational_snapshot", {}).get("delivery_day")
        if isinstance(day, str):
            found.append((day, str(path), path, record))
    require(found, "No completed annual run found; provide --seed-run <folder containing run.json>")
    day, _, path, record = max(found)
    require(date.fromisoformat(day).isoformat() == day, "Invalid donor delivery day")
    bundle, output = Path(record["bundle"]).resolve(), Path(record["output"]).resolve()
    require(bundle.is_relative_to(repo) and output.is_relative_to(repo), "Donor paths escape annual repository")
    receipt_path = output / "receipt.json"
    receipt = read_json(receipt_path)
    require(receipt.get("status") == "REPORTS_COMPLETE" and receipt.get("delivery_day") == day
        and receipt.get("future_labels_used") is False, "Donor reports are not sealed or contain future labels")
    hashes = receipt.get("input_sha256")
    require(isinstance(hashes, dict) and hashes, "Donor has no bound inputs")
    saturn = read_json(safe_bound(bundle, "source_receipts/saturn.json", hashes))
    require(saturn.get("price_reference") == "EPEX" and saturn.get("price_reference_policy") == "epex_only_v1"
        and saturn.get("source_mixing") is False and saturn.get("actual_values_imputed") is False,
        "Research donor must use the sealed, unmixed EPEX target")
    return {"repo": repo, "run_path": path, "run": record, "day": day,
            "bundle": bundle, "output": output, "receipt_path": receipt_path,
            "receipt": receipt, "hashes": hashes}


def coverage(seed):
    bundle, hashes = seed["bundle"], seed["hashes"]
    starts, stops, origins = [], [], []
    details = {}
    for zone in ZONES:
        frames = {}
        for relative in (f"reference_oof/{zone}.parquet", f"reference_inputs/base292/{zone}.parquet",
                         f"baseline_history/{zone}.parquet", f"features/cwe_residual_2000/{zone}.parquet"):
            frames[relative] = hourly(pd.read_parquet(safe_bound(bundle, relative, hashes)), relative)
        oof = frames[f"reference_oof/{zone}.parquet"]
        ref = frames[f"reference_inputs/base292/{zone}.parquet"]
        base = frames[f"baseline_history/{zone}.parquet"]
        expert = frames[f"features/cwe_residual_2000/{zone}.parquet"]
        first = max(oof.index[0].tz_convert("Europe/Paris").date() + timedelta(days=90),
                    ref.index[0].tz_convert("Europe/Paris").date() + timedelta(days=365),
                    base.index[0].tz_convert("Europe/Paris").date() + timedelta(days=372))
        stop = min(date.fromisoformat(seed["day"]),
                   oof.index[-1].tz_convert("Europe/Paris").date() + timedelta(days=1),
                   ref.index[-1].tz_convert("Europe/Paris").date())
        starts.append(first)
        stops.append(stop)
        details[zone] = {"baseline_first_day": str(base.index[0].tz_convert("Europe/Paris").date()),
            "reference_oof_first_day": str(oof.index[0].tz_convert("Europe/Paris").date()),
            "reference_input_first_day": str(ref.index[0].tz_convert("Europe/Paris").date()),
            "stored_expert_first_day": str(expert.index[0].tz_convert("Europe/Paris").date()),
            "earliest_conditional_origin": str(first), "stop_day_exclusive": str(stop)}
    first, stop = max(starts), min(stops)
    days = [str(value.date()) for value in pd.date_range(str(first), str(stop), freq="D", inclusive="left")] if first < stop else []
    return {"protocol": PROTOCOL, "seed_day": seed["day"], "seed_run": str(seed["run_path"].parent),
        "conditional_j1_available_days": len(days), "conditional_j1_days": days, "countries": details,
        "baseline_replay_required": False, "j2_retrospective_available": False,
        "explanation": "835 states include foundation warmup. A 365-day price fit and 90-day reference policy need earlier held-out reference forecasts; current-run annual statistics do not provide those automatically.",
        "provenance_status": "conditional", "production_qualification": False}


def configure(annual_repo, implementation_root=None):
    repo = Path(annual_repo).resolve()
    require((repo / "chronos2_hourly").is_dir(), "Annual repository package missing")
    sys.path.insert(0, str(repo))
    package = importlib.import_module("chronos2_hourly")
    if implementation_root:
        extra = Path(implementation_root).resolve()
        require(extra.is_dir(), "Research daily-refit implementation missing")
        package.__path__ = [str(extra), *list(package.__path__)]
    return repo


def load_research_inputs(seed):
    """Rebuild older auxiliary rows as declared missing, without inventing data.

    The reference inputs are kept byte-for-byte as sealed. Only expert rows
    predating their stored 365-day window are reconstructed. Existing captured
    expert rows are never overwritten by that reconstruction.
    """
    from chronos2_hourly.nyx_annual_cpu_bundle_builder import BuiltFeatures, build_feature_matrices
    from chronos2_hourly import nyx_thermal_capacity_features as thermal
    bundle, hashes = seed["bundle"], seed["hashes"]
    consumed = {str(seed["run_path"]): sha(seed["run_path"]), str(seed["receipt_path"]): sha(seed["receipt_path"])}
    def read(relative):
        path = safe_bound(bundle, relative, hashes)
        consumed[str(path)] = hashes[relative]
        frame = pd.read_parquet(path)
        require(sha(path) == hashes[relative], "Input changed during read")
        return frame
    bases = {z: hourly(read(f"baseline_history/{z}.parquet"), z) for z in ZONES}
    references = {kind: {z: hourly(read(f"reference_inputs/{kind}/{z}.parquet"), kind + z) for z in ZONES}
                  for kind in ("base292", "augmented334", "test2")}
    index = references["base292"]["FR"].index
    from chronos2_hourly.nyx_local_extra_features import cutoff_times
    for zone, frame in bases.items():
        require("forecast_origin_utc" in frame, zone + ": baseline origin lineage missing")
        stamps = pd.to_datetime(frame.forecast_origin_utc, utc=True)
        limits = cutoff_times(frame.index, "Europe/Paris")
        require(stamps.notna().all() and (stamps <= limits).all(),
            zone + ": baseline includes a late reconstruction; it cannot enter this chronological replay")
    actual = {z: bases[z].actual for z in ZONES}
    quantiles = {z: bases[z][["nyx__q10", "nyx__q50", "nyx__q90"]].rename(columns=lambda n: n[5:]) for z in ZONES}
    prices = {z: read(f"source_artifacts/auction_prices/{z}.parquet").price_eur_mwh for z in ZONES}
    covariates = read("source_artifacts/saturn/covariates.parquet")
    origins_frame = read("source_artifacts/saturn/origins.parquet")
    require("forecast_origin_utc" in origins_frame, "Sealed daily covariate origins missing")
    origins = {z: pd.to_datetime(origins_frame.forecast_origin_utc, utc=True).copy() for z in ZONES}
    for zone in ZONES:
        overlap = origins[zone].index.intersection(bases[zone].index)
        baseline_origins = pd.to_datetime(bases[zone].loc[overlap, "forecast_origin_utc"], utc=True)
        existing = origins[zone].loc[overlap]
        origins[zone].loc[overlap] = existing.where(existing >= baseline_origins, baseline_origins)
    def extend(frame):
        missing = index.difference(frame.index)
        if len(missing):
            addition = pd.DataFrame(np.nan, index=missing, columns=frame.columns)
            for col in addition:
                if col.endswith("__available"):
                    addition[col] = 0.
            frame = pd.concat([frame, addition]).sort_index()
        return frame
    jao = "source_artifacts/jao_initial/"
    old = jao + "original_features.parquet"
    new = jao + "refreshed_features.parquet"
    if old not in hashes:
        old = new = jao + "features_365d_plus_delivery.parquet"
    thermal_sources = {name: read(f"source_artifacts/thermal_capacity/{name}.parquet") for name in thermal.SOURCES}
    # The original collector frames naturally preserve NaN availability after
    # exact-hour alignment. Newer prospective Pmax frames are converted only
    # where they contain real complete captured values.
    for name, frame in tuple(thermal_sources.items()):
        if list(frame) == ["pmax_gw", "asof_query_utc"]:
            keep = frame.pmax_gw.notna()
            source = frame.loc[keep]
            stamps = pd.to_datetime(source.asof_query_utc, utc=True)
            thermal_sources[name] = pd.DataFrame({"value_time_utc": source.index,
                "snapshot_time_utc": stamps.to_numpy(), "revision_time_utc": stamps.to_numpy(),
                "value": source.pmax_gw.to_numpy(), "downloaded_at_utc": stamps.to_numpy()})
            for col in ("value_time_utc", "snapshot_time_utc", "revision_time_utc"):
                thermal_sources[name][col] = pd.to_datetime(thermal_sources[name][col], utc=True)
    rebuilt = build_feature_matrices(seed["day"], prices=prices, nyx_quantiles=quantiles,
        forecast_origins=origins, covariates=covariates,
        fuel=read("source_artifacts/fuel/market_fuel_features.parquet"),
        hydro_features=extend(read("source_artifacts/public_hydro/features.parquet")),
        exchange_features=extend(read("source_artifacts/lagged_exchange/features.parquet")),
        jao_original=extend(read(old)), jao_refreshed=extend(read(new)),
        thermal_sources=thermal_sources, feature_index=index, reference_index=index)
    features, extended = {}, {}
    for family in FAMILIES:
        features[family], extended[family] = {}, {}
        for zone in ZONES:
            original = read(f"features/{family}/{zone}.parquet")
            require(list(original) == list(rebuilt.features[family][zone]), "Feature schema changed")
            missing = index.difference(original.index)
            features[family][zone] = pd.concat([rebuilt.features[family][zone].loc[missing], original]).sort_index().loc[index]
            extended[family][zone] = len(missing)
    prior = {z: read(f"reference_oof/{z}.parquet") for z in ZONES}
    built = BuiltFeatures(features, references["base292"], references["augmented334"], references["test2"],
        {"conditional_replay": True, "older_expert_rows_rebuilt": extended,
         "uncaptured_auxiliary_rows": "NaN with explicit zero availability; no numerical imputation"})
    return {"built": built, "actual": actual, "quantiles": quantiles, "prior": prior,
            "consumed": consumed, "feature_audit": built.audits}


def verify_inputs(inputs):
    for name, digest in inputs.items():
        require(Path(name).is_file() and sha(name) == digest, "Frozen research input changed: " + name)


def predict_origin(inputs, day, destination, threads=8, progress=None):
    from chronos2_hourly.nyx_daily_refit_reference import fit_daily_reference
    from chronos2_hourly.nyx_pooled_cpu_price_model import PooledConfig, fit_pooled_block
    from chronos2_hourly.nyx_annual_cpu_live import compose_price
    first = date.fromisoformat(day)
    current = grid(first, first + timedelta(days=1))
    cutoff = pd.Timestamp(f"{first - timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC")
    # Preserve each inner origin's target embargo, even though the donor has
    # later labels available for evaluation after prediction sealing.
    bounded_actual = {z: value.where(value.index < current[0]) for z, value in inputs["actual"].items()}
    prior = {z: value.loc[value.index < current[0]].copy() for z, value in inputs["prior"].items()}
    refs, _, ref_audit, _ = fit_daily_reference(day, cutoff, inputs["built"], inputs["quantiles"],
        bounded_actual, prior, destination, progress=progress)
    points, audits = {}, {}
    expert_index = inputs["built"].features[FAMILIES[0]][ZONES[0]].index
    fit_actual = {z: bounded_actual[z].reindex(expert_index) for z in ZONES}
    fit_nyx = {z: inputs["quantiles"][z].q50.reindex(expert_index) for z in ZONES}
    for family in FAMILIES:
        config = PooledConfig(target_mode="absolute" if family == "cwe_absolute_2000" else "residual",
                              iterations=1000 if family == "fr_residual_1000" else 2000)
        result, audit = fit_pooled_block(inputs["built"].features[family], fit_actual,
            fit_nyx, origin_day=day,
            stop_day=str(first + timedelta(days=1)), config=config,
            model_path=Path(destination) / "price_models" / (family + ".cbm"),
            initial_training_day=str(first - timedelta(days=365)), thread_count=threads)
        require(audit.get("training_days") == 365 and audit.get("training_labels_before_origin") is True
                and audit.get("Storm_used_as_input") is False, "Price replay embargo or recipe changed")
        points[family], audits[family] = result, audit
        if progress:
            progress("price_expert", expert=family, status="complete")
    frames = {}
    for zone in ZONES:
        experts = {family: points[family][zone].point.loc[current] for family in FAMILIES}
        reference = refs[zone].loc[current]
        frames[zone] = pd.DataFrame({"price_eur_mwh": compose_price(zone, experts, reference),
            "reference_eur_mwh": reference,
            **{family + "_eur_mwh": point for family, point in experts.items()},
            "forecast_origin_utc": cutoff, "horizon_day": 1}, index=current)
    return frames, {"reference": ref_audit, "price_experts": audits,
                    "negative_models_replayed": False, "training_days": 365}


def validate_prediction(frame, zone, day):
    current = grid(date.fromisoformat(day), date.fromisoformat(day) + timedelta(days=1))
    require(frame.index.equals(current), zone + ": incomplete replay delivery day")
    names = ["price_eur_mwh", "reference_eur_mwh", *[f + "_eur_mwh" for f in FAMILIES]]
    require(set(names) <= set(frame) and np.isfinite(frame[names].to_numpy(float)).all(),
            zone + ": invalid price replay")
    expected = pd.Timestamp(f"{date.fromisoformat(day)-timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC")
    require(pd.to_datetime(frame.forecast_origin_utc, utc=True).eq(expected).all()
            and frame.horizon_day.eq(1).all(), "Forecast origin/horizon differs")


def replay(seed, output, *, days=0, threads=8, dry_run=False, implementation_root=None, inputs=None, predictor=predict_origin):
    output = Path(output).resolve()
    require(not output.is_relative_to(seed["bundle"]) and not output.is_relative_to(seed["output"])
            and not seed["bundle"].is_relative_to(output) and not seed["output"].is_relative_to(output),
            "Research outputs must be isolated from the annual donor")
    report = coverage(seed)
    choices = report["conditional_j1_days"]
    require(choices, "No complete causal reference-policy window remains; use collect or provide an earlier/full daily archive")
    require(days >= 0 and (days == 0 or days <= len(choices)),
            f"Requested {days} origins; only {len(choices)} conditional J+1 origins have complete 365-day training and prior90 support")
    selected = choices[-days:] if days else choices
    if dry_run:
        return {**report, "selected_days": selected, "models_per_origin": "12 HGB + 2 Test2 pairs + 3 price experts; no foundation calls", "dry_run": True}
    configure(seed["repo"], implementation_root)
    if inputs is None:
        inputs = load_research_inputs(seed)
    source_hashes = inputs["consumed"]
    code_hashes = {"replay.py": sha(__file__)}
    for module in ("nyx_daily_refit_reference", "nyx_pooled_cpu_price_model", "nyx_annual_cpu_live", "nyx_annual_cpu_bundle_builder"):
        code_hashes[module] = sha(importlib.import_module("chronos2_hourly." + module).__file__)
    plan = {"protocol": PROTOCOL, "days": selected, "source_sha256": source_hashes,
        "seed_run": str(seed["run_path"].parent),
        "code_sha256": code_hashes, "runtime_versions": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "catboost", "scikit-learn", "joblib")}},
        "threads": threads, "training_days": 365,
        "historical_provider_vintages_certified": False, "provenance_status": "conditional",
        "production_qualification": False, "scope": "final_experts", "j2_relabeling_permitted": False,
        "feature_audit": inputs.get("feature_audit", {})}
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "replay_plan.json"
    if plan_path.exists():
        require(read_json(plan_path) == plan, "Research plan, code, donor inputs or threads changed; choose a new output directory")
    else:
        atomic_json(plan_path, plan)
    digest = sha(plan_path)
    for position, day in enumerate(selected, 1):
        destination = output / "origins" / day
        receipt = destination / "receipt.json"
        if receipt.exists():
            document = read_json(receipt)
            require(document.get("plan_sha256") == digest and document.get("protocol") == PROTOCOL
                and document.get("day") == day and document.get("predictions_sealed_before_scoring") is True,
                "Resumed origin belongs to a different plan or has no prediction seal")
            inventory = document.get("artifact_sha256")
            require(isinstance(inventory, dict) and {"model_audit.json", *[z + ".parquet" for z in ZONES]} <= set(inventory),
                "Resumed prediction seal lacks mandatory bound artifacts")
            verify_inputs(source_hashes)
            for relative, expected in inventory.items():
                safe_bound(destination, relative, document["artifact_sha256"])
            for zone in ZONES:
                validate_prediction(pd.read_parquet(destination / (zone + ".parquet")), zone, day)
        else:
            if destination.exists():
                abandoned = output / "interrupted" / (day + "_" + uuid4().hex)
                abandoned.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(abandoned))
            destination.mkdir(parents=True)
            def progress(stage, **details):
                print(json.dumps({"stage": stage, "day": day, "completed_days": position-1,
                    "total_days": len(selected), **details}, default=str), flush=True)
            verify_inputs(source_hashes)
            frames, audit = predictor(inputs, day, destination, threads=threads, progress=progress)
            require(set(frames) == set(ZONES), "Four countries required")
            for zone, frame in frames.items():
                validate_prediction(frame, zone, day)
                frame.to_parquet(destination / (zone + ".parquet"))
            atomic_json(destination / "model_audit.json", audit)
            verify_inputs(source_hashes)
            artifacts = {p.relative_to(destination).as_posix(): sha(p) for p in destination.rglob("*") if p.is_file()}
            atomic_json(receipt, {"protocol": PROTOCOL, "day": day, "plan_sha256": digest,
                "predictions_sealed_before_scoring": True, "artifact_sha256": artifacts})
        atomic_json(output / "status.json", {"state": "PREDICTING", "completed": position, "total": len(selected), "day": day})
    # Open evaluation targets only after every requested prediction has sealed.
    verify_inputs(source_hashes)
    require(sha(__file__) == code_hashes["replay.py"], "Research replay code changed during execution")
    rows = []
    for day in selected:
        for zone in ZONES:
            frame = pd.read_parquet(output / "origins" / day / (zone + ".parquet"))
            target = inputs["actual"][zone].reindex(frame.index)
            require(np.isfinite(target.to_numpy(float)).all(), "Missing EPEX evaluation targets")
            rows.append(canonical_rows(frame, zone, target, availability_policy="assumed_conservative"))
    verify_inputs(source_hashes)
    result = pd.concat(rows, ignore_index=True).sort_values(["forecast_origin_utc", "country", "delivery_utc"])
    csv = output / "OOF.csv"
    result.to_csv(csv, index=False)
    manifest = {**plan, "oof_csv_sha256": sha(csv), "forecast_source_status": "conditional_inner_origin_refits",
        "target_availability_policy": "research_assumption_delivery_day_plus_1_at_08_Paris; current donor EPEX revisions; original publication not certified",
        "caveats": ["Historical profiles and EPEX revisions come from the sealed donor cutoff; original provider vintages are not certified.",
            "Uncaptured auxiliary descriptors in older expert training rows remain NaN with availability zero.",
            "Only J+1 is reconstructed; J+2 requires actual shared-cutoff historical captures.",
            "Negative-price classifiers do not affect price predictions and are not replayed.",
            "This short pilot cannot qualify annual production performance or prove an improvement."],
        "replay_days": len(selected), "first_delivery_day": selected[0], "last_delivery_day": selected[-1]}
    atomic_json(output / "OOF.manifest.json", manifest)
    atomic_json(output / "status.json", {"state": "COMPLETE", "completed": len(selected), "total": len(selected), "csv": str(csv)})
    return {"state": "COMPLETE", "csv": str(csv), "manifest": str(output / "OOF.manifest.json"), "days": len(selected), "provenance_status": "conditional"}


def canonical_rows(frame, zone, target, *, availability_policy="source_receipt", available_at=None):
    hourly(frame, zone)
    require(target.index.equals(frame.index), "Target and forecasts differ")
    require("forecast_origin_utc" in frame and "horizon_day" in frame, "Forecast lineage missing")
    required = ["price_eur_mwh", "reference_eur_mwh", *[f + "_eur_mwh" for f in FAMILIES]]
    require(set(required) <= set(frame) and np.isfinite(frame[required].to_numpy(float)).all(), "Final price expert components missing")
    if availability_policy == "assumed_conservative":
        available = [pd.Timestamp(f"{d+timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC")
                     for d in frame.index.tz_convert("Europe/Paris").date]
    else:
        stamp = pd.Timestamp(available_at)
        require(stamp.tzinfo is not None and not pd.isna(stamp), "Explicit label-source cutoff required")
        available = [stamp.tz_convert("UTC")] * len(frame)
    origins = pd.to_datetime(frame.forecast_origin_utc, utc=True)
    horizons = frame.horizon_day.astype(int)
    for delivery, origin, horizon in zip(frame.index, origins, horizons):
        require(horizon in (1, 2) and origin < delivery
            and (delivery.tz_convert("Europe/Paris").date()-origin.tz_convert("Europe/Paris").date()).days == horizon,
            "Forecast vintage/horizon invalid; J+1 cannot be relabeled as J+2")
    return pd.DataFrame({"country": zone, "delivery_utc": frame.index,
        "forecast_origin_utc": origins.to_numpy(), "horizon_day": horizons.to_numpy(),
        "target_price": target.to_numpy(), "target_available_utc": available,
        "reference_price": frame.reference_eur_mwh.to_numpy(),
        "expert_fr_residual": frame.fr_residual_1000_eur_mwh.to_numpy(),
        "expert_cwe_residual": frame.cwe_residual_2000_eur_mwh.to_numpy(),
        "expert_cwe_absolute": frame.cwe_absolute_2000_eur_mwh.to_numpy(),
        "nyx_current": frame.price_eur_mwh.to_numpy(), "storm_price": np.nan, "storm_origin_utc": ""})


def collect(annual_repo, app_repo, output, labels=None):
    """Export receipt-bound daily final components, keeping unobserved days pending.

    Supply an explicit EPEX label CSV with country,delivery_utc,target_price,
    target_available_utc. Optional storm_price/storm_origin_utc are never model
    inputs. Without it this command still creates a useful sealed forecast
    archive and a clear missing-label inventory.
    """
    roots = [Path(value).resolve() / "runs" for value in (annual_repo, app_repo) if value]
    destination = Path(output).resolve()
    all_rows, evidence, omitted = [], {}, []
    label_frame = None
    if labels:
        labels = Path(labels).resolve()
        label_frame = pd.read_csv(labels)
        require({"country", "delivery_utc", "target_price", "target_available_utc"} <= set(label_frame), "EPEX labels CSV schema missing")
        label_frame["delivery_utc"] = pd.to_datetime(label_frame.delivery_utc, utc=True)
        label_frame["target_available_utc"] = pd.to_datetime(label_frame.target_available_utc, utc=True)
        require(not label_frame.duplicated(["country", "delivery_utc"]).any(), "Duplicate EPEX labels")
        evidence[str(labels)] = sha(labels)
    seen = {}
    for root in roots:
        if not root.is_dir():
            continue
        for receipt_path in sorted(root.glob("**/daily_refit/receipt.json")):
            if destination == receipt_path.parent or destination in receipt_path.parents:
                continue
            receipt = read_json(receipt_path)
            require(receipt.get("protocol") == "nyx_daily_model_refit_v1" and receipt.get("status") == "MODELS_COMPLETE"
                and receipt.get("future_labels_used") is False, "Invalid final daily model receipt: " + str(receipt_path))
            hashes = receipt.get("artifact_sha256", {})
            evidence[str(receipt_path)] = sha(receipt_path)
            for zone in ZONES:
                path = safe_bound(receipt_path.parent, f"forecasts/{zone}.parquet", hashes)
                evidence[str(path)] = hashes[f"forecasts/{zone}.parquet"]
                frame = hourly(pd.read_parquet(path), zone)
                expected = pd.Timestamp(receipt["shared_cutoff_utc"]).tz_convert("UTC")
                require(pd.to_datetime(frame.forecast_origin_utc, utc=True).eq(expected).all(), "Final forecast origin differs from its sealed receipt")
                first = date.fromisoformat(receipt["first_delivery_day"])
                require(frame.index.equals(grid(first, first+timedelta(days=2))), "Incomplete sealed two-day forecast")
                identity = (zone, expected.isoformat())
                signature = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()
                if identity in seen:
                    require(seen[identity] == signature, "Conflicting sealed forecasts share the same country/origin; separate those runs explicitly")
                    continue
                seen[identity] = signature
                target = pd.Series(np.nan, index=frame.index)
                rows = canonical_rows(frame, zone, target, available_at=pd.Timestamp.now(tz="UTC"))
                rows["target_available_utc"] = pd.NaT
                if label_frame is not None:
                    subset = label_frame[label_frame.country.eq(zone)].set_index("delivery_utc").reindex(frame.index)
                    rows["target_price"] = subset.target_price.to_numpy()
                    rows["target_available_utc"] = subset.target_available_utc.to_numpy()
                    for name in ("storm_price", "storm_origin_utc"):
                        if name in subset:
                            rows[name] = subset[name].to_numpy()
                all_rows.append(rows)
    data = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=COLUMNS)
    observed, pending = [], []
    if len(data):
        for _, group in data.groupby(["country", "forecast_origin_utc", "horizon_day"], dropna=False):
            complete = group.target_price.notna().all() and group.target_available_utc.notna().all()
            (observed if complete else pending).append(group)
    observed_data = pd.concat(observed, ignore_index=True) if observed else pd.DataFrame(columns=COLUMNS)
    pending_data = pd.concat(pending, ignore_index=True) if pending else pd.DataFrame(columns=COLUMNS)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "sealed_forecasts.csv"
    data.to_csv(archive, index=False)
    observed_data.to_csv(destination / "OOF.csv", index=False)
    pending_data.to_csv(destination / "pending_forecasts.csv", index=False)
    manifest = {"protocol": "nyx_sealed_final_forecast_collection_v1", "provenance_status": "conditional", "scope": "final_experts",
        "forecast_source_status": "sealed_daily_final_forecasts; provider_vintages_not_certified_by_collection",
        "target_availability_policy": "explicit_label_CSV_timestamp; missing labels remain pending",
        "source_sha256": evidence, "oof_csv_sha256": sha(destination / "OOF.csv"),
        "sealed_forecast_rows": len(data), "scorable_rows": len(observed_data), "pending_rows": len(pending_data),
        "caveats": ["Collector verifies sealed model forecasts; it does not certify original provider publication vintages.",
            "Forecasts without a complete observed civil day remain pending and are excluded from OOF.csv.",
            "No J+2 forecast is reconstructed from J+1 history."]}
    atomic_json(destination / "OOF.manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "replay", "collect"))
    parser.add_argument("--annual-repo", type=Path, default=Path.home()/"NYX_CWE_CPU")
    parser.add_argument("--app-repo", type=Path, default=Path.home()/"chronos2_v1")
    parser.add_argument("--seed-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, default=0, help="0 = every fully supported conditional J+1 origin")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--implementation-root", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args(argv)
    require(1 <= args.threads <= 128, "Threads must be between 1 and 128")
    if args.command == "collect":
        result = collect(args.annual_repo, args.app_repo, args.output, args.labels)
    else:
        seed = seed_record(args.annual_repo, args.seed_run)
        if args.command == "inventory":
            result = coverage(seed)
            args.output.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output/"inventory.json", result)
        else:
            result = replay(seed, args.output, days=args.days, threads=args.threads,
                dry_run=args.dry_run, implementation_root=args.implementation_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print("NYX research stopped: " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(2)
