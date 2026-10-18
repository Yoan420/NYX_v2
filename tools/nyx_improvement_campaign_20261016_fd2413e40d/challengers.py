"""Isolated 365-day pooled price challengers on already sealed NYX J+1 origins.

This program never calls Chronos or refits the canonical reference/price experts.
Historical features inherit the donor replay's CONDITIONAL vintage limitations.
The linear challenger is Ridge with LEAR-like explanatory inputs, not a canonical
LEAR implementation. No model is activated in production.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from importlib.metadata import version
import importlib
import json
from pathlib import Path
import platform
import shutil
import sys
from uuid import uuid4

import numpy as np
import pandas as pd

import replay as r

PROTOCOL = "nyx_isolated_price_challengers_v1"
EXTRA_COLUMNS = ("challenger_ridge", "challenger_rmse", "challenger_mae",
                 "challenger_huber", "challenger_regime")
MODEL_FILES = ("ridge.joblib", "rmse.cbm", "mae.cbm", "huber.cbm", "regime.cbm")
KEYS = ["country", "delivery_utc", "forecast_origin_utc", "horizon_day"]


def runtime_versions():
    return {"python": platform.python_version(), **{name: version(name) for name in
            ("numpy", "pandas", "catboost", "scikit-learn", "joblib")}}


def source_module():
    return importlib.import_module("chronos2_hourly.nyx_pooled_cpu_price_model")


def code_hashes():
    hashes = {"challengers.py": r.sha(__file__), "replay.py": r.sha(r.__file__),
              "nyx_pooled_cpu_price_model": r.sha(source_module().__file__)}
    # Feature reconstruction has transitive helpers beyond the canonical replay's
    # minimal receipt. Freeze the complete Python package for this new campaign.
    package = importlib.import_module("chronos2_hourly")
    for number, directory in enumerate(package.__path__):
        root = Path(directory).resolve()
        for path in sorted(root.rglob("*.py")):
            r.require(not path.is_symlink() and path.resolve().is_relative_to(root),
                      "Symlinked annual source is not supported")
            key = f"chronos2_hourly[{number}]/" + path.relative_to(root).as_posix()
            hashes[key] = r.sha(path)
    return hashes


def recipe(iterations=300):
    r.require(type(iterations) is int and 1 <= iterations <= 2000,
              "Iterations must be an integer from 1 to 2000")
    return {"training_days": 365, "feature_family": "cwe_residual_2000",
        "target_mode": "residual", "ridge_alpha": 10.,
        "ridge_imputation": "training medians + missing indicators; empty columns zero",
        "ridge_identity": "regularized linear challenger; not canonical LEAR",
        "catboost_iterations": iterations, "catboost_depth": 6,
        "catboost_learning_rate": .035, "catboost_l2_leaf_reg": 20.,
        "huber_delta_eur_mwh": 20., "random_seed": 20261016,
        "regime_weight": "1+2*(abs(past residual)>=past pooled 95th percentile)",
        "raw_labels": True, "prediction_clipped": False,
        "early_stopping": False, "Storm_used_as_input": False,
        "current_delivery_targets_used_for_training": False,
        "j2_reconstruction_permitted": False}


def validate_original_plan(base, seed, inputs):
    """Bind the donor and installed canonical recipe before reusing any seal."""
    base = Path(base).resolve()
    plan_path = base / "replay_plan.json"
    plan = r.read_json(plan_path)
    r.require(plan.get("protocol") == r.PROTOCOL and plan.get("training_days") == 365
              and plan.get("scope") == "final_experts"
              and plan.get("j2_relabeling_permitted") is False,
              "Unsupported original replay protocol")
    r.require(Path(plan["seed_run"]).resolve() == seed["run_path"].parent.resolve(),
              "Original replay and requested donor differ")
    r.require(plan.get("source_sha256") == inputs["consumed"],
              "Original replay donor source inventory changed")
    r.verify_inputs(plan["source_sha256"])
    hashes = plan.get("code_sha256", {})
    required = {"replay.py", "nyx_daily_refit_reference", "nyx_pooled_cpu_price_model",
                "nyx_annual_cpu_live", "nyx_annual_cpu_bundle_builder"}
    r.require(required <= set(hashes), "Original replay lacks canonical code receipts")
    for name, expected in hashes.items():
        path = r.__file__ if name == "replay.py" else importlib.import_module(
            "chronos2_hourly." + name).__file__
        r.require(r.sha(path) == expected,
                  "Original canonical replay code changed: " + name)
    r.require(plan.get("runtime_versions") == runtime_versions(),
              "Original replay dependencies differ; restore its pinned environment")
    days = plan.get("days")
    r.require(isinstance(days, list) and days and len(set(days)) == len(days)
              and days == sorted(days), "Invalid original replay origin list")
    for day in days:
        r.require(date.fromisoformat(day).isoformat() == day, "Invalid origin day")
    available = set(r.coverage(seed)["conditional_j1_days"])
    r.require(set(days) <= available, "Original replay origins exceed donor coverage")
    return plan, plan_path


def validated_base_origin(base, day, plan_digest):
    """Missing receipt means pending; an existing invalid receipt always fails."""
    destination = Path(base) / "origins" / day
    path = destination / "receipt.json"
    if not path.is_file():
        return None
    receipt = r.read_json(path)
    r.require(receipt.get("protocol") == r.PROTOCOL and receipt.get("day") == day
              and receipt.get("plan_sha256") == plan_digest
              and receipt.get("predictions_sealed_before_scoring") is True,
              "Original origin has an invalid prediction seal: " + day)
    inventory = receipt.get("artifact_sha256", {})
    mandatory = {"model_audit.json", *[z + ".parquet" for z in r.ZONES],
                 *["price_models/" + f + ".cbm" for f in r.FAMILIES]}
    r.require(mandatory <= set(inventory), "Original seal lacks models or predictions")
    for relative in inventory:
        r.safe_bound(destination, relative, inventory)
    audit = r.read_json(destination / "model_audit.json")
    r.require(audit.get("training_days") == 365,
              "Original origin training window differs")
    reference = audit.get("reference", {})
    r.require(reference.get("current_fit_counts") == {"hgb": 12, "test2_pairs": 2}
              and reference.get("future_labels_used") is False
              and reference.get("storm_used_as_input") is False,
              "Original reference models lack causal fit receipts")
    reference_models = [name for name in inventory
                        if name.startswith("models/" + day + "/") and name.endswith(".joblib")]
    r.require(len(reference_models) == 14, "Original seal lacks the 14 daily reference models")
    for name, expected in reference.get("artifact_sha256", {}).items():
        r.require(inventory.get(name) == expected, "Original reference artifact receipt differs")
    for family in r.FAMILIES:
        expert = audit.get("price_experts", {}).get(family, {})
        r.require(expert.get("training_days") == 365
                  and expert.get("training_labels_before_origin") is True
                  and expert.get("Storm_used_as_input") is False,
                  "Original price expert lacks causal recipe audit")
        model_digest = expert.get("model_sha256") or expert.get("model", {}).get("sha256")
        r.require(model_digest == inventory["price_models/" + family + ".cbm"],
                  "Original expert model receipt differs")
    frames = {}
    for zone in r.ZONES:
        frames[zone] = pd.read_parquet(destination / (zone + ".parquet"))
        r.validate_prediction(frames[zone], zone, day)
    return {"path": path, "sha256": r.sha(path), "frames": frames}


def build_training(inputs, day, threads):
    """The shared builder enforces the exact civil grid and forbidden columns."""
    module = source_module()
    first = date.fromisoformat(day)
    current = r.grid(first, first + timedelta(days=1))
    features = inputs["built"].features["cwe_residual_2000"]
    index = features[r.ZONES[0]].index
    bounded = {z: inputs["actual"][z].where(inputs["actual"][z].index < current[0]).reindex(index)
               for z in r.ZONES}
    nyx = {z: inputs["quantiles"][z].q50.reindex(index) for z in r.ZONES}
    config = module.PooledConfig(target_mode="residual", iterations=300)
    train_x, target, current_x, base, audit = module.build_inputs(
        features, bounded, nyx, origin_day=day, stop_day=str(first + timedelta(days=1)),
        initial_training_day=str(first - timedelta(days=365)), config=config,
        thread_count=threads)
    train_hours = r.grid(first - timedelta(days=365), first)
    expected = pd.MultiIndex.from_product([train_hours, r.ZONES], names=["timestamp_utc", "zone"])
    r.require(train_x.index.equals(expected) and target.index.equals(expected),
              "Challengers require exactly 365 prior civil days in all four countries")
    r.require(np.isfinite(target.to_numpy(float)).all()
              and audit.get("training_days") == 365
              and audit.get("training_labels_before_origin") is True
              and audit.get("Storm_used_as_input") is False,
              "Training target embargo or recipe invalid")
    r.require((train_x.index.get_level_values("timestamp_utc") < current[0]).all(),
              "Current or future delivery target entered training")
    return train_x, target, current_x, base, audit


def fit_challengers(inputs, day, destination, threads=8, iterations=300, progress=None):
    """Fit a small fixed research menu; future EPEX/Storm are never opened here."""
    from catboost import CatBoostRegressor
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = recipe(iterations)
    destination = Path(destination)
    x, y, current, baseline, training_audit = build_training(inputs, day, threads)
    models = destination / "models"
    models.mkdir(parents=True, exist_ok=True)
    points, audits = {}, {}
    linear = make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                         keep_empty_features=True),
                           StandardScaler(), Ridge(alpha=cfg["ridge_alpha"], solver="lsqr"))
    linear.fit(x, y)
    points["challenger_ridge"] = baseline.to_numpy(float) + linear.predict(current)
    joblib.dump(linear, models / "ridge.joblib")
    audits["challenger_ridge"] = {"identity": cfg["ridge_identity"], "alpha": cfg["ridge_alpha"],
                                  "imputer_fitted_on_training_only": True}
    if progress:
        progress("challenger_model", model="ridge", status="complete")
    cutoff = float(np.quantile(np.abs(y.to_numpy(float)), .95))
    weights = 1. + 2. * (np.abs(y.to_numpy(float)) >= cutoff)
    for name, loss in (("rmse", "RMSE"), ("mae", "MAE"),
                       ("huber", "Huber:delta=20"), ("regime", "RMSE")):
        params = {"loss_function": loss, "iterations": iterations, "depth": 6,
                  "learning_rate": .035, "l2_leaf_reg": 20., "random_seed": 20261016,
                  "task_type": "CPU", "thread_count": threads, "verbose": False,
                  "allow_writing_files": False, "nan_mode": "Min", "has_time": True,
                  "boosting_type": "Plain", "bootstrap_type": "No",
                  "use_best_model": False}
        estimator = CatBoostRegressor(**params)
        estimator.fit(x, y, sample_weight=weights if name == "regime" else None)
        prediction = estimator.predict(current, task_type="CPU", thread_count=threads)
        points["challenger_" + name] = baseline.to_numpy(float) + np.asarray(prediction, float)
        estimator.save_model(str(models / (name + ".cbm")))
        audits["challenger_" + name] = {"parameters": params,
            "tree_count": int(estimator.tree_count_),
            "training_only_tail_threshold": cutoff if name == "regime" else None,
            "sample_weighting": cfg["regime_weight"] if name == "regime" else "uniform"}
        if progress:
            progress("challenger_model", model=name, status="complete")
    result = pd.DataFrame(points, index=current.index)
    r.require(list(result) == list(EXTRA_COLUMNS) and np.isfinite(result.to_numpy(float)).all(),
              "Challenger predictions are incomplete or nonfinite")
    return {z: result.xs(z, level="zone").copy() for z in r.ZONES}, {
        "protocol": PROTOCOL, "recipe": cfg, "training": training_audit,
        "models": audits, "future_labels_used": False, "foundation_calls": 0,
        "training_label_availability": "day-ahead prices for prior delivery days; original revision vintages conditional"}


def validate_extra(frame, day):
    first = date.fromisoformat(day)
    r.require(frame.index.equals(r.grid(first, first + timedelta(days=1)))
              and list(frame) == list(EXTRA_COLUMNS)
              and np.isfinite(frame.to_numpy(float)).all(),
              "Invalid challenger day or ordered prediction schema")


def run(seed, base_replay, output, *, threads=8, iterations=300, max_days=0,
        inputs=None, predictor=fit_challengers, implementation_root=None):
    r.require(type(threads) is int and 1 <= threads <= 128, "Threads must be 1..128")
    r.require(type(max_days) is int and max_days >= 0, "max-days must be nonnegative")
    cfg = recipe(iterations)
    base, output = Path(base_replay).resolve(), Path(output).resolve()
    for protected in (base, seed["bundle"], seed["output"]):
        protected = Path(protected).resolve()
        r.require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                  "Challenger output must be separate from canonical replay and donor")
    r.configure(seed["repo"], implementation_root)
    inputs = r.load_research_inputs(seed) if inputs is None else inputs
    original, original_path = validate_original_plan(base, seed, inputs)
    digest = r.sha(original_path)
    sealed, pending = {}, []
    for day in original["days"]:
        evidence = validated_base_origin(base, day, digest)
        if evidence is None:
            pending.append({"delivery_day": day, "reason": "canonical replay not sealed"})
        else:
            sealed[day] = evidence
    choices = sorted(sealed)
    if max_days:
        choices = choices[-max_days:]
    r.require(choices, "No sealed canonical origin yet; resume canonical Replay before Challengers")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "challenger_plan.json"
    frozen = r.read_json(plan_path) if plan_path.exists() else None
    if frozen:
        # Newly finished canonical days do not mutate the original comparison.
        selected = frozen["days"]
        r.require(set(selected) <= set(sealed), "A previously selected canonical origin is missing")
        r.require(frozen.get("max_days") == max_days, "max-days changed; use a new output directory")
    else:
        selected = choices
    selected_evidence = {str(sealed[day]["path"]): sealed[day]["sha256"] for day in selected}
    source_hashes = {**inputs["consumed"], str(original_path): digest, **selected_evidence}
    plan = {"protocol": PROTOCOL, "scope": "final_experts", "days": selected,
        "max_days": max_days, "seed_run": str(seed["run_path"].parent),
        "base_replay": str(base), "base_replay_plan_sha256": digest,
        "source_sha256": source_hashes, "code_sha256": code_hashes(),
        "runtime_versions": runtime_versions(), "threads": threads, "recipe": cfg,
        "extra_components": list(EXTRA_COLUMNS), "provenance_status": "conditional",
        "production_qualification": False, "historical_provider_vintages_certified": False,
        "foundation_calls": 0, "production_modified": False, "j2_relabeling_permitted": False}
    if frozen:
        r.require(frozen == plan, "Challenger plan, inputs, code, recipe or dependencies changed; use a new output directory")
    else:
        r.atomic_json(plan_path, plan)
    plan_digest = r.sha(plan_path)
    pd.DataFrame(pending, columns=["delivery_day", "reason"]).to_csv(output / "pending_origins.csv", index=False)
    for position, day in enumerate(selected, 1):
        destination = output / "origins" / day
        receipt_path = destination / "receipt.json"
        r.verify_inputs(source_hashes)
        validated_base_origin(base, day, digest)
        if receipt_path.exists():
            receipt = r.read_json(receipt_path)
            r.require(receipt.get("protocol") == PROTOCOL and receipt.get("day") == day
                      and receipt.get("plan_sha256") == plan_digest
                      and receipt.get("predictions_sealed_before_scoring") is True,
                      "Invalid resumed challenger seal")
            inventory = receipt.get("artifact_sha256", {})
            mandatory = {"model_audit.json", *[z + ".parquet" for z in r.ZONES],
                         *["models/" + name for name in MODEL_FILES]}
            r.require(mandatory <= set(inventory), "Challenger seal lacks models or predictions")
            for relative in inventory:
                r.safe_bound(destination, relative, inventory)
            for zone in r.ZONES:
                validate_extra(pd.read_parquet(destination / (zone + ".parquet")), day)
        else:
            if destination.exists():
                abandoned = output / "interrupted" / (day + "_" + uuid4().hex)
                abandoned.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(abandoned))
            destination.mkdir(parents=True)
            def progress(stage, **details):
                print(json.dumps({"stage": stage, "day": day, "completed_days": position-1,
                                  "total_days": len(selected), **details}), flush=True)
            frames, audit = predictor(inputs, day, destination, threads=threads,
                                      iterations=iterations, progress=progress)
            r.require(set(frames) == set(r.ZONES) and audit.get("future_labels_used") is False,
                      "Challenger fit lacks four countries or causal audit")
            for zone, frame in frames.items():
                validate_extra(frame, day)
                frame.to_parquet(destination / (zone + ".parquet"))
            r.atomic_json(destination / "model_audit.json", audit)
            r.verify_inputs(source_hashes)
            r.require(code_hashes() == plan["code_sha256"], "Challenger code changed during fit")
            inventory = {p.relative_to(destination).as_posix(): r.sha(p)
                         for p in destination.rglob("*") if p.is_file()}
            r.require({"models/" + n for n in MODEL_FILES} <= set(inventory),
                      "Fitted challengers were not saved")
            r.atomic_json(receipt_path, {"protocol": PROTOCOL, "day": day,
                "plan_sha256": plan_digest, "predictions_sealed_before_scoring": True,
                "artifact_sha256": inventory})
        r.atomic_json(output / "status.json", {"state": "PREDICTING", "completed": position,
                      "total": len(selected), "day": day})
    # Evaluation targets are accessed only after ALL selected challenger predictions seal.
    r.verify_inputs(source_hashes)
    rows = []
    for day in selected:
        evidence = validated_base_origin(base, day, digest)
        for zone in r.ZONES:
            frame = evidence["frames"][zone]
            target = inputs["actual"][zone].reindex(frame.index)
            r.require(np.isfinite(target.to_numpy(float)).all(), "Missing EPEX evaluation targets")
            part = r.canonical_rows(frame, zone, target, availability_policy="assumed_conservative")
            extras = pd.read_parquet(output / "origins" / day / (zone + ".parquet"))
            for name in EXTRA_COLUMNS:
                part[name] = extras[name].to_numpy(float)
            rows.append(part)
    result = pd.concat(rows, ignore_index=True).sort_values(KEYS)
    r.require(not result.duplicated(KEYS).any(), "Duplicate final challenger forecast keys")
    csv = output / "OOF.csv"
    result.to_csv(csv, index=False, float_format="%.17g")
    manifest = {**plan, "oof_csv_sha256": r.sha(csv), "replay_days": len(selected),
        "base_pending_days": len(pending), "scorable_rows": len(result),
        "forecast_source_status": "sealed canonical replay + isolated newly fitted challengers",
        "target_availability_policy": "assumed_conservative_delivery_day_plus_1_at_08_Paris; original provider revision not certified",
        "caveats": ["Only actual sealed J+1 origins are included; no synthetic J+2 history.",
            "Current donor revisions and reconstructed older auxiliary rows make this conditional research.",
            "Ridge is not a canonical LEAR benchmark. Fixed research recipes do not claim optimal parameters.",
            "A few supported days cannot qualify annual thresholds or choose a production model.",
            "No training, feature or model input uses Storm.",
            "New canonical seals after this plan was created require a new output directory."]}
    r.atomic_json(output / "OOF.manifest.json", manifest)
    r.verify_inputs(source_hashes)
    r.require(code_hashes() == plan["code_sha256"], "Challenger code changed during evaluation")
    result_status = {"state": "COMPLETE", "completed": len(selected), "total": len(selected),
        "csv": str(csv), "manifest": str(output / "OOF.manifest.json"),
        "pending_canonical_days": len(pending), "extra_components": list(EXTRA_COLUMNS),
        "production_qualification": False}
    r.atomic_json(output / "status.json", result_status)
    return result_status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-repo", type=Path, required=True)
    parser.add_argument("--seed-run", type=Path)
    parser.add_argument("--base-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--implementation-root", type=Path,
                        default=Path(r.__file__).parent / "annual_extension")
    args = parser.parse_args(argv)
    seed_path = args.seed_run
    if seed_path is None:
        seed_path = r.read_json(args.base_replay / "replay_plan.json")["seed_run"]
    seed = r.seed_record(args.annual_repo, seed_path)
    result = run(seed, args.base_replay, args.output, threads=args.threads,
                 iterations=args.iterations, max_days=args.max_days,
                 implementation_root=args.implementation_root)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print("[NYX challengers] " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(2)
