"""Refit every trainable NYX model for a new two-day operational forecast."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

PROTOCOL = "nyx_daily_model_refit_v1"
ZONES = ("FR", "DE", "BE", "NL")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "Daily refit output already exists: " + str(path))
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2,
                              allow_nan=False, default=str) + "\n", encoding="utf-8")


def _read(path, *, expected_sha256=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(),
            "Missing or invalid daily refit receipt: " + str(path))
    if expected_sha256 is None:
        require(path.stat().st_size <= 8_000_000,
                "Daily refit receipt exceeds the 8 MB metadata limit: " + str(path))
    else:
        # Detailed provenance can exceed the small-receipt limit. Only an
        # audit explicitly bound to the completed model inventory may do so.
        require(re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256))
                and sha(path) == expected_sha256,
                "Sealed daily refit audit checksum differs: " + str(path))
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if expected_sha256 is not None:
        require(sha(path) == expected_sha256,
                "Sealed daily refit audit changed during read: " + str(path))
    require(isinstance(value, dict), "Daily refit receipt must be an object")
    return value


def _checked_forecasts(forecasts, first_day, cutoff):
    from .nyx_saved_horizon_inference import grid
    current = grid(first_day)
    require(set(forecasts) == set(ZONES), "Four freshly fitted forecasts required")
    required = ("price_eur_mwh", "p_negative", "p_negative_raw", "reference_eur_mwh",
                "nyx__q10", "nyx__q50", "nyx__q90", "forecast_origin_utc", "horizon_day")
    for zone, frame in forecasts.items():
        require(isinstance(frame, pd.DataFrame) and frame.index.equals(current)
                and frame.columns.is_unique and set(required) <= set(frame),
                zone + ": complete J+1/J+2 forecast grid required")
        numeric = frame.loc[:, [name for name in required if name != "forecast_origin_utc"]]
        require(np.isfinite(numeric.to_numpy(dtype=float)).all(), zone + ": invalid refitted forecast")
        require(frame[["p_negative", "p_negative_raw"]].ge(0).all().all()
                and frame[["p_negative", "p_negative_raw"]].le(1).all().all(),
                zone + ": invalid refitted probabilities")
        origins = pd.to_datetime(frame.forecast_origin_utc, utc=True)
        require((origins == pd.Timestamp(cutoff).tz_convert("UTC")).all(),
                zone + ": refitted forecasts do not share the original cutoff")
        require(not any(name.lower() in {"actual", "target", "observed"} for name in frame),
                "Future observed labels cannot enter refitted forecasts")


def _check_manifest(root, hashes, label):
    root = Path(root).resolve()
    require(isinstance(hashes, dict) and hashes, label + ": missing fitted artifact hashes")
    for name, expected in hashes.items():
        require(isinstance(name, str) and "\\" not in name and not Path(name).is_absolute()
                and ".." not in Path(name).parts and re.fullmatch(r"[0-9a-f]{64}", str(expected)),
                label + ": invalid artifact identity")
        path = root / name
        require(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
                and sha(path) == expected, label + ": fitted artifact changed: " + name)


def _check_fit_bindings(directory, audit):
    _check_manifest(directory, audit["model_artifact_sha256"], "Final models")
    _check_manifest(directory, audit["audit_artifact_sha256"], "Final model audits")
    _check_manifest(Path(directory) / "reference", audit["reference"]["artifact_sha256"], "Reference models")
    training = audit.get("training_receipt")
    if training is not None:
        path = Path(training["path"])
        require(path.is_file() and not path.is_symlink() and sha(path) == training["sha256"],
                "Training receipt changed after fitting")
        _check_manifest(path.parent, training["artifact_sha256"], "Training matrices")


def _seal(directory, *, first_day, cutoff, seed_output, horizon_directory, forecasts, audit):
    directory = Path(directory)
    _check_fit_bindings(directory, audit)
    _checked_forecasts(forecasts, first_day, cutoff)
    for zone, frame in forecasts.items():
        path = directory / "forecasts" / (zone + ".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        require(not path.exists(), "Refitted forecast output already exists")
        frame.to_parquet(path)
    _write(directory / "model_audit.json", audit)
    files = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "Redirected refit model artifact")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = sha(path)
    require("receipt.json" not in files, "Daily refit receipt already exists")
    bindings = {name: sha(Path(horizon_directory) / name) for name in (
        "baseline_extension/receipt.json", "horizon_sources/receipt.json")}
    receipt = {"protocol": PROTOCOL, "status": "MODELS_COMPLETE",
        "first_delivery_day": str(first_day), "forecast_days": 2,
        "shared_cutoff_utc": pd.Timestamp(cutoff).tz_convert("UTC").isoformat(),
        "seed_report_receipt_sha256": sha(Path(seed_output) / "receipt.json"),
        "source_receipts_sha256": bindings, "artifact_sha256": files,
        "future_labels_used": False, "new_forecast_refit_required": True,
        "chronos_foundation_weights_refitted": False,
        "final_models_fitted": audit["final_models_fitted"],
        "final_model_refit_attempts": audit["final_model_refit_attempts"],
        "reference_current_fit_counts": audit["reference"]["current_fit_counts"]}
    _write(directory / "receipt.json", receipt)
    return {"path": str(directory / "receipt.json"), "sha256": sha(directory / "receipt.json")}


def produce_refit_horizon(repo, seed_bundle, seed_output, first_day, forecast_issued_at_utc,
                         output_directory, context=None, *, progress=None):
    """Build the current CPU baseline, then fit fresh references and experts.

    The established producer is deliberately unchanged so its verified rolling
    histories remain compatible. Its quick saved-model inference is discarded;
    failures of the subsequent fresh fits never return that intermediate result.
    """
    from .nyx_saved_horizon_inference import produce_horizon
    output_directory = Path(output_directory)
    directory = output_directory / "daily_refit"
    require(not directory.exists(), "Daily refit directory already exists")
    _write(directory / "request.json", {"protocol": PROTOCOL,
        "first_delivery_day": str(first_day), "cutoff": str(forecast_issued_at_utc),
        "refit_all_trainable_nyx_models": True})
    source_seed_hash = sha(Path(seed_output) / "receipt.json")
    horizon = produce_horizon(repo, seed_bundle, seed_output, first_day,
        forecast_issued_at_utc, output_directory, context=context)
    return _fit_horizon_models(repo, seed_bundle, seed_output, horizon,
        output_directory, source_seed_hash=source_seed_hash, context=context, progress=progress)


def _fit_horizon_models(repo, seed_bundle, seed_output, horizon, output_directory,
                        *, source_seed_hash, context=None, progress=None):
    from .nyx_daily_refit_history import prepare_training_history, publish_training_state
    from .nyx_daily_refit_reference import fit_daily_reference
    from .nyx_daily_refit_experts import fit_daily_experts
    output_directory = Path(output_directory)
    directory = output_directory / "daily_refit"
    first_day = horizon["first_delivery_day"]
    if progress:
        progress("training_history", status="running")
    history = prepare_training_history(repo, seed_bundle, seed_output, horizon,
                                       output_directory, progress=progress)
    if progress:
        progress("training_history", status="complete")
    references, distributions, reference_audit, oof = fit_daily_reference(
        first_day, horizon["cutoff"], history["built"], history["quantiles"],
        history["actual"], history["prior_oof"], directory, progress=progress)
    matrices = history["built"].features
    index = next(iter(next(iter(matrices.values())).values())).index
    forecasts, experts_audit = fit_daily_experts(first_day, horizon["cutoff"], history["built"],
        {zone: history["quantiles"][zone].reindex(index) for zone in ZONES},
        {zone: history["actual"][zone].reindex(index) for zone in ZONES},
        references, distributions, directory, progress=progress)
    require(sha(Path(seed_output) / "receipt.json") == source_seed_hash,
            "Completed seed changed during daily refitting")
    _checked_forecasts(forecasts, first_day, horizon["cutoff"])
    require(experts_audit.get("future_labels_used") is False
            and reference_audit.get("future_labels_used") is False,
            "Fresh fits must explicitly exclude future labels")
    require(experts_audit.get("final_model_refit_attempts") == 7
            and reference_audit.get("current_fit_counts") == {"hgb": 12, "test2_pairs": 2},
            "Every current NYX expert and reference must be freshly fitted")
    state = publish_training_state(history, reference_oof=oof)
    old_audit = horizon["model_audit"]
    audit = {**experts_audit, "protocol": PROTOCOL,
        "frequency": "every_new_forecast", "daily_refit_completed": True,
        "refit_all_trainable_nyx_models": True, "reference": reference_audit,
        "first_delivery_day": str(first_day), "forecast_days": 2,
        "shared_cutoff_utc": pd.Timestamp(horizon["cutoff"]).isoformat(),
        "future_labels_used": False, "full_835_day_replay": False,
        "qualified_for_production": False, "training_history": history["evidence"],
        "training_receipt": history.get("training_receipt"),
        "training_state": state, "bounded_baseline": old_audit.get("bounded_baseline"),
        "rolling_daily_state": old_audit.get("rolling_daily_state"),
        "new_operational_context": context,
        "seed_report_receipt_sha256": source_seed_hash,
        "seed_inference_used_for_final_forecast": False,
        "chronos_foundation_weights_refitted": False}
    if horizon.get("refit_continuation"):
        audit["refit_continuation"] = horizon["refit_continuation"]
    sealed = _seal(directory, first_day=first_day, cutoff=horizon["cutoff"], seed_output=seed_output,
                   horizon_directory=output_directory, forecasts=forecasts, audit=audit)
    audit["daily_refit_receipt"] = sealed
    return {**horizon, "forecasts": forecasts, "model_audit": audit}


def _copy_completed_inputs(source, destination):
    """Copy only receipt-bound CPU/source artifacts, checking both copies."""
    from .nyx_saved_horizon_resume import _normalized_hashes
    source, destination = Path(source), Path(destination)
    require(not destination.exists(), "Continuation output already exists")
    receipts, files = {}, {}
    for relative, field, prefix in (
        ("baseline_extension/receipt.json", "artifact_sha256", "baseline_extension/"),
        ("horizon_sources/receipt.json", "source_artifact_sha256", ""),
    ):
        path = source / relative
        document = _read(path)
        receipts[relative] = sha(path)
        for name, expected in _normalized_hashes(document.get(field), relative).items():
            key = prefix + name
            require(key not in files, "Repeated continuation input")
            files[key] = expected
    files.update(receipts)
    destination.mkdir(parents=True, exist_ok=False)
    for relative, expected in files.items():
        path = source / relative
        require(path.resolve().is_relative_to(source.resolve()) and path.is_file()
                and not any(item.is_symlink() or getattr(item, "is_junction", lambda: False)()
                    for item in (path, *path.parents)) and sha(path) == expected,
                "Completed CPU/source input changed: " + relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        require(sha(target) == expected and sha(path) == expected,
                "Completed CPU/source input changed during copy: " + relative)
    for relative, expected in files.items():
        require(sha(source / relative) == expected and sha(destination / relative) == expected,
                "Completed CPU/source inventory changed during continuation: " + relative)
    return receipts


def continue_refit_horizon(repo, seed_bundle, seed_output, failed_report_directory,
                           output_directory, *, owned_report_roots=(), progress=None):
    """Keep verified CPU/source inputs and refit every model in a new output.

    This is a continuation of an incomplete forecast at its original cutoff.
    No partial estimator or training matrix from the failed attempt is reused.
    """
    from .nyx_saved_horizon_resume import resume_horizon
    failed = Path(failed_report_directory)
    extension = failed / "forecast_extension"
    destination = Path(output_directory).resolve()
    for protected in (failed, seed_bundle, seed_output):
        root = Path(protected).resolve()
        require(not (destination.is_relative_to(root) or root.is_relative_to(destination)),
                "Continuation output must remain separate from the failed run and seed")
    require(not (extension / "daily_refit/receipt.json").exists(),
            "The refit is already sealed: use report-only resume")
    source_seed_hash = sha(Path(seed_output) / "receipt.json")
    horizon = resume_horizon(repo, seed_bundle, seed_output, failed,
                             owned_report_roots=owned_report_roots)
    request_path = extension / "daily_refit/request.json"
    request_hash = sha(request_path)
    request = _read(request_path)
    require(request.get("protocol") == PROTOCOL
            and request.get("refit_all_trainable_nyx_models") is True
            and request.get("first_delivery_day") == horizon["first_delivery_day"]
            and pd.Timestamp(request.get("cutoff")) == pd.Timestamp(horizon["cutoff"]),
            "Incomplete refit request differs from the verified CPU/source calculation")
    inputs = _copy_completed_inputs(extension, output_directory)
    verified = horizon.get("resume_audit", {})
    require(inputs == {
        "baseline_extension/receipt.json": verified.get("baseline_receipt_sha256"),
        "horizon_sources/receipt.json": verified.get("source_receipt_sha256")},
        "CPU/source receipts changed after the verified resume read")
    require(sha(request_path) == request_hash,
            "Incomplete refit request changed during continuation")
    continuation = {"source_report_directory": str(failed.resolve()),
        "request_sha256": request_hash, "source_receipts_sha256": inputs,
        "original_first_delivery_day": horizon["first_delivery_day"],
        "original_cutoff_utc": pd.Timestamp(horizon["cutoff"]).isoformat(),
        "cpu_baseline_recomputed": False, "model_source_fetches": 0,
        "partial_estimator_weights_reused": False, "all_model_recipes_restarted": True}
    horizon["refit_continuation"] = continuation
    _write(Path(output_directory) / "daily_refit/request.json", {
        "protocol": PROTOCOL, "first_delivery_day": horizon["first_delivery_day"],
        "cutoff": pd.Timestamp(horizon["cutoff"]).isoformat(),
        "refit_all_trainable_nyx_models": True, "continuation": continuation})
    result = _fit_horizon_models(repo, seed_bundle, seed_output, horizon, output_directory,
        source_seed_hash=source_seed_hash, context=horizon.get("context"), progress=progress)
    result["resume_audit"] = {**horizon.get("resume_audit", {}),
        "daily_refit_forecasts_reused": False,
        "final_expert_fits": result["model_audit"]["final_models_fitted"],
        "final_expert_refit_attempts": result["model_audit"]["final_model_refit_attempts"],
        "reference_model_fits": 14, "refit_continuation": continuation}
    return result


def resume_refit_horizon(repo, seed_bundle, seed_output, failed_report_directory,
                        *, owned_report_roots=()):
    """Reload the sealed fresh forecast for rendering, without repeating fits."""
    from .nyx_saved_horizon_resume import resume_horizon
    failed = Path(failed_report_directory)
    horizon = resume_horizon(repo, seed_bundle, seed_output, failed,
                             owned_report_roots=owned_report_roots)
    extension = failed / "forecast_extension"
    directory = extension / "daily_refit"
    receipt_path = directory / "receipt.json"
    require(directory.is_dir() and receipt_path.is_file(),
            "Le réajustement des modèles n'a pas terminé : lancer une nouvelle prévision, pas une reprise HTML.")
    receipt = _read(receipt_path)
    receipt_hash = sha(receipt_path)
    require(receipt.get("protocol") == PROTOCOL and receipt.get("status") == "MODELS_COMPLETE"
            and receipt.get("future_labels_used") is False and receipt.get("forecast_days") == 2
            and receipt.get("first_delivery_day") == horizon["first_delivery_day"]
            and pd.Timestamp(receipt.get("shared_cutoff_utc")) == pd.Timestamp(horizon["cutoff"])
            and receipt.get("seed_report_receipt_sha256") == sha(Path(seed_output) / "receipt.json"),
            "Daily refit identity differs from the resumed forecast")
    for name in ("baseline_extension/receipt.json", "horizon_sources/receipt.json"):
        require(receipt.get("source_receipts_sha256", {}).get(name) == sha(extension / name),
                "Daily refit source receipt changed")
    hashes = receipt.get("artifact_sha256")
    require(isinstance(hashes, dict) and hashes, "Daily fitted model inventory missing")
    require("model_audit.json" in hashes, "Daily model audit is missing from the sealed inventory")
    actual_files = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
                    if path.is_file() and path != receipt_path}
    require(set(hashes) == actual_files, "Daily fitted model inventory differs")
    for name, expected in hashes.items():
        require(isinstance(name, str) and "\\" not in name and not name.startswith("/")
                and ".." not in Path(name).parts and re.fullmatch(r"[0-9a-f]{64}", str(expected)),
                "Invalid daily fitted model artifact name or hash")
        path = directory / name
        require(path.resolve().is_relative_to(directory.resolve()) and not path.is_symlink()
                and path.is_file() and sha(path) == expected, "Daily refitted artifact changed: " + name)
    forecasts = {zone: pd.read_parquet(directory / "forecasts" / (zone + ".parquet")) for zone in ZONES}
    _checked_forecasts(forecasts, horizon["first_delivery_day"], horizon["cutoff"])
    audit = _read(directory / "model_audit.json", expected_sha256=hashes["model_audit.json"])
    require(audit.get("daily_refit_completed") is True
            and audit.get("future_labels_used") is False and audit.get("final_model_refit_attempts") == 7,
            "The resumed forecast was not fully refitted")
    _check_fit_bindings(directory, audit)
    require(sha(receipt_path) == receipt_hash, "Daily refit receipt changed during read")
    resume = {**horizon.get("resume_audit", {}), "daily_refit_forecasts_reused": True,
              "final_expert_fits": 0, "reference_model_fits": 0,
              "daily_refit_receipt_sha256": receipt_hash}
    audit.update(report_resume=resume,
                 daily_refit_receipt={"path": str(receipt_path), "sha256": receipt_hash})
    return {**horizon, "forecasts": forecasts, "model_audit": audit, "resume_audit": resume}
