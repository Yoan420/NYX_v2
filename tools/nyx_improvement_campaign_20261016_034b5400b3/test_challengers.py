"""Causality, seal reuse and finite price-challenger checks."""
from datetime import date, timedelta
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import challengers as c
import replay as r


def load_pooled_source():
    candidates = [Path(__file__).parents[1] / "nyx_prod_reports/interval_research/chronos2_hourly/nyx_pooled_cpu_price_model.py"]
    try:
        import chronos2_hourly.nyx_pooled_cpu_price_model as loaded
        return loaded
    except ImportError:
        pass
    for path in candidates:
        if path.is_file():
            # Building matrices does not invoke CatBoost; avoid imposing a model
            # runtime just to verify the canonical chronology and schema checks.
            name = "_nyx_challenger_test_pooled"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            fake = types.ModuleType("catboost")
            fake.CatBoostRegressor = object
            with patch.dict(sys.modules, {name: module, "catboost": fake}):
                spec.loader.exec_module(module)
            if importlib.util.find_spec("catboost") is None:
                module._versions = lambda: {"test_only_builder": "CatBoost not invoked"}
            return module
    raise unittest.SkipTest("Canonical pooled builder not available in this source checkout")


def inputs_fixture(first=date(2026, 10, 10)):
    index = r.grid(first - timedelta(days=365), first + timedelta(days=1))
    signal = np.sin(np.arange(len(index)) / 23.)
    features = {z: pd.DataFrame({"signal": signal, "missing": np.nan,
                                 "missing__available": 0.}, index=index) for z in r.ZONES}
    actual = {z: pd.Series(55. + j + signal * 8., index=index, name="price")
              for j, z in enumerate(r.ZONES)}
    quantiles = {z: pd.DataFrame({"q10": 40., "q50": 50., "q90": 60.}, index=index)
                 for z in r.ZONES}
    return {"built": types.SimpleNamespace(features={"cwe_residual_2000": features}),
            "actual": actual, "quantiles": quantiles}


def write_reference_block(directory, first, stop, purpose="current", cutoff=None):
    """Producer-shaped JSON/model-byte fixture; no estimator deserialization."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    first_training, last_training = str(first-timedelta(days=365)), str(first-timedelta(days=1))
    records = []
    for zone in r.ZONES:
        for variant in c.REFERENCE_VARIANTS:
            name = f"{zone}_{variant}.joblib"
            (directory / name).write_bytes(b"reference-model")
            records.append({"kind": "hgb", "zone": zone, "variant": variant, "model": name,
                "audit": {"zone": zone, "variant": variant, "origin_day": str(first),
                    "stop_day_exclusive": str(stop), "training_first_day": first_training,
                    "training_last_day": last_training, "training_days": 365,
                    "training_labels_before_origin": True, "Storm_used_as_input": False}})
    for pair in c.REFERENCE_PAIRS:
        name = "Test2_" + "_".join(pair) + ".joblib"
        (directory / name).write_bytes(b"reference-model")
        records.append({"kind": "test2", "pair": list(pair), "model": name,
            "audit": {"pair": list(pair), "origin_day": str(first),
                "train_first_day": first_training, "train_last_day": last_training,
                "train_unique_days": 365, "future_labels_used": False}})
    r.atomic_json(directory / "fit_audit.json", {"purpose": purpose, "origin_day": str(first),
        "stop_day_exclusive": str(stop), "training_first_day": first_training,
        "training_last_day": last_training, "training_days": 365,
        "shared_forecast_cutoff_utc": cutoff.isoformat() if cutoff is not None else None,
        "future_labels_used": False, "prediction_labels_masked": True, "models": records})


def write_reference_fixture(destination, day, bridge=False):
    first = date.fromisoformat(day)
    cutoff = pd.Timestamp(f"{first-timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC")
    reference = destination / "reference"
    write_reference_block(reference / "models" / day, first, first+timedelta(days=2), cutoff=cutoff)
    if bridge:
        write_reference_block(reference / "bridge_models" / str(first-timedelta(days=7)),
                              first-timedelta(days=7), first, purpose="bridge")
    audit = {"protocol": "nyx_daily_refit_reference_v1", "first_delivery_day": day,
        "last_delivery_day": str(first+timedelta(days=1)), "shared_cutoff_utc": cutoff.isoformat(),
        "training_days": 365, "training_first_day": str(first-timedelta(days=365)),
        "training_last_day": str(first-timedelta(days=1)),
        "current_fit_counts": {"hgb": 12, "test2_pairs": 2}, "reference_models_fitted": 14,
        "future_labels_used": False, "storm_used_as_input": False,
        "artifact_sha256": {p.relative_to(reference).as_posix(): r.sha(p)
                            for p in reference.rglob("*") if p.is_file()}}
    r.atomic_json(reference / "receipt.json", audit)
    return audit


class TrainingChecks(unittest.TestCase):
    def test_exact_365_training_and_future_label_invariance(self):
        module = load_pooled_source()
        day = "2026-10-10"
        inputs = inputs_fixture()
        with patch.object(c, "source_module", return_value=module):
            first = c.build_training(inputs, day, 2)
            current = r.grid(date.fromisoformat(day), date.fromisoformat(day) + timedelta(days=1))
            for labels in inputs["actual"].values():
                labels.loc[current] = 1e12
            second = c.build_training(inputs, day, 2)
        pd.testing.assert_frame_equal(first[0], second[0])
        pd.testing.assert_series_equal(first[1], second[1])
        self.assertEqual(first[4]["training_days"], 365)
        self.assertEqual(len(first[0]), len(r.grid(date(2025, 10, 10), date(2026, 10, 10))) * 4)
        self.assertLess(first[0].index.get_level_values(0).max(), current[0])

    def test_forbidden_target_or_storm_feature_rejected(self):
        module = load_pooled_source()
        for forbidden in ("storm_price", "target_actual"):
            inputs = inputs_fixture()
            inputs["built"].features["cwe_residual_2000"]["FR"][forbidden] = 99.
            with patch.object(c, "source_module", return_value=module):
                with self.assertRaisesRegex(ValueError, "forbidden"):
                    c.build_training(inputs, "2026-10-10", 2)

    def test_missing_prior_day_is_not_silently_shortened(self):
        module = load_pooled_source()
        inputs = inputs_fixture()
        for zone in r.ZONES:
            inputs["built"].features["cwe_residual_2000"][zone] = inputs["built"].features["cwe_residual_2000"][zone].iloc[24:]
        with patch.object(c, "source_module", return_value=module):
            with self.assertRaisesRegex(ValueError, "Incomplete civil grid"):
                c.build_training(inputs, "2026-10-10", 2)

    def test_dst_prediction_uses_physical_hours(self):
        for day, count in (("2026-03-29", 23), ("2026-10-25", 25)):
            first = date.fromisoformat(day)
            frame = pd.DataFrame(50., index=r.grid(first, first + timedelta(days=1)),
                                 columns=c.EXTRA_COLUMNS)
            c.validate_extra(frame, day)
            self.assertEqual(len(frame), count)

    @unittest.skipUnless(importlib.util.find_spec("catboost"), "CatBoost optional in authoring runtime")
    def test_real_small_catboost_ridge_fit(self):
        module = load_pooled_source()
        inputs = inputs_fixture()
        with tempfile.TemporaryDirectory() as folder, patch.object(c, "source_module", return_value=module):
            frames, audit = c.fit_challengers(inputs, "2026-10-10", folder,
                                              threads=2, iterations=2)
            self.assertEqual(set(frames), set(r.ZONES))
            for frame in frames.values():
                c.validate_extra(frame, "2026-10-10")
            self.assertEqual(audit["training"]["training_days"], 365)
            self.assertFalse(audit["future_labels_used"])
            for model in c.MODEL_FILES:
                self.assertGreater((Path(folder) / "models" / model).stat().st_size, 0)


class ReceiptChecks(unittest.TestCase):
    """Exercise whole orchestration without requiring a Parquet engine.

    Pickle substitution is test-only; the delivered command always uses Parquet.
    Model files in these fixtures are receipt sentinels, not fitted estimators.
    """
    def fixture(self, root):
        annual = root / "annual"
        seed_dir = annual / "runs" / "seed"
        seed_dir.mkdir(parents=True)
        run_path = seed_dir / "run.json"
        run_path.write_text("{}")
        bundle, report = annual / "bundle", annual / "report"
        bundle.mkdir(); report.mkdir()
        source = root / "source"
        source.write_text("sealed")
        seed = {"repo": annual, "run_path": run_path, "bundle": bundle, "output": report}
        base = root / "canonical"
        base.mkdir()
        days = ["2026-10-10", "2026-10-11"]
        plan = {"days": days, "seed_run": str(seed_dir)}
        r.atomic_json(base / "replay_plan.json", plan)
        first = date.fromisoformat(days[0])
        index = r.grid(first, first + timedelta(days=2))
        inputs = {"consumed": {str(source): r.sha(source)},
                  "actual": {z: pd.Series(55., index=index) for z in r.ZONES}}
        self.seal_canonical(base, days[0])
        return seed, base, inputs, source

    def seal_canonical(self, base, day, bridge=False):
        first = date.fromisoformat(day)
        current = r.grid(first, first + timedelta(days=1))
        origin = pd.Timestamp(f"{first-timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC")
        destination = base / "origins" / day
        destination.mkdir(parents=True)
        for zone in r.ZONES:
            frame = pd.DataFrame({"price_eur_mwh": 51., "reference_eur_mwh": 50.,
                **{f + "_eur_mwh": 52. + j for j, f in enumerate(r.FAMILIES)},
                "forecast_origin_utc": origin, "horizon_day": 1}, index=current)
            frame.to_pickle(destination / (zone + ".parquet"))
        experts = {}
        for family in r.FAMILIES:
            path = destination / "price_models" / (family + ".cbm")
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"sealed-canonical-model")
            experts[family] = {"training_days": 365, "training_labels_before_origin": True,
                               "Storm_used_as_input": False, "model_sha256": r.sha(path)}
        reference = write_reference_fixture(destination, day, bridge=bridge)
        r.atomic_json(destination / "model_audit.json", {"training_days": 365,
            "price_experts": experts, "reference": reference})
        artifacts = {p.relative_to(destination).as_posix(): r.sha(p)
                     for p in destination.rglob("*") if p.is_file()}
        r.atomic_json(destination / "receipt.json", {"protocol": r.PROTOCOL, "day": day,
            "plan_sha256": r.sha(base / "replay_plan.json"), "predictions_sealed_before_scoring": True,
            "artifact_sha256": artifacts})

    def reseal(self, destination, *, inner=True, outer_audit=True):
        """Re-sign test mutations to distinguish structural and byte checks."""
        reference_path = destination / "reference/receipt.json"
        reference = r.read_json(reference_path)
        if inner:
            reference["artifact_sha256"] = {p.relative_to(reference_path.parent).as_posix(): r.sha(p)
                for p in reference_path.parent.rglob("*") if p.is_file() and p != reference_path}
            r.atomic_json(reference_path, reference)
        if outer_audit:
            audit_path = destination / "model_audit.json"
            audit = r.read_json(audit_path)
            audit["reference"] = reference
            r.atomic_json(audit_path, audit)
        receipt_path = destination / "receipt.json"
        receipt = r.read_json(receipt_path)
        receipt["artifact_sha256"] = {p.relative_to(destination).as_posix(): r.sha(p)
            for p in destination.rglob("*") if p.is_file() and p != receipt_path}
        r.atomic_json(receipt_path, receipt)

    def check_origin(self, base, day="2026-10-10"):
        with patch.object(pd, "read_parquet", side_effect=pd.read_pickle):
            return c.validated_base_origin(base, day, r.sha(base / "replay_plan.json"))

    def test_actual_reference_prefix_and_bridge_models_are_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            _, base, _, _ = self.fixture(Path(folder))
            self.assertIsNotNone(self.check_origin(base))
            self.seal_canonical(base, "2026-10-11", bridge=True)
            self.assertIsNotNone(self.check_origin(base, "2026-10-11"))
            receipt = r.read_json(base / "origins/2026-10-11/receipt.json")
            current = [n for n in receipt["artifact_sha256"] if n.startswith("reference/models/")
                       and n.endswith(".joblib")]
            bridges = [n for n in receipt["artifact_sha256"] if n.startswith("reference/bridge_models/")
                       and n.endswith(".joblib")]
            self.assertEqual((len(current), len(bridges)), (14, 14))

    def test_missing_or_tampered_reference_model_fails_global_byte_check(self):
        for mutation in ("missing", "tampered"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                _, base, _, _ = self.fixture(Path(folder))
                model = base / "origins/2026-10-10/reference/models/2026-10-10/Test2_BE_FR.joblib"
                if mutation == "missing":
                    model.unlink()
                else:
                    model.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "missing or changed"):
                    self.check_origin(base)

    def test_reference_inner_receipt_is_mandatory_and_globally_bound(self):
        for mutation in ("unbound", "changed"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                _, base, _, _ = self.fixture(Path(folder))
                destination = base / "origins/2026-10-10"
                if mutation == "unbound":
                    path = destination / "receipt.json"
                    receipt = r.read_json(path)
                    del receipt["artifact_sha256"]["reference/receipt.json"]
                    r.atomic_json(path, receipt)
                    message = "lacks models or predictions"
                else:
                    path = destination / "reference/receipt.json"
                    path.write_text(path.read_text()+" ")
                    message = "missing or changed"
                with self.assertRaisesRegex(ValueError, message):
                    self.check_origin(base)

    def test_reference_inner_outer_audit_and_inventory_must_agree(self):
        for mutation in ("audit", "inventory", "extra", "unsafe"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                _, base, _, _ = self.fixture(Path(folder))
                destination = base / "origins/2026-10-10"
                inner_path = destination / "reference/receipt.json"
                inner = r.read_json(inner_path)
                if mutation == "audit":
                    inner["training_days"] = 364
                    r.atomic_json(inner_path, inner)
                    self.reseal(destination, inner=False, outer_audit=False)
                    message = "inner and outer audits differ"
                elif mutation == "inventory":
                    del inner["artifact_sha256"]["models/2026-10-10/Test2_BE_FR.joblib"]
                    r.atomic_json(inner_path, inner)
                    self.reseal(destination, inner=False)
                    message = "artifact receipt differs"
                elif mutation == "extra":
                    (destination / "reference/unbound_extra.joblib").write_bytes(b"extra")
                    self.reseal(destination, inner=False)
                    message = "artifact receipt differs"
                else:
                    inner["artifact_sha256"]["/absolute.joblib"] = "0" * 64
                    r.atomic_json(inner_path, inner)
                    self.reseal(destination, inner=False)
                    message = "unsafe research input"
                with self.assertRaisesRegex(ValueError, message):
                    self.check_origin(base)

    def test_daily_reference_count_pair_identity_and_causal_flags_are_checked(self):
        for mutation in ("pair", "duplicate", "kind", "training", "future", "model_count", "flat"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                _, base, _, _ = self.fixture(Path(folder))
                destination = base / "origins/2026-10-10"
                path = destination / "reference/models/2026-10-10/fit_audit.json"
                fit = r.read_json(path)
                message = "reference"
                if mutation == "pair":
                    fit["models"][-1]["pair"] = ["BE", "FR"]
                elif mutation == "duplicate":
                    fit["models"][-1] = fit["models"][-2]
                    message = "Duplicate or unexpected"
                elif mutation == "kind":
                    fit["models"][0]["kind"] = "other"
                elif mutation == "training":
                    fit["models"][0]["audit"]["training_days"] = 364
                elif mutation == "future":
                    fit["models"][-1]["audit"]["future_labels_used"] = True
                elif mutation == "model_count":
                    fit["models"].pop()
                    message = "exactly 14"
                else:
                    # Reject the old synthetic fixture layout even when its
                    # hashes and nominal count have been re-signed.
                    shutil.move(str(destination / "reference/models"), destination / "models")
                    message = "reference lacks artifact receipts"
                if mutation != "flat":
                    r.atomic_json(path, fit)
                self.reseal(destination)
                with self.assertRaisesRegex(ValueError, message):
                    self.check_origin(base)

    def test_real_reference_producer_receipt_layout_is_accepted(self):
        """Run the actual producer orchestration, stub only expensive fitting.

        Directory placement, fit_audit references, nested receipt generation and
        relative inventory come from the delivered fit_daily_reference source.
        """
        source = Path(__file__).parent / "annual_extension/nyx_daily_refit_reference.py"
        spec = importlib.util.spec_from_file_location("_nyx_reference_receipt_test", source)
        producer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(producer)
        first = date(2026, 10, 10)
        current = r.grid(first, first+timedelta(days=2))
        prior_index = r.grid(first-timedelta(days=90), first)
        cutoff = pd.Timestamp("2026-10-09 08:00", tz="Europe/Paris").tz_convert("UTC")
        prepared = {z: pd.DataFrame({"feature": 1.}, index=current) for z in r.ZONES}
        old = {z: pd.DataFrame({"ensemble__q50": 50.}, index=prior_index) for z in r.ZONES}
        quantiles = {z: pd.DataFrame({"q50": 50.}, index=current) for z in r.ZONES}
        actual = {z: pd.Series(55., index=prior_index) for z in r.ZONES}
        prior_module = types.ModuleType("chronos2_hourly.nyx_annual_prior90_cpu")
        prior_module.select_prior90_policy = lambda *a, **k: {
            "history_start_day": str(first-timedelta(days=90)), "history_end_exclusive": str(first),
            "history_hours": len(prior_index), "selected_rule_index": 0, "selected_rule": {}}
        prior_module.apply_prior90_daily = lambda f, *a, **k: pd.DataFrame({"prior90_active": False}, index=f.index)
        pair_module = types.ModuleType("chronos2_hourly.nyx_annual_reference_pair")
        pair_module.PARTNER = {"FR": "BE", "BE": "FR", "DE": "NL", "NL": "DE"}
        pair_module.SIGNALS = ("ensemble__q50", "nyx__q50", "test2__q50", "prior90_active",
                               "spike_probability", "partner_nyx__q50")
        pair_module.build_confirmed_pair = lambda f, **k: pd.DataFrame({"scarcity_confirmed_pair": 50.}, index=f.index)
        def fit(origin, stop, *, directory, purpose, source_cutoff=None, **kwargs):
            write_reference_block(directory, origin, stop, purpose, source_cutoff)
            features = {z: pd.DataFrame({"ensemble__q50": 50., "nyx__q50": 50.,
                "test2__q50": 50., "spike_probability": .1}, index=r.grid(origin, stop)) for z in r.ZONES}
            return features, {z: f[["test2__q50"]] for z, f in features.items()}
        def parquet_write(frame, path, *args, **kwargs):
            frame.to_pickle(path)
        with tempfile.TemporaryDirectory() as folder:
            _, base, _, _ = self.fixture(Path(folder))
            destination = base / "origins/2026-10-10"
            shutil.rmtree(destination / "reference")
            with patch.object(producer, "_prepare", return_value=(first, cutoff, current, prior_index,
                    prepared, old, first)), patch.object(producer, "_fit_block", side_effect=fit), \
                 patch.dict(sys.modules, {prior_module.__name__: prior_module, pair_module.__name__: pair_module}), \
                 patch.object(pd.DataFrame, "to_parquet", parquet_write):
                _, _, reference, _ = producer.fit_daily_reference(str(first), cutoff, None,
                    quantiles, actual, old, destination)
            audit_path = destination / "model_audit.json"
            audit = r.read_json(audit_path)
            audit["reference"] = reference
            r.atomic_json(audit_path, audit)
            self.reseal(destination, inner=False, outer_audit=False)
            self.assertIsNotNone(self.check_origin(base))

    def fake_predictor(self, calls):
        def predict(inputs, day, destination, **kwargs):
            calls.append(day)
            for name in c.MODEL_FILES:
                path = destination / "models" / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"new-model")
            first = date.fromisoformat(day)
            frames = {z: pd.DataFrame(54., index=r.grid(first, first+timedelta(days=1)),
                                      columns=c.EXTRA_COLUMNS) for z in r.ZONES}
            return frames, {"future_labels_used": False, "training": {"training_days": 365}}
        return predict

    def run_fixture(self, root, operation):
        seed, base, inputs, source = self.fixture(root)
        def original(base_path, s, i):
            return r.read_json(base_path / "replay_plan.json"), base_path / "replay_plan.json"
        def parquet_write(frame, path, *args, **kwargs):
            frame.to_pickle(path)
        with patch.object(r, "configure"), patch.object(c, "validate_original_plan", side_effect=original), \
             patch.object(c, "code_hashes", return_value={"test": "fixed"}), \
             patch.object(c, "runtime_versions", return_value={"test": "fixed"}), \
             patch.object(pd, "read_parquet", side_effect=pd.read_pickle), \
             patch.object(pd.DataFrame, "to_parquet", parquet_write):
            operation(seed, base, inputs, source)

    def test_resume_partial_seals_preserves_current_and_components(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def operation(seed, base, inputs, source):
                calls = []
                output = root / "challengers"
                prediction = self.fake_predictor(calls)
                first = c.run(seed, base, output, inputs=inputs, predictor=prediction)
                self.assertEqual(first["completed"], 1)
                self.assertEqual(first["pending_canonical_days"], 1)
                frame = pd.read_csv(output / "OOF.csv")
                self.assertTrue(frame.nyx_current.eq(51.).all())
                self.assertTrue(frame.expert_fr_residual.eq(52.).all())
                self.assertTrue(set(c.EXTRA_COLUMNS) <= set(frame))
                self.assertTrue(frame.horizon_day.eq(1).all())
                c.run(seed, base, output, inputs=inputs, predictor=prediction)
                self.assertEqual(calls, ["2026-10-10"])
                # Finishing the canonical replay never silently changes an existing challenger plan.
                self.seal_canonical(base, "2026-10-11")
                c.run(seed, base, output, inputs=inputs, predictor=prediction)
                self.assertEqual(calls, ["2026-10-10"])
                self.assertEqual(len(pd.read_csv(output / "OOF.csv")), 96)
            self.run_fixture(root, operation)

    def test_recipe_change_source_change_and_seal_tamper_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def operation(seed, base, inputs, source):
                output = root / "challengers"
                kwargs = {"inputs": inputs, "predictor": self.fake_predictor([])}
                c.run(seed, base, output, **kwargs)
                with self.assertRaisesRegex(ValueError, "plan, inputs"):
                    c.run(seed, base, output, iterations=301, **kwargs)
                source.write_text("changed")
                with self.assertRaisesRegex(ValueError, "Frozen research input changed"):
                    c.run(seed, base, output, **kwargs)
                source.write_text("sealed")
                model = output / "origins/2026-10-10/models/rmse.cbm"
                model.write_bytes(model.read_bytes() + b"tamper")
                with self.assertRaisesRegex(ValueError, "changed"):
                    c.run(seed, base, output, **kwargs)
            self.run_fixture(root, operation)

    def test_missing_all_original_seals_has_actionable_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def operation(seed, base, inputs, source):
                (base / "origins/2026-10-10/receipt.json").unlink()
                with self.assertRaisesRegex(ValueError, "No sealed canonical origin"):
                    c.run(seed, base, root / "challengers", inputs=inputs)
            self.run_fixture(root, operation)

    def test_original_plan_verifies_sources_code_and_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            seed, base, inputs, source = self.fixture(root)
            module_path = root / "canonical_source.py"
            module_path.write_text("canonical source")
            names = ("nyx_daily_refit_reference", "nyx_pooled_cpu_price_model",
                     "nyx_annual_cpu_live", "nyx_annual_cpu_bundle_builder")
            plan = {"protocol": r.PROTOCOL, "training_days": 365, "scope": "final_experts",
                "j2_relabeling_permitted": False, "seed_run": str(seed["run_path"].parent),
                "source_sha256": inputs["consumed"], "days": ["2026-10-10"],
                "runtime_versions": {"test": "fixed"},
                "code_sha256": {"replay.py": r.sha(r.__file__),
                                **{name: r.sha(module_path) for name in names}}}
            r.atomic_json(base / "replay_plan.json", plan)
            module = types.SimpleNamespace(__file__=str(module_path))
            with patch.object(c.importlib, "import_module", return_value=module), \
                 patch.object(c, "runtime_versions", return_value={"test": "fixed"}), \
                 patch.object(r, "coverage", return_value={"conditional_j1_days": ["2026-10-10"]}):
                c.validate_original_plan(base, seed, inputs)
                module_path.write_text("changed canonical source")
                with self.assertRaisesRegex(ValueError, "code changed"):
                    c.validate_original_plan(base, seed, inputs)
                module_path.write_text("canonical source")
                source.write_text("changed frozen donor")
                with self.assertRaisesRegex(ValueError, "Frozen research input changed"):
                    c.validate_original_plan(base, seed, inputs)
                source.write_text("sealed")
                with patch.object(c, "runtime_versions", return_value={"test": "new"}):
                    with self.assertRaisesRegex(ValueError, "dependencies differ"):
                        c.validate_original_plan(base, seed, inputs)

    @unittest.skipUnless(importlib.util.find_spec("catboost") and importlib.util.find_spec("pyarrow"),
                         "Real model + Parquet integration dependencies unavailable")
    def test_real_parquet_saved_model_seal_and_resume(self):
        module = load_pooled_source()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            seed, base, fixture_inputs, source = self.fixture(root)
            inputs = inputs_fixture()
            inputs["consumed"] = fixture_inputs["consumed"]
            # Convert only the fixture predictions to actual Parquet. Original
            # canonical model sentinels still test reuse, never fresh model fits.
            destination = base / "origins/2026-10-10"
            for path in destination.glob("*.parquet"):
                pd.read_pickle(path).to_parquet(path)
            receipt_path = destination / "receipt.json"
            receipt = r.read_json(receipt_path)
            receipt["artifact_sha256"] = {p.relative_to(destination).as_posix(): r.sha(p)
                for p in destination.rglob("*") if p.is_file() and p != receipt_path}
            r.atomic_json(receipt_path, receipt)
            def original(path, s, i):
                return r.read_json(path / "replay_plan.json"), path / "replay_plan.json"
            output = root / "real_challengers"
            with patch.object(r, "configure"), patch.object(c, "validate_original_plan", side_effect=original), \
                 patch.object(c, "source_module", return_value=module), \
                 patch.object(c, "code_hashes", return_value={"test_only_code": r.sha(c.__file__)}):
                c.run(seed, base, output, inputs=inputs, iterations=2, threads=2)
                def no_fit(*args, **kwargs):
                    raise AssertionError("Resume must not train sealed models")
                c.run(seed, base, output, inputs=inputs, iterations=2, threads=2, predictor=no_fit)
                manifest = r.read_json(output / "OOF.manifest.json")
                self.assertEqual(manifest["oof_csv_sha256"], r.sha(output / "OOF.csv"))
                self.assertEqual(len(pd.read_csv(output / "OOF.csv")), 96)
                model = output / "origins/2026-10-10/models/huber.cbm"
                model.write_bytes(model.read_bytes() + b"tamper")
                with self.assertRaisesRegex(ValueError, "changed"):
                    c.run(seed, base, output, inputs=inputs, iterations=2, threads=2, predictor=no_fit)


if __name__ == "__main__":
    unittest.main()
