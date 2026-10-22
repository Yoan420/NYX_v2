"""Cloud orchestration tests with fake adapters and S3; no model or provider I/O."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from nyx_runtime import cli, snapshot, storage


FIRST_DAY = "2026-03-28"
ORIGIN = "2026-03-27T07:00:00+00:00"


class FrozenClock:
    @classmethod
    def now(cls, tz=None):
        value = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
        return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    fromisoformat = staticmethod(datetime.fromisoformat)
    combine = staticmethod(datetime.combine)


class FakeStore:
    def __init__(self, events, local_latest, *, fail_publication=False):
        self.events, self.local_latest = events, local_latest
        self.fail_publication = fail_publication

    @contextmanager
    def lock(self):
        self.events.append("cloud_lock")
        try:
            yield self
        finally:
            self.events.append("cloud_unlock")

    def restore_state(self, destination):
        self.events.append("restore_state")
        return None

    def publish_state(self, source, run_id):
        self.events.append("publish_state")
        self._previous_pointer_unchanged()

    def publish_run(self, source, run_id, success):
        self.events.append("publish_run_success" if success else "publish_run_failure")
        self._previous_pointer_unchanged()
        status = json.loads((source / "status.json").read_text())
        if success:
            if status["state"] != "COMPLETE" or not (source / "reports/report.html").is_file():
                raise AssertionError("Cloud publication preceded complete HTML production")
            if self.fail_publication:
                raise OSError("synthetic S3 publication failure")

    def _previous_pointer_unchanged(self):
        if self.local_latest.read_text() != '{"run_id":"previous-success"}':
            raise AssertionError("Local success pointer changed before full cloud publication")


class RuntimeCLITests(unittest.TestCase):
    def test_plan_preserves_shared_cutoff_and_physical_dst_days(self):
        with patch.object(cli, "datetime", FrozenClock):
            spring = cli.make_plan("2026-03-28", "2026-03-27T08:00:00+01:00")
            # A frozen later wall clock permits both sides of the autumn switch.
        class LaterClock(FrozenClock):
            @classmethod
            def now(cls, tz=None):
                value = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
                return value.astimezone(tz) if tz else value.replace(tzinfo=None)
        with patch.object(cli, "datetime", LaterClock):
            autumn = cli.make_plan("2026-10-24", "2026-10-23T08:00:00+02:00")
        self.assertEqual(spring["forecast_origin_utc"], ORIGIN)
        self.assertEqual([d["hours_per_country"] for d in spring["deliveries"]], [24, 23])
        self.assertEqual([d["hours_per_country"] for d in autumn["deliveries"]], [24, 25])
        self.assertEqual([d["horizon_day"] for d in spring["deliveries"]], [1, 2])
        self.assertFalse(spring["qualified_for_production"])

    def test_plan_rejects_naive_wrong_utc_offset_future_or_noncanonical_date(self):
        cases = [
            ("2026-03-28", "2026-03-27T08:00:00"),
            ("2026-03-28", "2026-03-27T08:00:00+02:00"),
            ("2026-10-24", "2026-10-23T08:00:00+02:00"),
            ("20260328", ORIGIN),
        ]
        with patch.object(cli, "datetime", FrozenClock):
            for first, origin in cases:
                with self.subTest(first=first, origin=origin), self.assertRaises(ValueError):
                    cli.make_plan(first, origin)

    def test_origin_alone_selects_the_next_paris_civil_day(self):
        plan = cli.make_plan(as_of="2026-03-27T07:00:00Z")
        self.assertEqual(plan["first_day"], FIRST_DAY)
        self.assertEqual(plan["forecast_origin_utc"], ORIGIN)

    def test_plan_does_not_import_scientific_or_aws_dependencies(self):
        code = """
import contextlib, io, json, sys
from nyx_runtime.cli import main
with contextlib.redirect_stdout(io.StringIO()):
    result = main(['plan', '--first-day', '2026-03-28', '--as-of', '2026-03-27T07:00:00Z'])
print(json.dumps({'code': result, 'loaded': sorted(set(sys.modules) & {'torch', 'pandas', 'numpy', 'boto3', 'tshistory_lite', 'chronos'})}))
"""
        done = subprocess.run([sys.executable, "-c", code], cwd=cli.ROOT,
                              text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(done.stdout), {"code": 0, "loaded": []})

    def test_main_rejects_other_horizons_countries_or_duplicate_countries(self):
        for extra in (("--horizons", "1"), ("--horizons", "1", "2", "2"),
                      ("--zones", "FR", "DE", "BE", "ES"),
                      ("--zones", "FR", "FR", "DE", "BE")):
            with self.subTest(extra=extra), patch.object(cli, "execute") as execute, \
                 patch("sys.stderr", new_callable=io.StringIO):
                result = cli.main(["plan", "--first-day", FIRST_DAY, *extra])
                self.assertEqual(result, 2)
                execute.assert_not_called()

    def _run(self, root, *, command="forecast", ready=True, cloud=True,
             publication_fails=False, model_error=None, snapshot_error=None):
        inputs, outputs = root / "inputs", root / "outputs"
        inputs.mkdir()
        (inputs / "snapshot.json").write_text(json.dumps({"protocol": snapshot.PROTOCOL, "fixture": True}))
        outputs.mkdir()
        latest = outputs / "latest.json"
        latest.write_text('{"run_id":"previous-success"}')
        events = []
        store = FakeStore(events, latest, fail_publication=publication_fails)

        def inspect(*args):
            events.append("inspect")
            return {"ready": ready, "blockers": [] if ready else ["synthetic missing seed"]}

        def run(*args):
            events.append("fit")
            if model_error is not None:
                raise model_error
            output = args[-1]
            csv = output / "predictions.csv"
            csv.write_text("synthetic new model output")
            return {"predictions_csv": str(csv), "qualified_for_production": False}

        def render(source, destination):
            events.append("render")
            destination.mkdir()
            html = destination / "report.html"
            html.write_text("synthetic rendered report")
            return {"html_files": [str(html)]}

        fake_live = SimpleNamespace(inspect_live=Mock(side_effect=inspect),
            run_live=Mock(side_effect=run), _redact=str)
        fake_report = SimpleNamespace(render_reports=Mock(side_effect=render))
        args = argparse.Namespace(command=command, input_root=inputs, output_root=outputs,
            input_s3_uri=None, output_s3_uri="s3://test-bucket/results" if cloud else None,
            skip_reporting_sources=True)
        patches = [
            patch.object(cli, "runtime_lock", return_value=nullcontext()),
            patch.object(cli, "verify_runtime"), patch.object(cli, "copy_state"),
            patch.object(cli, "adapter", return_value=(fake_live, fake_report)),
            patch.object(snapshot, "validate_snapshot", side_effect=snapshot_error,
                         return_value={"protocol": snapshot.PROTOCOL}),
            patch.object(storage, "S3Store", return_value=store),
            patch.dict(os.environ), patch("sys.stdout", new_callable=io.StringIO),
            patch.object(cli, "RUNTIME", root / "runtime"),
        ]
        from contextlib import ExitStack
        with ExitStack() as stack:
            active = [stack.enter_context(value) for value in patches]
            try:
                code = cli.execute(args, cli.make_plan(FIRST_DAY, ORIGIN))
                error = None
            except BaseException as value:
                code, error = None, value
        return SimpleNamespace(code=code, error=error, events=events, latest=latest,
            outputs=outputs, live=fake_live, report=fake_report, mocks=active)

    def test_cloud_success_pointer_follows_full_render_and_remote_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self._run(Path(folder))
            self.assertIsNone(result.error)
            self.assertEqual(result.code, 0)
            self.assertLess(result.events.index("fit"), result.events.index("render"))
            self.assertLess(result.events.index("render"), result.events.index("publish_run_success"))
            self.assertEqual(result.events[-1], "cloud_unlock")
            pointer = json.loads(result.latest.read_text())
            self.assertNotEqual(pointer["run_id"], "previous-success")
            state = json.loads((result.outputs / pointer["status_file"]).read_text())
            self.assertEqual(state["state"], "COMPLETE")
            self.assertTrue(state["new_predictions_computed"])

    def test_remote_publication_failure_preserves_previous_success_and_local_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self._run(Path(folder), publication_fails=True)
            self.assertIsInstance(result.error, OSError)
            self.assertEqual(json.loads(result.latest.read_text())["run_id"], "previous-success")
            states = [json.loads(p.read_text()) for p in result.outputs.glob("*/status.json")]
            self.assertEqual([p["state"] for p in states], ["FAILED"])
            self.assertIn("publish_run_failure", result.events)

    def test_model_failure_preserves_previous_success_and_never_publishes_state(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self._run(Path(folder), model_error=ValueError("synthetic fit failure"))
            self.assertIsInstance(result.error, ValueError)
            self.assertNotIn("render", result.events)
            self.assertNotIn("publish_state", result.events)
            self.assertNotIn("publish_run_success", result.events)
            self.assertEqual(json.loads(result.latest.read_text())["run_id"], "previous-success")

    def test_preflight_never_fits_renders_or_publishes(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self._run(Path(folder), command="preflight", ready=False)
            self.assertIsNone(result.error)
            self.assertEqual(result.code, 2)
            result.live.run_live.assert_not_called()
            result.report.render_reports.assert_not_called()
            self.assertFalse(any(event.startswith("publish_") for event in result.events))
            self.assertEqual(list(result.outputs.glob("*/status.json")), [])

    def test_failed_input_verification_blocks_adapter_and_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self._run(Path(folder), snapshot_error=ValueError("corrupt immutable snapshot"))
            self.assertIsInstance(result.error, ValueError)
            result.mocks[3].assert_not_called()
            self.assertNotIn("inspect", result.events)
            self.assertNotIn("fit", result.events)

    def test_blocked_forecast_records_blocker_without_reusing_previous_forecast(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self._run(Path(folder), ready=False)
            self.assertEqual(result.code, 2)
            self.assertIsNone(result.error)
            result.live.run_live.assert_not_called()
            self.assertEqual(json.loads(result.latest.read_text())["run_id"], "previous-success")
            statuses = list(result.outputs.glob("*/status.json"))
            self.assertEqual(json.loads(statuses[0].read_text())["state"], "BLOCKED")
            self.assertEqual(list(result.outputs.glob("*/predictions.csv")), [])

    def test_input_and_output_roots_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(input_root=root, output_root=root / "outputs",
                                      output_s3_uri=None, command="forecast")
            with self.assertRaisesRegex(ValueError, "distinctes et non imbriquées"):
                cli.execute(args, cli.make_plan(FIRST_DAY, ORIGIN))

    def test_overlapping_s3_prefixes_fail_before_credentials_or_download(self):
        pairs = (
            ("s3://test-bucket/nyx", "s3://test-bucket/nyx"),
            ("s3://test-bucket/nyx/inputs", "s3://test-bucket/nyx"),
            ("s3://test-bucket/nyx", "s3://test-bucket/nyx/results"),
            ("s3://test-bucket/nyx/", "s3://test-bucket/nyx"),
            ("s3://test-bucket/", "s3://test-bucket/results"),
            ("s3://test-bucket/inputs", "s3://test-bucket/"),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for input_uri, output_uri in pairs:
                args = argparse.Namespace(command="forecast", input_root=root / "inputs",
                    output_root=root / "outputs", input_s3_uri=input_uri,
                    output_s3_uri=output_uri, skip_reporting_sources=True)
                with self.subTest(input_uri=input_uri, output_uri=output_uri), \
                     patch.object(storage, "S3Store") as store, \
                     patch.object(storage, "download_prefix") as download, \
                     patch.object(snapshot, "validate_snapshot") as validate, \
                     patch.object(cli, "runtime_lock") as lock:
                    with self.assertRaisesRegex(ValueError, "préfixes S3"):
                        cli.execute(args, cli.make_plan(FIRST_DAY, ORIGIN))
                    store.assert_not_called()
                    download.assert_not_called()
                    validate.assert_not_called()
                    lock.assert_not_called()

    def test_state_restore_removes_stale_directories_absent_from_selected_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / "source", root / "runtime"
            source.mkdir()
            for relative in cli.STATE_PATHS:
                stale = target / relative / "stale.json"
                stale.parent.mkdir(parents=True)
                stale.write_text("stale prior state")
            fresh = source / cli.STATE_PATHS[0] / "fresh.json"
            fresh.parent.mkdir(parents=True)
            fresh.write_text("verified selected state")
            cli.copy_state(source, target)
            self.assertEqual((target / cli.STATE_PATHS[0] / "fresh.json").read_text(), "verified selected state")
            self.assertFalse(list(target.rglob("stale.json")))
            for relative in cli.STATE_PATHS[1:]:
                self.assertFalse((target / relative).exists())

    def test_repeated_local_runs_keep_newer_state_until_input_snapshot_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, outputs, runtime = root / "inputs", root / "outputs", root / "runtime"
            inputs.mkdir()
            snapshot_file = inputs / "snapshot.json"
            snapshot_file.write_text('{"protocol":"synthetic_snapshot","version":1}')
            seed_state = inputs / "state" / cli.STATE_PATHS[0] / "state.json"
            seed_state.parent.mkdir(parents=True)
            seed_state.write_text("original bootstrap state")
            continued_state = runtime / cli.STATE_PATHS[0] / "state.json"
            fake_live = SimpleNamespace(inspect_live=Mock(return_value={"ready": True}))
            args = argparse.Namespace(command="preflight", input_root=inputs, output_root=outputs,
                input_s3_uri=None, output_s3_uri=None, skip_reporting_sources=True)
            with patch.object(cli, "RUNTIME", runtime), \
                 patch.object(cli, "runtime_lock", side_effect=lambda: nullcontext()), \
                 patch.object(cli, "verify_runtime"), \
                 patch.object(snapshot, "validate_snapshot", return_value={"protocol": snapshot.PROTOCOL}), \
                 patch.object(cli, "adapter", return_value=(fake_live, None)), \
                 patch.object(cli, "copy_state", wraps=cli.copy_state) as copy_state, \
                 patch.dict(os.environ), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(cli.execute(args, cli.make_plan(FIRST_DAY, ORIGIN)), 0)
                self.assertEqual(continued_state.read_text(), "original bootstrap state")
                continued_state.write_text("newer verified daily continuation")
                self.assertEqual(cli.execute(args, cli.make_plan(FIRST_DAY, ORIGIN)), 0)
                self.assertEqual(continued_state.read_text(), "newer verified daily continuation")
                self.assertEqual(copy_state.call_count, 1)
                snapshot_file.write_text('{"protocol":"synthetic_snapshot","version":2}')
                self.assertEqual(cli.execute(args, cli.make_plan(FIRST_DAY, ORIGIN)), 0)
                self.assertEqual(continued_state.read_text(), "original bootstrap state")
                self.assertEqual(copy_state.call_count, 2)


if __name__ == "__main__":
    unittest.main()
