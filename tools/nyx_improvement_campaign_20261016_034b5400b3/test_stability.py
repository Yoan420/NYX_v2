"""Independent synthetic checks of frozen selection and paired day intervals."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import stability


def synthetic(directory, days=30, horizons=(1,), countries=("FR",), incomplete=False, selected="level_14"):
    rows, selections = [], []
    for country in countries:
        for horizon in horizons:
            first = pd.Timestamp("2026-03-10")
            begin = first + pd.Timedelta(days=3)
            end = begin + pd.Timedelta(days=days-1)
            cutoff = (begin-pd.Timedelta(days=horizon)+pd.Timedelta(hours=8)).tz_localize(stability.TZ).tz_convert("UTC")
            for number in range(days+3):
                day = first + pd.Timedelta(days=number)
                origin = (day-pd.Timedelta(days=horizon)+pd.Timedelta(hours=8)).tz_localize(stability.TZ).tz_convert("UTC")
                for hour, delivery in enumerate(stability.expected_grid(day)):
                    target = 50. + number + np.sin(hour)
                    heldout = day >= begin
                    rows.append({"country":country,"horizon_day":horizon,"delivery_utc":delivery.isoformat(),
                        "forecast_origin_utc":origin.isoformat(),"target_price":target,
                        "storm_price":target+8.,"prediction_current":target+10.,
                        "prediction_level_14":target if heldout else target+10000.,
                        "prediction_component_better_on_holdout":target if selected=="current" else target-50.})
            selections.append({"country":country,"horizon_day":horizon,"ready":True,
                "benchmark_selection_ready":True,"selected":selected,"selection_status":"RESEARCH_CANDIDATE_SELECTED",
                "holdout_start":str(begin.date()),"holdout_end":str(end.date()),"holdout_days":days,
                "selection_cutoff_utc":cutoff.isoformat(),"tuning_days_list":[str(first.date())]})
    directory=Path(directory)
    directory.mkdir(parents=True)
    frame=pd.DataFrame(rows)
    if incomplete:
        frame.loc[frame.index[-1],"storm_price"]=np.nan
    frame.to_csv(directory/"predictions.csv",index=False)
    (directory/"study.json").write_text(json.dumps({"scope":"final_experts","selections":selections}),encoding="utf-8")
    seal(directory)
    return directory


def seal(directory):
    (directory/"complete.json").write_text(json.dumps({"output_sha256":{
        name:stability.sha(directory/name) for name in ("study.json","predictions.csv")}}),encoding="utf-8")


class StabilityTests(unittest.TestCase):
    def test_perfect_selected_improvement_uses_holdout_only_and_dst_hours(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=synthetic(Path(temporary)/"study")
            output=Path(temporary)/"stability"
            stability.run(source,output)
            result=json.loads((output/"status.json").read_text())
            self.assertEqual(result["status"],"HELDOUT_STABILITY_COMPLETE")
            self.assertFalse(result["production_qualification"])
            score=result["groups"][0]
            self.assertEqual(score["selected"],"level_14")
            self.assertEqual(score["complete_paired_days"],30)
            self.assertEqual(score["hours"],30*24-1)
            self.assertAlmostEqual(score["rmse"],0.)
            self.assertAlmostEqual(score["rmse_diff_storm"],-8.)
            self.assertAlmostEqual(score["mae_diff_current"],-10.)
            self.assertAlmostEqual(score["strict_win_rate"],1.)
            self.assertAlmostEqual(score["rmse_diff_storm_ci_high"],-8.)
            self.assertAlmostEqual(score["strict_win_rate_ci_low"],1.)
            daily=pd.read_csv(output/"daily_paired_losses.csv")
            self.assertEqual(len(daily),30)
            self.assertEqual(daily.hours.min(),23)
            self.assertEqual(set(daily.delivery_day),set(pd.date_range("2026-03-13",periods=30).strftime("%Y-%m-%d")))
            strata=pd.read_csv(output/"strata_metrics.csv")
            for _,part in strata.groupby("stratum"):
                self.assertEqual(part.hours.sum(),score["hours"])
            self.assertTrue(strata.descriptive_only.all())

    def test_frozen_current_is_not_replaced_by_better_holdout_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=synthetic(Path(temporary)/"study",selected="current")
            output=Path(temporary)/"out"
            stability.run(source,output)
            score=json.loads((output/"status.json").read_text())["groups"][0]
            self.assertEqual(score["selected"],"current")
            self.assertEqual(score["selection_family"],"current_control")
            self.assertAlmostEqual(score["rmse_diff_current"],0.)
            self.assertAlmostEqual(score["strict_win_rate"],0.)

    def test_missing_storm_keeps_comparators_paired_and_refuses_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=synthetic(Path(temporary)/"study",incomplete=True)
            output=Path(temporary)/"out"
            stability.run(source,output)
            status=json.loads((output/"status.json").read_text())
            self.assertEqual(status["status"],"NOT_READY")
            score=status["groups"][0]
            self.assertEqual(score["complete_paired_days"],29)
            self.assertEqual(score["excluded_holdout_hours"],24)
            self.assertNotIn("strict_win_rate_ci_low",score)
            daily=pd.read_csv(output/"daily_paired_losses.csv")
            self.assertEqual(score["hours"],daily.hours.sum())
            self.assertAlmostEqual(score["storm_rmse"],8.)

    def test_too_short_refuses_confidence_and_groups_remain_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=synthetic(Path(temporary)/"study",days=13,horizons=(1,2),countries=("FR","BE"))
            output=Path(temporary)/"out"
            stability.run(source,output)
            status=json.loads((output/"status.json").read_text())
            self.assertEqual(status["status"],"NOT_READY")
            self.assertEqual(len(status["groups"]),4)
            self.assertEqual({(r["country"],r["horizon_day"]) for r in status["groups"]},
                             {("FR",1),("FR",2),("BE",1),("BE",2)})
            self.assertTrue(all(r["complete_paired_days"]==13 for r in status["groups"]))
            self.assertTrue(all("strict_win_rate_ci_low" not in r for r in status["groups"]))

    def test_verified_resume_source_mutation_and_output_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=synthetic(Path(temporary)/"study")
            output=Path(temporary)/"out"
            original=stability.run(source,output)
            self.assertEqual(stability.run(source,output),original)
            (source/"predictions.csv").write_text((source/"predictions.csv").read_text()+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"study output changed"):
                stability.run(source,output)
            seal(source)
            with self.assertRaisesRegex(ValueError,"pinned to different"):
                stability.run(source,output)
        with tempfile.TemporaryDirectory() as temporary:
            source=synthetic(Path(temporary)/"study")
            output=Path(temporary)/"out"
            stability.run(source,output)
            (output/"metrics.csv").write_text("corrupt",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"stability output changed"):
                stability.run(source,output)

    def test_tuning_overlap_and_duplicate_hours_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=synthetic(Path(temporary)/"study")
            doc=json.loads((source/"study.json").read_text())
            doc["selections"][0]["tuning_days_list"]=[doc["selections"][0]["holdout_start"]]
            (source/"study.json").write_text(json.dumps(doc),encoding="utf-8")
            seal(source)
            with self.assertRaisesRegex(ValueError,"Tuning overlaps"):
                stability.run(source,Path(temporary)/"out")
        with tempfile.TemporaryDirectory() as temporary:
            source=synthetic(Path(temporary)/"study")
            frame=pd.read_csv(source/"predictions.csv")
            pd.concat([frame,frame.iloc[[-1]]]).to_csv(source/"predictions.csv",index=False)
            seal(source)
            with self.assertRaisesRegex(ValueError,"Duplicate physical"):
                stability.run(source,Path(temporary)/"out")

    def test_strict_ties_are_not_wins(self):
        frame=pd.DataFrame({"target_price":[0.,0.],"storm_price":[2.,2.],
            "prediction_current":[2.,2.],"prediction_level_14":[2.,1.]})
        score=stability.measures(stability.sums(frame,"level_14"))
        self.assertEqual(score["strict_win_rate"],.5)
        self.assertEqual(score["tie_rate"],.5)
        self.assertEqual(score["current_strict_win_rate"],0.)


if __name__=="__main__":
    unittest.main()
