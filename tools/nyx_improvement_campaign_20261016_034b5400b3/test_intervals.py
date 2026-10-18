"""Empirical intervals must respect label availability and selected-rule splits."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import intervals


def fixture(days=95,horizons=(1,)):
    rows=[]
    for horizon in horizons:
        for n in range(days):
            day=pd.Timestamp("2026-01-01")+pd.Timedelta(days=n)
            origin=(day-pd.Timedelta(days=horizon)+pd.Timedelta(hours=8)).tz_localize("Europe/Paris").tz_convert("UTC")
            available=(day+pd.Timedelta(days=1)).tz_localize("Europe/Paris").tz_convert("UTC")
            grid=pd.date_range(day.tz_localize("Europe/Paris"),(day+pd.Timedelta(days=1)).tz_localize("Europe/Paris"),freq="h",inclusive="left").tz_convert("UTC")
            for hour,stamp in enumerate(grid):
                point=60.+np.sin(hour/4)
                rows.append({"country":"FR","horizon_day":horizon,"delivery_utc":stamp.isoformat(),
                    "forecast_origin_utc":origin.isoformat(),"target_available_utc":available.isoformat(),
                    "reference_price":point,"expert_fr_residual":point+1.,"expert_cwe_residual":point+2.,
                    "expert_cwe_absolute":point+3.,"nyx_current":point,"prediction_current":point,
                    "target_price":point+4+np.sin(n/3)+np.cos(hour/4)+10*(horizon-1),"storm_price":point+2.})
    raw=pd.DataFrame(rows)
    document={"scope":"final_experts","production_modified":False}
    frame=intervals.validate(raw,document)
    selections=[]
    for horizon in horizons:
        group=frame[frame.horizon_day==horizon]
        dates=sorted(group.delivery_day.unique())
        selections.append({"country":"FR","horizon_day":horizon,"selected":"current","ready":days>=90,
            "holdout_days":30,"holdout_start":str(dates[-30] if days>=30 else dates[0]),"holdout_end":str(dates[-1]),
            "selection_status":"NO_CANDIDATE_PASSED_TUNING_GATES" if days>=90 else "NOT_READY"})
    return frame,{**document,"selections":selections}


def completed_source(directory,frame,document):
    p=Path(directory)/"source";p.mkdir()
    (p/"study.json").write_text(json.dumps(document),encoding="utf-8")
    frame.drop(columns=[c for c in frame if c.startswith("_")]).to_csv(p/"predictions.csv",index=False)
    (p/"complete.json").write_text(json.dumps({"output_sha256":{
        name:intervals.sha(p/name) for name in ("study.json","predictions.csv")}}),encoding="utf-8")
    return p


class IntervalTests(unittest.TestCase):
    def test_future_labels_do_not_change_earlier_bounds(self):
        frame,doc=fixture()
        baseline,_=intervals.generate(frame,doc["selections"])
        changed=frame.copy()
        dates=sorted(frame.delivery_day.unique())
        marker=dates[-15]
        changed.loc[changed.delivery_day>=marker,"target_price"]+=1000.
        alternate,_=intervals.generate(changed,doc["selections"])
        before=frame.delivery_day<=marker
        np.testing.assert_allclose(baseline.loc[before,["lower","upper"]],alternate.loc[before,["lower","upper"]],equal_nan=True)

    def test_embargo_partial_days_and_no_same_day_residual(self):
        frame,doc=fixture()
        dates=sorted(frame.delivery_day.unique())
        current=dates[-30]
        # Leave exactly 30 earlier complete days, then delay one hour of one
        # of them: the entire day must be excluded and no interval fabricated.
        origin=frame.loc[frame.delivery_day==current,"forecast_origin_utc"].iloc[0]
        eligible_dates=sorted(frame.loc[(frame.forecast_origin_utc<origin)&
            (frame.target_available_utc<=origin),"delivery_day"].unique())[-30:]
        keep=frame.delivery_day.isin(eligible_dates)|frame.delivery_day.ge(current)
        reduced=frame.loc[keep].copy()
        a,audit=intervals.generate(reduced,doc["selections"])
        mask=a.delivery_day==current
        self.assertTrue(np.isfinite(a.loc[mask,"lower"]).all())
        delayed=reduced.copy()
        pos=delayed.index[delayed.delivery_day==eligible_dates[0]][0]
        delayed.loc[pos,"target_available_utc"]=origin+pd.Timedelta(hours=1)
        b,_=intervals.generate(delayed,doc["selections"])
        self.assertTrue(b.loc[mask,"lower"].isna().all())
        self.assertTrue((b.loc[mask,"calibration_days"]==29).all())
        for item in audit:
            self.assertLessEqual(pd.Timestamp(item["latest_used_label_available_utc"]),pd.Timestamp(item["origin"]))

    def test_j1_j2_are_never_pooled(self):
        frame,doc=fixture(horizons=(1,2))
        a,_=intervals.generate(frame,doc["selections"])
        changed=frame.copy();changed.loc[changed.horizon_day==2,"target_price"]+=500.
        b,_=intervals.generate(changed,doc["selections"])
        mask=frame.horizon_day==1
        np.testing.assert_allclose(a.loc[mask,["lower","upper"]],b.loc[mask,["lower","upper"]],equal_nan=True)
        self.assertGreater(a.loc[(a.horizon_day==2)&a.heldout_block,"lower_residual_quantile"].mean(),
                           a.loc[(a.horizon_day==1)&a.heldout_block,"lower_residual_quantile"].mean()+5)

    def test_short_actual_like_archive_has_no_calibrated_outputs_and_resume_pinned(self):
        frame,doc=fixture(days=1)
        with tempfile.TemporaryDirectory() as directory:
            source=completed_source(directory,frame,doc)
            output=Path(directory)/"intervals"
            intervals.run(source,output)
            status=json.loads((output/"status.json").read_text())
            prediction=pd.read_csv(output/"interval_predictions.csv")
            metrics=pd.read_csv(output/"metrics.csv")
            self.assertEqual(status["status"],"NOT_READY_NO_CALIBRATED_HOLDOUT")
            self.assertEqual(status["calibrated_prediction_hours"],0)
            self.assertFalse(status["production_qualification"])
            self.assertTrue(prediction.lower.isna().all())
            self.assertEqual(int(metrics.evaluated_hours.sum()),0)
            self.assertEqual(intervals.run(source,output),output/"report.html")
            (source/"predictions.csv").write_text("changed",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"source changed"):
                intervals.run(source,output)

    def test_interval_metrics_same_hours_and_reserved_block_only(self):
        frame,doc=fixture()
        predictions,_=intervals.generate(frame,doc["selections"])
        self.assertFalse(np.isfinite(predictions.loc[~predictions.heldout_block,"lower"]).any())
        metrics=intervals.evaluate(predictions,doc["selections"])
        self.assertEqual(int(metrics.evaluated_days.iloc[0]),30)
        self.assertEqual(int(metrics.evaluated_hours.iloc[0]),len(predictions[predictions.heldout_block]))
        rows=predictions[predictions.heldout_block]
        expected=((rows.target_price>=rows.lower)&(rows.target_price<=rows.upper)).mean()
        self.assertAlmostEqual(metrics.coverage.iloc[0],expected)
        width=rows.upper-rows.lower
        interval_score=width+10*np.maximum(rows.lower-rows.target_price,0)+10*np.maximum(rows.target_price-rows.upper,0)
        self.assertAlmostEqual(metrics.mean_interval_score.iloc[0],interval_score.mean())
        self.assertAlmostEqual(metrics.mean_width.iloc[0],width.mean())
        self.assertTrue(bool(metrics.complete_requested_holdout.iloc[0]))
        self.assertTrue((rows.lower_residual_quantile>0).all())


if __name__=="__main__":unittest.main()
