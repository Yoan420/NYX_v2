"""Causal chronology, immutable resume and honest-short-history tests."""
from __future__ import annotations
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import calibration
import study


def data(days=95, horizon=1, final=False):
    rows=[]
    for number in range(days):
        day=pd.Timestamp("2026-01-01")+pd.Timedelta(days=number)
        origin=(day-pd.Timedelta(days=horizon)+pd.Timedelta(hours=8)).tz_localize(study.calibration.TZ).tz_convert("UTC")
        grid=pd.date_range(day.tz_localize("Europe/Paris"),
            (day+pd.Timedelta(days=1)).tz_localize("Europe/Paris"),freq="h",inclusive="left").tz_convert("UTC")
        available=(day+pd.Timedelta(days=1)).tz_localize("Europe/Paris").tz_convert("UTC")
        for hour,stamp in enumerate(grid):
            base=60+10*np.sin(2*np.pi*hour/24)+.03*number
            row={"country":"FR","delivery_utc":stamp.isoformat(),"forecast_origin_utc":origin.isoformat(),
                "horizon_day":horizon,"target_price":base+6+np.sin(number/5),"target_available_utc":available.isoformat(),
                "reference_price":base,"nyx_baseline":base+1,"test2_price":base+3,"storm_price":base+2}
            if final:
                row.update(expert_fr_residual=base+4,expert_cwe_residual=base+5,expert_cwe_absolute=base+2,
                           nyx_current=base)  # FR proposed differs by 4 (<20): frozen reference.
            rows.append(row)
    return pd.DataFrame(rows)


def validated(frame,final=False):
    return calibration.validate_input(frame,calibration.COMPONENTS if final else study.REFERENCE,
        "nyx_current" if final else "nyx_baseline","final_experts" if final else "reference_component_pilot")


def bound(directory,frame,scope="reference_component_pilot"):
    p=Path(directory)/"OOF.csv"
    frame.to_csv(p,index=False)
    p.with_suffix(".manifest.json").write_text(json.dumps({"scope":scope,
        "oof_csv_sha256":study.sha(p),"provenance_status":"conditional_not_certified"}),encoding="utf-8")
    return p


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=validated(data())
        cls.plan=study.split_plan(cls.frame)
        cls.pred,cls.audit=study.generate(cls.frame,study.REFERENCE,{("FR",1)})

    def test_future_label_cannot_change_earlier_predictions_or_selection(self):
        changed=self.frame.copy()
        cutoff=pd.Timestamp(self.plan["selection_cutoff_utc"])
        mask=changed.forecast_origin_utc>=cutoff
        changed.loc[mask,"target_price"]+=1000.
        other,_=study.generate(changed,study.REFERENCE,{("FR",1)})
        columns=[c for c in self.pred if c.startswith("prediction_")]
        before=self.pred.forecast_origin_utc<=cutoff
        np.testing.assert_allclose(self.pred.loc[before,columns],other.loc[before,columns],equal_nan=True)
        _,first=study.evaluate(self.pred,{("FR",1):self.plan})
        _,second=study.evaluate(other,{("FR",1):study.split_plan(changed)})
        self.assertEqual(first[0]["selected"],second[0]["selected"])
        self.assertEqual(first[0]["tuning_scores"],second[0]["tuning_scores"])

    def test_every_fit_label_available_by_origin_and_complete_day(self):
        self.assertTrue(self.audit)
        for audit in self.audit:
            self.assertLessEqual(pd.Timestamp(audit["latest_used_label_available_utc"]),pd.Timestamp(audit["origin"]))
        frame=self.frame.copy()
        day=sorted(frame.delivery_day.unique())[5]
        indices=frame.index[frame.delivery_day==day]
        frame.loc[indices[0],"target_available_utc"]=pd.Timestamp("2027-01-01",tz="UTC")
        origin=frame.loc[frame.delivery_day==sorted(frame.delivery_day.unique())[50],"forecast_origin_utc"].iloc[0]
        eligible=study.eligible_history(frame,origin,sorted(frame.delivery_day.unique())[50])
        self.assertNotIn(day,set(eligible.delivery_day))

    def test_selection_embargo_excludes_delayed_old_label_day(self):
        frame=self.frame.copy()
        tuning_day=pd.Timestamp(self.plan["tuning_days_list"][-1]).date()
        frame.loc[frame.delivery_day==tuning_day,"target_available_utc"]=pd.Timestamp(self.plan["selection_cutoff_utc"])+pd.Timedelta(hours=1)
        plan=study.split_plan(frame)
        self.assertNotIn(str(tuning_day),plan["tuning_days_list"])
        self.assertEqual(plan["tuning_days"],self.plan["tuning_days"]-1)

    def test_storm_cannot_enter_prediction_generation(self):
        changed=self.frame.copy()
        changed["storm_price"]=-999999.
        # No model retraining needed: explicitly deleting Storm is mandatory at entry.
        original=study.convex
        calls=[]
        def wrapped(x,y,loss):
            calls.append(True)
            return original(x,y,loss)
        study.convex=wrapped
        try:
            short=self.frame[self.frame.delivery_day.isin(sorted(self.frame.delivery_day.unique())[:34])]
            altered=changed.loc[short.index]
            a,_=study.generate(short,study.REFERENCE,{("FR",1)})
            b,_=study.generate(altered,study.REFERENCE,{("FR",1)})
            columns=[c for c in a if c.startswith("prediction_")]
            np.testing.assert_allclose(a[columns],b[columns],equal_nan=True)
            self.assertTrue(calls)
        finally:
            study.convex=original

    def test_short_archive_scores_without_training_and_resume_is_hash_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            p=bound(directory,data(days=1,final=True),"final_experts")
            output=Path(directory)/"results"
            study.run(p,output)
            status=json.loads((output/"status.json").read_text())
            self.assertEqual(status["status"],"NOT_READY_DESCRIPTIVE_ONLY")
            self.assertEqual(status["trained_country_horizon_groups"],0)
            self.assertFalse(status["production_qualification"])
            metrics=pd.read_csv(output/"metrics.csv")
            self.assertTrue((metrics.days==1).all())
            self.assertTrue((metrics.period=="available_descriptive").all())
            self.assertEqual(study.run(p,output),output/"report.html")
            (output/"metrics.csv").write_text("corrupted",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"output changed"):
                study.run(p,output)

    def test_manifest_binding_and_frozen_final_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            p=bound(directory,data(days=1,final=True),"final_experts")
            altered=pd.read_csv(p);altered["nyx_current"]+=1
            p=bound(directory,altered,"final_experts")
            with self.assertRaisesRegex(ValueError,"frozen production"):
                study.run(p,Path(directory)/"results")
            p.write_text(p.read_text()+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"hash differs"):
                study.run(p,Path(directory)/"results")

    def test_annual_request_not_relaxed_and_gapped_holdout_not_annual(self):
        with tempfile.TemporaryDirectory() as directory:
            p=bound(directory,data(days=3,final=True),"final_experts")
            output=Path(directory)/"annual"
            study.run(p,output,holdout_days=365)
            status=json.loads((output/"status.json").read_text())
            detail=json.loads((output/"study.json").read_text())
            self.assertEqual(status["status"],"NOT_READY_DESCRIPTIVE_ONLY")
            self.assertEqual(detail["holdout_days"],365)
            self.assertEqual(detail["selections"][0]["minimum_requested_days"],424)
            self.assertEqual(detail["selections"][0]["available_days"],3)
            self.assertFalse(status["production_qualification"])
        missing=sorted(self.frame.delivery_day.unique())[-5]
        gapped=self.frame[self.frame.delivery_day!=missing]
        plan=study.split_plan(gapped)
        self.assertFalse(plan["ready"])
        self.assertEqual(plan["missing_holdout_calendar_days"],1)

    def test_five_challengers_preserve_frozen_final_schema(self):
        extras=("challenger_ridge","challenger_rmse","challenger_mae","challenger_huber","challenger_regime")
        frame=data(days=1,final=True)
        for number,name in enumerate(extras):
            frame[name]=frame.reference_price+number+1.
        with tempfile.TemporaryDirectory() as directory:
            p=bound(directory,frame,"final_experts")
            output=Path(directory)/"challenger_study"
            study.run(p,output,extras=extras)
            result=json.loads((output/"study.json").read_text())
            self.assertEqual(len(result["components"]),9)
            self.assertEqual(result["components"][-5:],list(extras))
            scores=pd.read_csv(output/"metrics.csv")
            self.assertTrue(set("component_"+name for name in extras)<=set(scores.algorithm))
            bad=frame.copy();bad["nyx_current"]+=1.
            p=bound(directory,bad,"final_experts")
            with self.assertRaisesRegex(ValueError,"frozen production"):
                study.run(p,Path(directory)/"bad",extras=extras)
        # The enlarged simplex itself is exercised, beyond the short-input path.
        x=np.tile(np.arange(9,dtype=float),(48,1))+np.arange(48)[:,None]
        for loss in ("mse","mae","huber"):
            weights=study.convex(x,x[:,4]+2.,loss)
            self.assertEqual(len(weights),9)
            self.assertAlmostEqual(weights.sum(),1.)
            self.assertTrue((weights>=0).all())

    def test_gate_thresholds_strict_win_and_no_j2_cap(self):
        current={"rmse":25.,"mae":20.,"storm_rmse":24.,"storm_mae":19.,"win_rate":.52}
        good={"rmse":18.,"mae":12.,"storm_rmse":24.,"storm_mae":19.,"win_rate":.6}
        self.assertTrue(study.gates(good,current,"FR",1)["all_gates_passed"])
        tie={**good,"win_rate":.52}
        self.assertFalse(study.gates(tie,current,"FR",1)["all_gates_passed"])
        self.assertFalse(study.gates({**good,"rmse":19.},current,"FR",1)["all_gates_passed"])
        self.assertTrue(study.gates({**good,"rmse":19.},current,"FR",2)["all_gates_passed"])
        self.assertIsNone(study.gates(good,current,"DE",1)["historical_j1_rmse_cap"])

    def test_identical_matched_samples_and_missing_storm(self):
        pred=self.pred.copy()
        pred.loc[pred.index[::17],"storm_price"]=np.nan
        metrics,selections=study.evaluate(pred,{("FR",1):self.plan})
        for _,part in metrics.groupby(["country","horizon_day","period","matched_storm_only"]):
            self.assertEqual(part.hours.nunique(),1)
        self.assertEqual(len(pred),len(self.pred))
        missing=pred.copy();missing["storm_price"]=np.nan
        _,selections=study.evaluate(missing,{("FR",1):self.plan})
        self.assertEqual(selections[0]["selected"],"current")
        self.assertFalse(selections[0]["benchmark_selection_ready"])

    def test_convex_constraints_and_level_can_escape_hull(self):
        x=np.column_stack([np.arange(40),np.arange(40)+1.,np.arange(40)+2.])
        for loss in ("mse","mae","huber"):
            weights=study.convex(x,np.arange(40)+10.,loss)
            self.assertAlmostEqual(weights.sum(),1.)
            self.assertTrue((weights>=0).all())
        frame=self.pred[self.pred.delivery_day.isin(sorted(self.pred.delivery_day.unique())[-10:])]
        self.assertTrue((frame.prediction_level_60>frame[list(study.REFERENCE)].max(axis=1)).any())

    def test_nine_component_spike_solver_is_stable(self):
        generator=np.random.default_rng(102)
        for scale in (1.,100.,1000.,3000.,10000.):
            for trial in range(20):
                x=generator.normal(size=(720,9))*scale
                y=generator.normal(size=720)*scale
                for loss in ("mse","mae","huber"):
                    weights=study.convex(x,y,loss)
                    self.assertTrue(np.isfinite(weights).all())
                    self.assertAlmostEqual(weights.sum(),1.)
                    self.assertTrue((weights>=0).all())


if __name__=="__main__":
    unittest.main()
