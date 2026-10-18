"""Meaningful research provenance, embargo, physical-hour and resume checks."""
from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import replay as r


class ReplayChecks(unittest.TestCase):
    def frame(self, first, stop=None):
        stop = stop or first+timedelta(days=1)
        index = r.grid(first, stop)
        origin = pd.Timestamp(f"{first-timedelta(days=1)} 08:00", tz="Europe/Paris").tz_convert("UTC")
        return pd.DataFrame({"price_eur_mwh": 50., "reference_eur_mwh": 40.,
            **{f+"_eur_mwh": 60.+j for j,f in enumerate(r.FAMILIES)},
            "forecast_origin_utc": origin,
            "horizon_day": [(d-first).days+1 for d in index.tz_convert("Europe/Paris").date]}, index=index)

    def test_dst_actual_horizons_and_no_j1_j2_relabel(self):
        for first, count in [(date(2026,3,29),23),(date(2026,10,25),25)]:
            frame = self.frame(first)
            result = r.canonical_rows(frame, "FR", pd.Series(48.,index=frame.index), availability_policy="assumed_conservative")
            self.assertEqual(len(result), count)
            self.assertTrue(result.delivery_utc.is_unique)
            frame["horizon_day"] = 2
            with self.assertRaisesRegex(ValueError,"relabeled"):
                r.canonical_rows(frame,"FR",pd.Series(48.,index=frame.index),availability_policy="assumed_conservative")

    def test_receipt_bound_collection_keeps_missing_labels_pending_and_detects_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            daily = root/"annual/runs/a/forecast_extension/daily_refit"
            daily.mkdir(parents=True)
            first = date(2026,10,16)
            frame = self.frame(first,first+timedelta(days=2))
            hashes={}
            for zone in r.ZONES:
                path=daily/f"forecasts/{zone}.parquet";path.parent.mkdir(exist_ok=True)
                frame.to_parquet(path);hashes[f"forecasts/{zone}.parquet"]=r.sha(path)
            r.atomic_json(daily/"receipt.json",{"protocol":"nyx_daily_model_refit_v1","status":"MODELS_COMPLETE",
                "future_labels_used":False,"first_delivery_day":str(first),
                "shared_cutoff_utc":frame.forecast_origin_utc.iloc[0].isoformat(),"artifact_sha256":hashes})
            output=root/"collection"
            report=r.collect(root/"annual",None,output)
            self.assertEqual(report["sealed_forecast_rows"],192)
            self.assertEqual(report["scorable_rows"],0)
            self.assertEqual(report["pending_rows"],192)
            self.assertEqual(report["oof_csv_sha256"],r.sha(output/"OOF.csv"))
            labels=[]
            for zone in r.ZONES:
                for stamp in frame.index[:24]:
                    labels.append({"country":zone,"delivery_utc":stamp,"target_price":55.,
                        "target_available_utc":"2026-10-17T06:00:00Z"})
            label_path=root/"labels.csv";pd.DataFrame(labels).to_csv(label_path,index=False)
            report=r.collect(root/"annual",None,output,label_path)
            self.assertEqual(report["scorable_rows"],96)
            self.assertEqual(report["pending_rows"],96)
            path=daily/"forecasts/FR.parquet";path.write_bytes(path.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError,"changed"):
                r.collect(root/"annual",None,output,label_path)

    def seed_fixture(self,root):
        day=date(2026,10,15);bundle=root/"annual/bundle";bundle.mkdir(parents=True)
        output=root/"annual/report";output.mkdir()
        hashes={}
        for zone in r.ZONES:
            for relative,index in [(f"baseline_history/{zone}.parquet",r.grid(day-timedelta(days=469),day+timedelta(days=1))),
                (f"reference_oof/{zone}.parquet",r.grid(day-timedelta(days=97),day+timedelta(days=1))),
                (f"reference_inputs/base292/{zone}.parquet",r.grid(day-timedelta(days=462),day+timedelta(days=1))),
                (f"features/cwe_residual_2000/{zone}.parquet",r.grid(day-timedelta(days=365),day+timedelta(days=1)))]:
                path=bundle/relative;path.parent.mkdir(parents=True,exist_ok=True)
                pd.DataFrame({"a":1.},index=index).to_parquet(path);hashes[relative]=r.sha(path)
        return {"repo":root/"annual","day":str(day),"bundle":bundle,"output":output,"hashes":hashes}

    def test_835_warmup_does_not_fabricate_365_final_oof(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);seed=self.seed_fixture(root)
            seed["run_path"]=root/"run.json"
            report=r.coverage(seed)
            self.assertEqual(report["conditional_j1_available_days"],7)
            self.assertEqual(report["conditional_j1_days"][0],"2026-10-08")
            with self.assertRaisesRegex(ValueError,"only 7"):
                r.replay(seed,root/"research",days=365,dry_run=True)

    def test_sealed_resume_skips_fits_but_rechecks_source_and_forecast_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);seed=self.seed_fixture(root);seed["run_path"]=root/"run.json"
            raw=root/"source";raw.write_text("frozen")
            inputs={"consumed":{str(raw):r.sha(raw)},"actual":{},"feature_audit":{}}
            support=r.grid(date(2026,10,1),date(2026,10,15))
            inputs["actual"]={z:pd.Series(55.,index=support) for z in r.ZONES}
            calls=[]
            def predictor(inputs,day,destination,threads,progress):
                calls.append(day)
                return {z:self.frame(date.fromisoformat(day)) for z in r.ZONES},{"training_days":365}
            module=types.SimpleNamespace(__file__=str(raw))
            real_import=r.importlib.import_module
            def research_import(name,*args,**kwargs):
                return module if name.startswith("chronos2_hourly.") else real_import(name,*args,**kwargs)
            with patch.object(r,"configure"),patch.object(r.importlib,"import_module",side_effect=research_import):
                result=r.replay(seed,root/"research",days=2,inputs=inputs,predictor=predictor)
                self.assertEqual(result["days"],2);self.assertEqual(len(calls),2)
                r.replay(seed,root/"research",days=2,inputs=inputs,predictor=predictor)
                self.assertEqual(len(calls),2)
                receipt=root/"research/origins/2026-10-14/receipt.json"
                doc=r.read_json(receipt);doc["artifact_sha256"]={};r.atomic_json(receipt,doc)
                with self.assertRaisesRegex(ValueError,"mandatory"):
                    r.replay(seed,root/"research",days=2,inputs=inputs,predictor=predictor)

    def test_inner_training_labels_are_masked_and_365_recipe_retained(self):
        day=date(2026,10,14);support=r.grid(day-timedelta(days=370),day+timedelta(days=2))
        features={family:{z:pd.DataFrame({"feature":1.},index=support) for z in r.ZONES} for family in r.FAMILIES}
        built=types.SimpleNamespace(features=features)
        actual={z:pd.Series(50.,index=support) for z in r.ZONES}
        quantiles={z:pd.DataFrame({"q10":40.,"q50":50.,"q90":60.},index=support) for z in r.ZONES}
        prior={z:pd.DataFrame({"signal":1.},index=support[support<r.grid(day,day+timedelta(days=1))[0]]) for z in r.ZONES}
        inputs={"built":built,"actual":actual,"quantiles":quantiles,"prior":prior}
        checks=[]
        def refs(first,cutoff,built,quantiles,masked,prior,destination,progress):
            current=r.grid(day,day+timedelta(days=2))
            for z in r.ZONES:
                self.assertTrue(masked[z].loc[current].isna().all())
                self.assertTrue((prior[z].index<current[0]).all())
            return {z:pd.Series(48.,index=current) for z in r.ZONES},None,{"current_fit_counts":{"hgb":12,"test2_pairs":2}},None
        def price(features,masked,nyx,origin_day,stop_day,config,model_path,initial_training_day,thread_count):
            self.assertEqual(initial_training_day,str(day-timedelta(days=365)))
            current=r.grid(day,day+timedelta(days=1))
            train=r.grid(day-timedelta(days=365),day)
            for z in r.ZONES:
                self.assertEqual(masked[z].index.tolist(),features[z].index.tolist())
                self.assertTrue(masked[z].loc[current].isna().all())
                self.assertTrue(masked[z].loc[train].notna().all())
            checks.append(config.iterations)
            return {z:pd.DataFrame({"point":52.},index=current) for z in r.ZONES},{"training_days":365,"training_labels_before_origin":True,"Storm_used_as_input":False}
        parent=types.ModuleType("chronos2_hourly");parent.__path__=[]
        refmod=types.SimpleNamespace(fit_daily_reference=refs)
        poolmod=types.SimpleNamespace(PooledConfig=lambda **kw:types.SimpleNamespace(**kw),fit_pooled_block=price)
        livemod=types.SimpleNamespace(compose_price=lambda z,points,ref:points[r.FAMILIES[0]])
        modules={"chronos2_hourly":parent,"chronos2_hourly.nyx_daily_refit_reference":refmod,
            "chronos2_hourly.nyx_pooled_cpu_price_model":poolmod,"chronos2_hourly.nyx_annual_cpu_live":livemod}
        with patch.dict(sys.modules,modules),tempfile.TemporaryDirectory() as folder:
            frames,_=r.predict_origin(inputs,str(day),Path(folder))
        self.assertEqual(checks,[1000,2000,2000])
        self.assertEqual(set(frames),set(r.ZONES))

if __name__=="__main__":
    unittest.main()
