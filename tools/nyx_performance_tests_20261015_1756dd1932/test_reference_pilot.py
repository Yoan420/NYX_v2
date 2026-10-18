from datetime import timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import reference_pilot as pilot
from replay import ZONES, grid, seed_record


class ReferencePilotTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'annual'
        self.run = self.repo / 'runs/nyx_annual_anytime/seed'
        self.bundle = self.run / 'bundle'
        self.output = self.repo / 'runs/seed_reports'
        self.bundle.mkdir(parents=True)
        self.output.mkdir(parents=True)
        self.day = pd.Timestamp('2026-04-01').date()
        self.hashes = {}
        index = grid(self.day-timedelta(days=469), self.day+timedelta(days=1))
        days = index.tz_convert('Europe/Paris').date
        origins = [pd.Timestamp(str(d-timedelta(days=1))+' 08:00', tz='Europe/Paris').tz_convert('UTC') for d in days]
        self.history = {}
        for i, zone in enumerate(ZONES):
            point = 100 + i + 15*np.sin(np.arange(len(index))*.1)
            target = point + 2*np.cos(np.arange(len(index))*.07)
            target[index >= grid(self.day,self.day+timedelta(days=1))[0]] = np.nan
            base = pd.DataFrame({'nyx__q10':point-20,'nyx__q50':point,'nyx__q90':point+20,
                'actual':target,'forecast_origin_utc':origins},index=index)
            self.history[zone] = base
            self.write(f'baseline_history/{zone}.parquet',base)
            subset = base.loc[grid(self.day-timedelta(days=97),self.day+timedelta(days=1))]
            oof = pd.DataFrame({'ensemble__q50':subset.nyx__q50+1,
                'nyx__q50':subset.nyx__q50,'test2__q50':subset.nyx__q50-1},index=subset.index)
            self.write(f'reference_oof/{zone}.parquet',oof)
        source = self.bundle/'source_receipts/saturn.json'
        source.parent.mkdir()
        source.write_text(json.dumps({'price_reference':'EPEX','price_reference_policy':'epex_only_v1',
            'source_mixing':False,'actual_values_imputed':False}))
        self.hashes['source_receipts/saturn.json'] = hashlib.sha256(source.read_bytes()).hexdigest()
        (self.output/'receipt.json').write_text(json.dumps({'status':'REPORTS_COMPLETE','delivery_day':str(self.day),
            'future_labels_used':False,'input_sha256':self.hashes}))
        (self.run/'run.json').write_text(json.dumps({'state':'REPORTS_COMPLETE','operational_snapshot':{'delivery_day':str(self.day)},
            'bundle':str(self.bundle),'output':str(self.output)}))

    def write(self,relative,frame):
        path=self.bundle/relative;path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path)
        self.hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()

    def test_complete_reference_pilot_and_real_calibration_cli(self):
        csv=pilot.export(seed_record(self.repo),self.root/'pilot')
        frame=pd.read_csv(csv)
        self.assertEqual(frame.country.nunique(),4)
        self.assertEqual(frame.groupby('country').delivery_utc.count().unique().tolist(),[97*24-1])
        manifest=json.loads(csv.with_suffix('.manifest.json').read_text())
        self.assertFalse(manifest['final_expert_oof'])
        self.assertEqual(manifest['oof_csv_sha256'],hashlib.sha256(csv.read_bytes()).hexdigest())
        proc=subprocess.run([sys.executable,str(Path(pilot.__file__).parent/'calibration.py'),
            '--input',str(csv),'--output',str(self.root/'study'),'--scope',pilot.SCOPE,
            '--component-columns','reference_price,nyx_baseline,test2_price','--control-column','nyx_baseline',
            '--warmup-days','45','--holdout-days','30'],text=True,capture_output=True)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        metrics=pd.read_csv(self.root/'study/metrics.csv')
        self.assertTrue(metrics.loc[(metrics.period=='30')&~metrics.matched_storm_only,'complete_requested_days'].all())
        self.assertFalse(metrics.loc[metrics.period=='365','complete_requested_days'].any())

    def test_changed_source_is_rejected(self):
        path=self.bundle/'reference_oof/FR.parquet'
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            pilot.export(seed_record(self.repo),self.root/'pilot')

    def test_future_reconstructed_origins_are_excluded(self):
        frame=self.history['FR'].copy()
        index=grid(self.day-timedelta(days=3),self.day-timedelta(days=2))
        frame.loc[index,'forecast_origin_utc']=pd.Timestamp(str(self.day)+' 06:00',tz='UTC')
        self.write('baseline_history/FR.parquet',frame)
        receipt=json.loads((self.output/'receipt.json').read_text());receipt['input_sha256']=self.hashes
        (self.output/'receipt.json').write_text(json.dumps(receipt))
        csv=pilot.export(seed_record(self.repo),self.root/'pilot')
        manifest=json.loads(csv.with_suffix('.manifest.json').read_text())
        self.assertEqual(manifest['days_per_country']['FR']['days'],96)
        self.assertEqual(manifest['excluded_days']['FR'][0]['reason'],'baseline_origin_differs_from_logical_cutoff')

if __name__=='__main__':unittest.main()
