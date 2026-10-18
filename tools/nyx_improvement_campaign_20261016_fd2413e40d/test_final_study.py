import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import final_study
from test_calibration import synthetic


class FinalStudyTest(unittest.TestCase):
    def test_refresh_never_moves_explicit_label_availability_earlier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'OOF.csv';frame=synthetic(days=3)
            frame['target_available_utc']='2026-08-01T12:00:00+00:00'
            frame.to_csv(path,index=False)
            path.with_suffix('.manifest.json').write_text(json.dumps({'scope':'final_experts',
                'oof_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'provenance_status':'conditional'}))
            index=pd.DatetimeIndex(pd.to_datetime(frame.delivery_utc,utc=True))
            target=pd.Series(frame.target_price.to_numpy(),index=index)
            storm=pd.Series(frame.storm_price.to_numpy(),index=index)
            with patch.object(final_study,'seed_record',return_value={}),patch.object(final_study,'report_sources',return_value=({'FR':target},{'FR':storm},{'benchmark':'Storm'})):
                final_study.run(path,root/'study',root)
            attached=pd.read_csv(root/'study/OOF.csv')
            self.assertTrue((pd.to_datetime(attached.target_available_utc,utc=True)==pd.Timestamp('2026-08-01T12:00:00Z')).all())

    def test_short_final_history_scored_without_invented_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'OOF.csv'
            frame=pd.concat([synthetic(days=3,country=c,horizon=h) for c in ['FR','DE','BE','NL'] for h in [1,2]],ignore_index=True)
            frame.to_csv(path,index=False)
            path.with_suffix('.manifest.json').write_text(json.dumps({'scope':'final_experts',
                'oof_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'provenance_status':'conditional'}))
            with patch.object(final_study,'seed_record',return_value={}),patch.object(final_study,'report_sources',return_value=({}, {}, {'benchmark':'unavailable'})),patch.object(final_study.subprocess,'run') as process:
                report=final_study.run(path,root/'study',root,calibrate_when_ready=True)
            self.assertTrue(report.exists())
            process.assert_not_called()
            status=json.loads((root/'study/status.json').read_text())
            self.assertEqual(len(status['adaptive_studies']),2)
            self.assertTrue(all(not x['adaptive_study_ready'] for x in status['adaptive_studies']))
            metrics=pd.read_csv(root/'study/metrics.csv')
            self.assertEqual(set(metrics.horizon_day),{1,2})
            self.assertTrue(metrics.strict_win_rate.notna().all())
            self.assertTrue((metrics.days==3).all())

    def test_csv_tamper_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'OOF.csv';synthetic(days=3).to_csv(path,index=False)
            path.with_suffix('.manifest.json').write_text(json.dumps({'scope':'final_experts','oof_csv_sha256':'0'*64}))
            with self.assertRaisesRegex(ValueError,'differs'):
                final_study.run(path,root/'study',root)

if __name__=='__main__':unittest.main()
