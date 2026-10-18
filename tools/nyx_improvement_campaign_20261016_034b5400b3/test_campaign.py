from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import campaign
from replay import atomic_json, sha
import source_audit
import pandas as pd


class CampaignTests(unittest.TestCase):
    def test_stage_reuses_and_rejects_input_or_output_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'input'; source.write_text('source')
            def generate(out):
                out.mkdir(); (out/'result').write_text('done')
            out = campaign.stage(root, 'test', [source], generate)
            campaign.stage(root, 'test', [source], lambda _: self.fail('recomputed'))
            source.write_text('changed')
            with self.assertRaises(ValueError):campaign.stage(root, 'test', [source], generate)
            source.write_text('source'); (out/'result').write_text('bad')
            with self.assertRaises(ValueError):campaign.stage(root, 'test', [source], generate)

    def test_partial_stage_quarantined_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); src = root/'input'; src.write_text('same')
            old = root/'test'; old.mkdir(); (old/'partial').write_text('old')
            def generate(out):out.mkdir(); (out/'complete').write_text('new')
            campaign.stage(root, 'test', [src], generate)
            self.assertTrue((old/'complete').exists())
            self.assertEqual(len(list((root/'interrupted').glob('test_*/partial'))), 1)

    def test_checkpoint_import_only_sealed_and_no_source_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'original'; campaign_dir=root/'new'; campaign_dir.mkdir()
            replay_dir=old/'final_replay'; replay_dir.mkdir(parents=True)
            seed=root/'seed'; seed.mkdir()
            plan={'seed_run':str(seed),'threads':8}; atomic_json(replay_dir/'replay_plan.json',plan)
            origin=replay_dir/'origins/2026-10-10';origin.mkdir(parents=True)
            (origin/'FR.parquet').write_bytes(b'bytes')
            receipt={'day':'2026-10-10','plan_sha256':sha(replay_dir/'replay_plan.json'),
                     'predictions_sealed_before_scoring':True,'artifact_sha256':{'FR.parquet':sha(origin/'FR.parquet')}}
            atomic_json(origin/'receipt.json',receipt)
            unfinished=replay_dir/'origins/2026-10-11';unfinished.mkdir();(unfinished/'partial').write_text('unfinished')
            before={str(p):sha(p) for p in old.rglob('*') if p.is_file()}
            target=campaign.import_replay(campaign_dir,{'base_session':str(old),'seed_run':str(seed),'threads':8})
            self.assertTrue((target/'origins/2026-10-10/receipt.json').is_file())
            self.assertFalse((target/'origins/2026-10-11').exists())
            self.assertEqual(before,{str(p):sha(p) for p in old.rglob('*') if p.is_file()})

    def test_checkpoint_import_rejects_hash_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';new=root/'new';new.mkdir();seed=root/'seed';seed.mkdir()
            base=old/'final_replay';base.mkdir(parents=True)
            atomic_json(base/'replay_plan.json',{'seed_run':str(seed),'threads':8})
            origin=base/'origins/2026-10-10';origin.mkdir(parents=True)
            (origin/'data').write_text('different')
            atomic_json(origin/'receipt.json',{'day':origin.name,'plan_sha256':sha(base/'replay_plan.json'),
                        'predictions_sealed_before_scoring':True,'artifact_sha256':{'data':'0'*64}})
            with self.assertRaises(ValueError):campaign.import_replay(new,{'base_session':str(old),'seed_run':str(seed),'threads':8})

    def test_audit_describes_missing_numeric_without_imputation(self):
        x=pd.DataFrame({'x':[None,2.,4.]},index=pd.date_range('2026-10-01',periods=3,tz='UTC',freq='h'))
        out=source_audit.describe(x)['fields']['x']
        self.assertEqual(out['missing_rows'],1)
        self.assertEqual(out['finite_numeric_rows'],2)
        self.assertEqual(out['last_value'],4.)

    def test_summary_does_not_claim_annual_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'study';d.mkdir();atomic_json(d/'status.json',{'status':'NOT_READY'})
            (d/'report.html').write_text('result')
            campaign.summary(root)
            doc=json.loads((root/'campaign_status.json').read_text())
            self.assertFalse(doc['production_qualification'])
            self.assertIn('NOT_READY',(root/'index.html').read_text())


if __name__ == '__main__':unittest.main()
