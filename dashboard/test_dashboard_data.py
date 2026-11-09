"""Run from repository root: python dashboard/test_dashboard_data.py"""
from pathlib import Path
import json,hashlib
import pandas as pd
root=Path(__file__).resolve().parent.parent
s=root/'outputs/synopsis'; v=root/'outputs/final/research_validation'
m=json.loads((s/'summary_metrics.json').read_text())
seg=pd.read_csv(s/'rq1_segment_exposure.csv')
assert len(seg)==m['rq1']['segment_count']==86
assert int(seg.represented_sors.sum())==m['data_validation']['weekly_represented_sors']==430511443
assert len(pd.read_csv(s/'rq2_peer_divergence.csv'))==m['rq2']['comparable_segments']==86
weekly=pd.read_csv(s/'rq3_reliable_weekly.csv')
assert len(weekly)==m['rq3']['reliable_observations']==1203
assert int(weekly.temporal_alert.sum())==m['rq3']['alerts']==30
scored=pd.read_csv(s/'rq4_scored_observations.csv')
assert len(scored)==m['rq4']['eligible_observations']==1130
short=pd.read_csv(s/'rq4_stable_shortlist.csv')
assert len(short)==m['rq4']['stable_anomalies']==18
hold=pd.read_csv(v/'temporal_holdout_scored.csv')
manifest=json.loads((v/'final_validation_manifest.json').read_text())
for row in manifest['training_schedule']:
    sub=hold[hold.fold==row['fold']]
    assert len(sub)==row['test_model_rows']
    assert sub['week_start'].min()==row['min_test_date']
    assert sub['week_start'].max()==row['max_test_date']
    assert int(sub.if_stable_8of10.sum())==row['test_if_stable_8of10_count']
    assert not sub.duplicated(['week_start','platform_name','category']).any()
for rel,expected in json.loads((root/'dashboard/FROZEN_DASHBOARD_INPUT_SHA256.json').read_text()).items():
    p=root/rel
    assert p.exists(),rel
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,rel
print('PASS: frozen research/holdout files and SHA-256 integrity')
