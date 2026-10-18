#!/usr/bin/env python3
"""Independent Phase 2 output validation; reads frozen data, never retrains or pushes.

Usage: python scripts/validate_interim_eda.py [--root REPO] [--out OUTPUT_DIR]
Exits nonzero for missing outputs, input mutation or failed numerical invariants.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED={"P0":2228,"P1":2083,"P2":86,"P3":1203,"P4":1130,"P5":18,"P6":86,"P7":86}
KEY=['week_start','platform_name','category']
FROZEN_TOTAL=430511443
RATE_MAP={'fully_automated_rate':'fully_auto_sors', 'content_removed_rate':'content_removed_sors',
          'account_or_service_restriction_rate':'account_or_service_restriction_sors'}
FEATURES=['log1p_represented_sors','fully_automated_rate','content_removed_rate',
          'account_or_service_restriction_rate','weekly_peer_divergence_abs','automation_change_abs']

def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path)
    a=p.parse_args();root=a.root.resolve();out=(a.out or root/'outputs/interim/eda').resolve();meta=out/'metadata'
    report=[]
    def ck(i,analysis,desc,expected,actual,ok,severity='ERROR',population='ALL',notes=''):
        report.append(dict(test_id=i,analysis_id=analysis,input_population=population,description=desc,
               expected=str(expected),actual=str(actual),status='PASS' if ok else 'FAIL',severity=severity,notes=notes))
    manpath=meta/'eda_run_manifest.json'
    if not manpath.exists():raise FileNotFoundError('Run EDA stages before validating outputs')
    manifest=json.loads(manpath.read_text());reg=pd.read_csv(meta/'evidence_register.csv').fillna('')
    d={}
    files={"P0":"data/dsa_weekly_segments.csv", "P2":"data/dsa_segment_summary.csv",
          "P3":"outputs/synopsis/rq3_reliable_weekly.csv", "P4":"outputs/synopsis/rq4_scored_observations.csv",
          "P5":"outputs/synopsis/rq4_stable_shortlist.csv", "P6":"outputs/synopsis/rq2_peer_divergence.csv",
          "P7":"outputs/synopsis/rq1_segment_exposure.csv"}
    for name,path in files.items():
        d[name]=pd.read_csv(root/path,low_memory=False)
    d['P1']=d['P0'].loc[d['P0'].complete_study_week.astype(str).str.lower().eq('true')].copy()
    for name,n in EXPECTED.items():ck(f'POP_{name}','D01','Frozen population row count',n,len(d[name]),len(d[name])==n,population=name)
    ck('POP_WEEK_COUNT','D01','Complete study weeks',26,d['P1'].week_start.nunique(),d['P1'].week_start.nunique()==26,population='P1')
    ck('DATA_TOTAL_0','D02','P0 SoRs',FROZEN_TOTAL,int(d['P0'].represented_sors.sum()),int(d['P0'].represented_sors.sum())==FROZEN_TOTAL,population='P0')
    ck('DATA_TOTAL_2','D02','P2 SoRs',FROZEN_TOTAL,int(d['P2'].represented_sors.sum()),int(d['P2'].represented_sors.sum())==FROZEN_TOTAL,population='P2')
    ck('KEY_0','D02','Weekly key duplicates',0,int(d['P0'].duplicated(KEY).sum()),not d['P0'].duplicated(KEY).any(),population='P0')
    ck('KEY_2','D02','Full-period key duplicates',0,int(d['P2'].duplicated(KEY[1:]).sum()),not d['P2'].duplicated(KEY[1:]).any(),population='P2')
    for name in ['P0','P2']:
        df=d[name]
        for rate,num in RATE_MAP.items():
            recomputed=df[num]/df['represented_sors'].replace(0,np.nan)
            ok=np.allclose(recomputed,df[rate],rtol=0,atol=1e-10,equal_nan=True)
            ck(f'RATE_{name}_{rate}','U05','Published rate equals numerator / denominator',True,ok,ok,population=name)
    # Verify frozen key sets and RQ3 temporal selection rules.
    wk3={tuple(x) for x in d['P3'][KEY].itertuples(index=False,name=None)}
    wk4={tuple(x) for x in d['P4'][KEY].itertuples(index=False,name=None)}
    key5={tuple(x) for x in d['P5'][KEY].itertuples(index=False,name=None)}
    ck('RQ4_SUBSET_RQ3','D04','Model eligible observations are reliable changes',True,wk4.issubset(wk3),wk4.issubset(wk3),population='P3/P4')
    ck('RQ4_STABLE_SUBSET','R04','Shortlist subset of scored observations',True,key5.issubset(wk4),key5.issubset(wk4),population='P4/P5')
    p3=d['P3'];adj=(pd.to_datetime(p3.week_start)-pd.to_datetime(p3.previous_week_start)).dt.days
    ok=bool(adj.eq(7).all() and p3.represented_sors.ge(1000).all() and p3.previous_week_volume.ge(1000).all() and p3.reliable_comparisons.ge(12).all())
    ck('RQ3_ADJACENCY','T03','Frozen temporal adjacency, volume and series support',True,ok,ok,population='P3')
    z=p3.robust_change_score.abs().ge(3.5).fillna(False);change=p3.automation_change.abs().ge(.10)
    flag=p3.temporal_alert.astype(str).str.lower().eq('true')
    ck('RQ3_FLAGS','T03','30 alerts satisfy the two frozen criteria',30,int(flag.sum()),bool(flag.eq(z&change).all() and flag.sum()==30),population='P3')
    ck('RQ4_FEATURE_FINITE','M01','All six frozen model inputs finite',True,bool(np.isfinite(d['P4'][FEATURES].to_numpy(float)).all()),
       bool(np.isfinite(d['P4'][FEATURES].to_numpy(float)).all()),population='P4')
    ck('RQ4_STABILITY','R04','18 frozen priority rows have >=8 seeds',True,bool(d['P5'].seed_inclusion_count.ge(8).all()),
       bool(d['P5'].seed_inclusion_count.ge(8).all()),population='P5')
    sourcehashes=manifest.get('input_files_sha256',{})
    for path,original in sourcehashes.items():
        p=root/path; actual=sha256(p) if p.exists() else 'MISSING'
        ck('INPUT_HASH_'+path.replace('/','_'),'D02','Frozen input SHA-256 unchanged',original,actual,original==actual,population='FROZEN_INPUT')
    all_ids=set(reg.analysis_id)
    ck('REGISTER_26','ALL','Exactly 26 approved investigations',26,len(all_ids),len(all_ids)==26)
    for x in reg.itertuples(index=False):
        for rel in str(x.output_paths).split(';'):
            dest=out/rel
            ok=dest.is_file() and dest.stat().st_size>0
            ck('OUTPUT_'+x.analysis_id+'_'+dest.stem,x.analysis_id,'Expected table or chart exists and nonempty',True,ok,ok)
    cat=meta/'figure_catalog.csv'
    if cat.exists():
        figs=pd.read_csv(cat).fillna('')
        for row in figs.itertuples(index=False):
            for rel in [row.output_png,row.output_pdf]:
                fp=out/rel;okay=fp.is_file() and fp.stat().st_size>4000
                ck('FIG_'+row.figure_id+'_'+fp.suffix[1:],'ALL','Figure exists and renders substantial bytes',True,okay,okay)
    else:ck('FIG_CATALOG','ALL','Figure metadata catalog exists',True,False,False)
    # Independent summary reconciliation: no source mutation and correct denominator handling.
    wpath=out/'tables/u05_weighted_vs_unweighted.csv'
    if wpath.exists():
        weighted=pd.read_csv(wpath)
        for name in ['P0','P1','P2']:
            dfr=d[name]
            for rate,num in RATE_MAP.items():
                sub=weighted[(weighted.population_id==name)&(weighted.grouping=='overall')&(weighted.measure==rate)]
                val=dfr[num].sum()/dfr.represented_sors.sum()
                okay=len(sub)==1 and np.isclose(float(sub.iloc[0].sor_weighted_rate),val,rtol=0,atol=1e-12)
                ck('WEIGHT_'+name+'_'+rate,'U05','Aggregate ratio-of-sums matches input',val,
                   float(sub.iloc[0].sor_weighted_rate) if len(sub)==1 else 'MISSING',okay,population=name)
    mix=out/'tables/b05_category_mix.csv'
    if mix.exists():
        df=pd.read_csv(mix);s=df.groupby('platform_name').within_platform_share.sum()
        okay=bool(np.allclose(s.to_numpy(float),1,rtol=0,atol=1e-12))
        ck('CATEGORY_SUMS','B05','Observed within-platform SoR mix sums to one',1,s.to_dict(),okay,population='P2')
    m=out/'tables/m02_feature_correlation.csv'
    if m.exists():
        c=pd.read_csv(m);okay=(len(c)==36 and c.pair_n.eq(1130).all())
        ck('MATRIX_COMPLETE','M02','Full 6-by-6 feature-pair matrix and n',36,len(c),bool(okay),population='P4')
    cm=pd.read_csv(out/'tables/t03_alert_component_matrix.csv')
    total=int(cm.n_rows.sum())
    ck('T03_EXCLUSIVE_PARTITION','T03',
       'Mutually exclusive alert components cover all reliable observations',
       len(d['P3']),total,total==len(d['P3']),population='P3')
    passn=sum(x['status']=='PASS' for x in report);failn=sum(x['status']=='FAIL' for x in report)
    dest=meta/'validation_results.csv';pd.DataFrame(report).to_csv(dest,index=False)
    for ix in reg.index:
        id=reg.loc[ix,'analysis_id']; tests=[x for x in report if x['analysis_id']==id]
        if tests and all(x['status']=='PASS' for x in tests):reg.loc[ix,'status']='VALIDATED'
    reg.to_csv(meta/'evidence_register.csv',index=False)
    manifest['validation_summary']={'status':'PASS' if failn==0 else 'FAIL','checks_passed':passn,'checks_failed':failn,
                                   'evidence':'metadata/validation_results.csv'}
    manifest['generated_outputs_sha256']={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*')
                                        if p.is_file() and p!=manpath}
    manpath.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(f'Validation: {passn} passed, {failn} failed. File: {dest}')
    if failn:
        print('First failures:')
        for row in [x for x in report if x['status']=='FAIL'][:20]:print(' -',row['test_id'],row['actual'])
        raise SystemExit(1)
    print('PHASE 2 EDA VALIDATION: PASSED (pending human review of interpretations)')

if __name__=='__main__':main()
