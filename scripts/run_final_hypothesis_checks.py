#!/usr/bin/env python3
"""Predefined, exploratory category-blocked randomization tests for RQ1–RQ3.

The tests concern structural concentration, platform-matched peer divergence,
and comparative temporal volatility. They DO NOT test moderation correctness or
model effectiveness. RQ4 has no independently labelled outcomes and is evaluated separately
using model stability and robustness; do not invent an audit-usefulness p-value.

Each randomization is conditional on within-category exchangeability of the
chosen metric or rates across observed platform slots. This assumption is
substantive and may fail: p-values must be labelled exploratory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'outputs'/'synopsis'
DEST=ROOT/'outputs'/'final'/'hypothesis_checks'
SEED=6402026


def hhi(values):
    s=np.asarray(values,dtype=float)
    if not np.isfinite(s).all() or s.sum()<=0:raise ValueError('Invalid exposure')
    return float(np.square(s/s.sum()).sum())


def cat_groups(df):
    return [np.asarray(ids,int) for ids in df.groupby('category',sort=False).indices.values()]


def permute_within_category(values,groups,rng):
    out=values.copy()
    for ix in groups:
        if len(ix)>1:out[ix]=values[rng.permutation(ix)]
    return out


def peer_signed_rates(rates,groups):
    diffs=np.full(len(rates),np.nan,dtype=float)
    for group in groups:
        for i in group:
            others=group[group!=i]
            if len(others):diffs[i]=rates[i]-np.median(rates[others])
    return diffs


def max_platform_mean_abs(values,platforms):
    frame=pd.DataFrame({'p':platforms,'v':values}).dropna()
    return float(frame.groupby('p')['v'].mean().abs().max())


def platform_volatility(df):
    """Use series-level q90 abs change; series are the effective units, not weeks."""
    required={'platform_name','category','automation_change_abs'}
    if not required.issubset(df.columns):raise ValueError('P3 missing columns')
    v=df.groupby(['platform_name','category'],as_index=False)['automation_change_abs'].quantile(.9)
    v=v.sort_values(['category','platform_name']).reset_index(drop=True)
    return v


def randomization_tests(rq1:pd.DataFrame,rq2:pd.DataFrame,rq3:pd.DataFrame,n_perm=4000,seed=SEED):
    if n_perm<99:raise ValueError('Use >=99 permutations')
    for name,df in [('RQ1',rq1),('RQ2',rq2),('RQ3',rq3)]:
        if df[['platform_name','category']].isna().any().any():raise ValueError(name+' key missing')
    rng=np.random.default_rng(seed)
    # RQ1 rates assigned to the observed volume-bearing platform-category slots.
    # Reconstruct observed segment exposure, which matches frozen HHI.
    rq1=rq1.sort_values(['category','platform_name']).reset_index(drop=True)
    g1=cat_groups(rq1)
    vol=rq1.represented_sors.to_numpy(float)
    rates=rq1.fully_automated_rate.to_numpy(float)
    obs1=hhi(vol*rates)
    null1=np.empty(n_perm)
    for i in range(n_perm):null1[i]=hhi(vol*permute_within_category(rates,g1,rng))

    # RQ2 platform-specific mean signed LOO peer deviation, category-controlled.
    # max across platforms makes this a global test without cherry-picking one.
    rq2=rq2.sort_values(['category','platform_name']).reset_index(drop=True)
    g2=cat_groups(rq2)
    vals=rq2.fully_automated_rate.to_numpy(float)
    platforms=rq2.platform_name.to_numpy()
    obs2=max_platform_mean_abs(peer_signed_rates(vals,g2),platforms)
    null2=np.empty(n_perm)
    for i in range(n_perm):
        shuffled=permute_within_category(vals,g2,rng)
        null2[i]=max_platform_mean_abs(peer_signed_rates(shuffled,g2),platforms)

    # RQ3 measure heterogeneity between platforms in *upper-tail weekly volatility*
    # controlling broad category by permuting series-level q90 labels within cat.
    vol3=platform_volatility(rq3)
    g3=cat_groups(vol3)
    values3=vol3.automation_change_abs.to_numpy(float)
    platform3=vol3.platform_name.to_numpy()
    obs3=max_platform_mean_abs(values3,platform3)
    null3=np.empty(n_perm)
    for i in range(n_perm):
        null3[i]=max_platform_mean_abs(permute_within_category(values3,g3,rng),platform3)

    out=[]
    for label,stat,dist,h0,alt,unit in [
        ('RQ1',obs1,null1,'Within-category automation rates are exchangeable across platform volume slots',
         'Observed segment-level HHI is higher than its category-permuted reference','86 observed platform-category segments'),
        ('RQ2',obs2,null2,'Platform identity is exchangeable within broad category for automation rates',
         'At least one platform has unusually large mean signed peer divergence','86 platform-category segments nested by category/platform'),
        ('RQ3',obs3,null3,'Upper-tail within-series absolute changes are exchangeable across platforms within category',
         'At least one platform has unusually large mean series-level 90th percentile change','50 eligible platform-category temporal series'),
    ]:
        p=(1+int((dist>=stat-1e-12).sum()))/(n_perm+1)
        out.append({'rq':label,'H0':h0,'Ha':alt,'statistic':float(stat),
                    'randomizations':n_perm,'p_one_sided':p,'alpha':.05,
                    'reject_at_0_05':bool(p<.05),
                    'null_p05':float(np.quantile(dist,.05)),
                    'null_median':float(np.median(dist)),
                    'null_p95':float(np.quantile(dist,.95)),
                    'analytical_unit':unit,
                    'caution':'Exploratory label-exchangeability assumption; NOT proof of moderation errors, causation or generalisation'})
    return pd.DataFrame(out)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rq1',type=Path,default=FROZEN/'rq1_segment_exposure.csv')
    ap.add_argument('--rq2',type=Path,default=FROZEN/'rq2_peer_divergence.csv')
    ap.add_argument('--rq3',type=Path,default=FROZEN/'rq3_reliable_weekly.csv')
    ap.add_argument('--output-dir',type=Path,default=DEST)
    ap.add_argument('--permutations',type=int,default=4000)
    args=ap.parse_args()
    x=randomization_tests(pd.read_csv(args.rq1),pd.read_csv(args.rq2),pd.read_csv(args.rq3),n_perm=args.permutations)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    x.to_csv(args.output_dir/'rq1_rq3_exploratory_tests.csv',index=False)
    (args.output_dir/'hypothesis_test_metadata.json').write_text(json.dumps({
        'seed':SEED,'permutations':args.permutations,'exploratory':True,'p_value_type':'Monte Carlo one-sided plus-one',
        'multiple_testing':'Three separately selected exploratory RQs; no confirmatory familywise claims',
        'limitations':['Platform effects may violate exchangeability','Repeated/clustered units not independent',
                       'Descriptive claims do not imply moderation correctness']},indent=2)+'\n')
    print(x[['rq','statistic','p_one_sided','reject_at_0_05','analytical_unit']].to_string(index=False))
    print('NOTE: exploratory conditional tests, not established model accuracy or errors')

if __name__=='__main__':main()
