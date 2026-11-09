#!/usr/bin/env python3
"""Leakage-aware retrospective temporal holdout and model robustness validation.

Does not alter the frozen Synopsis/Interim analysis. No ground-truth outcome
labels exist; comparison metrics evaluate stability/ranking, NOT accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
from scipy.stats import spearmanr
import sklearn

ROOT = Path(__file__).resolve().parents[1]
WEEKLY = ROOT / 'data' / 'dsa_weekly_segments.csv'
FROZEN_METRICS = ROOT / 'outputs' / 'synopsis' / 'summary_metrics.json'
FROZEN_RQ4 = ROOT / 'outputs' / 'synopsis' / 'rq4_scored_observations.csv'
DEFAULT_OUT = ROOT / 'outputs' / 'final' / 'research_validation'
KEY = ['week_start','platform_name','category']
SERIES = ['platform_name','category']
FEATURES = [
  'log1p_represented_sors', 'fully_automated_rate',
  'content_removed_rate', 'account_or_service_restriction_rate',
  'weekly_peer_divergence_abs', 'automation_change_abs',
]
SEEDS=[11,22,33,44,55,66,77,88,99,111]
RELIABLE_MIN_VOLUME=1000
RELIABLE_MIN_COMPARISONS=12
STABLE_MIN_SEEDS=8
N_ESTIMATORS=300
TOP_K=20


def as_bool(s:pd.Series)->pd.Series:
    if pd.api.types.is_bool_dtype(s): return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(['true','1','yes'])


def jaccard(a,b)->float:
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a|b else 1.0


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def ensure_base(weekly:pd.DataFrame)->pd.DataFrame:
    needed=set(KEY+['complete_study_week','represented_sors','fully_automated_rate',
                   'content_removed_rate','account_or_service_restriction_rate'])
    missing=needed-set(weekly.columns)
    if missing:raise ValueError(f'Missing weekly columns: {sorted(missing)}')
    weekly=weekly.copy()
    weekly['week_start']=pd.to_datetime(weekly['week_start'])
    if weekly[KEY].isna().any().any() or weekly.duplicated(KEY).any():
        raise ValueError('Missing or duplicated weekly keys')
    weekly=weekly[as_bool(weekly['complete_study_week'])].copy()
    weekly=weekly.sort_values(SERIES+['week_start']).reset_index(drop=True)
    for c in ['represented_sors','fully_automated_rate','content_removed_rate','account_or_service_restriction_rate']:
        if not np.isfinite(pd.to_numeric(weekly[c],errors='coerce').to_numpy(float)).all():
            raise ValueError('Nonfinite base values: '+c)
    if not weekly['represented_sors'].ge(0).all():raise ValueError('Negative volume')
    for c in ['fully_automated_rate','content_removed_rate','account_or_service_restriction_rate']:
        if not weekly[c].between(0,1).all():raise ValueError('Invalid rate: '+c)
    return weekly


def build_candidates(weekly:pd.DataFrame)->pd.DataFrame:
    """Compute prior-week changes from ALL observed complete weeks, before eligibility."""
    weekly=ensure_base(weekly)
    g=weekly.groupby(SERIES,sort=False)
    weekly['previous_week_start']=g['week_start'].shift()
    weekly['previous_week_volume']=g['represented_sors'].shift()
    weekly['automation_change']=g['fully_automated_rate'].diff()
    weekly['automation_change_abs']=weekly['automation_change'].abs()
    weekly['days_since_previous']=(weekly['week_start']-weekly['previous_week_start']).dt.days
    candidate=weekly.loc[
      weekly['represented_sors'].ge(RELIABLE_MIN_VOLUME) &
      weekly['previous_week_volume'].ge(RELIABLE_MIN_VOLUME) &
      weekly['days_since_previous'].eq(7)
    ].copy()
    return candidate


def leave_one_out_sameweek(df:pd.DataFrame)->pd.DataFrame:
    """Peer reference uses peers available within SAME week, not later dates."""
    df=df.copy()
    med=[];counts=[]
    groups=df.groupby(['week_start','category'],sort=False).indices
    peer=np.full(len(df),np.nan,dtype=float)
    support=np.zeros(len(df),dtype=int)
    rates=df['fully_automated_rate'].to_numpy(float)
    for indices in groups.values():
        indices=np.asarray(indices)
        for idx in indices:
            other=indices[indices!=idx]
            support[idx]=len(other)
            if len(other):peer[idx]=np.median(rates[other])
    df['weekly_peer_median_automation_rate']=peer
    df['weekly_peer_support_count']=support
    df['weekly_peer_divergence_abs']=np.abs(rates-peer)
    df['log1p_represented_sors']=np.log1p(df['represented_sors'].to_numpy(float))
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def chronological_features(candidates:pd.DataFrame, weeks_train:int=18):
    """Eligibility and robust baseline fitted strictly on train weeks.

    Eligibility: training period must have >=12 reliable comparisons in series.
    Once eligible, the same series is permitted in later weeks without looking
    ahead at later observations. Same-week peers are available in batch context.
    """
    weeks=sorted(candidates['week_start'].unique())
    # First complete week has no prior comparison; cutoff anchored on observed
    # candidate weeks: 25 candidate weeks, 17 train + 8 holdout at weeks_train=17.
    if len(weeks)<20:raise ValueError('Too few distinct candidate weeks for validation')
    if not 12<=weeks_train<=len(weeks)-4:raise ValueError('Invalid train-week cutoff')
    first_test=weeks[weeks_train]
    tr0=candidates.loc[candidates.week_start.lt(first_test)].copy()
    te0=candidates.loc[candidates.week_start.ge(first_test)].copy()
    count=tr0.groupby(SERIES).size()
    eligible=set(count[count>=RELIABLE_MIN_COMPARISONS].index.tolist())
    tr0=tr0[tr0[SERIES].apply(tuple,axis=1).isin(eligible)].copy()
    te0=te0[te0[SERIES].apply(tuple,axis=1).isin(eligible)].copy()
    if len(eligible)<3 or tr0.empty or te0.empty:
        raise ValueError('Insufficient eligible train series or test records')
    # Train-only medians/MADs; do not use full-period RQ3 robust scores.
    median=tr0.groupby(SERIES)['automation_change'].median()
    mad={}
    for key,g in tr0.groupby(SERIES):
        m=median.loc[key];mad[key]=float(np.median(np.abs(g.automation_change-m)))
    def assign_robust(frame):
        frame=frame.copy()
        scores=[]
        for row in frame.itertuples(index=False):
            key=(row.platform_name,row.category)
            m=median.loc[key];d=mad[key]
            scores.append(0.6745*(row.automation_change-m)/d if d>0 else np.nan)
        frame['robust_change_score_train_baseline']=scores
        frame['temporal_alert_train_baseline']=(
            frame.robust_change_score_train_baseline.abs().ge(3.5) &
            frame.automation_change_abs.ge(.10))
        return frame
    tr=leave_one_out_sameweek(assign_robust(tr0))
    te=leave_one_out_sameweek(assign_robust(te0))
    if tr.empty or te.empty:raise ValueError('No complete-feature rows in train or test')
    assert tr.week_start.max()<te.week_start.min()
    assert set(te[SERIES].apply(tuple,axis=1)).issubset(eligible)
    return tr,te,{'first_test_week':str(pd.Timestamp(first_test).date()),
                  'candidate_train_weeks':weeks_train,
                  'candidate_test_weeks':len(weeks)-weeks_train,
                  'eligible_training_series':len(eligible),
                  'reliable_train_candidate_rows_before_peers':len(tr0),
                  'reliable_test_candidate_rows_before_peers':len(te0),
                  'train_model_rows':len(tr),'test_model_rows':len(te),
                  'train_temporal_alerts':int(tr.temporal_alert_train_baseline.sum()),
                  'test_temporal_alerts':int(te.temporal_alert_train_baseline.sum()),
                  'train_undefined_MAD':int(tr.robust_change_score_train_baseline.isna().sum()),
                  'test_undefined_MAD':int(te.robust_change_score_train_baseline.isna().sum())}


def top_order(s,k):
    return np.argsort(-np.asarray(s,float),kind='stable')[:min(k,len(s))]


def rank_vector(scores):
    # Ordinal ranks with stable ties, for each model independently.
    order=np.argsort(-np.asarray(scores,float),kind='stable')
    r=np.empty(len(order),dtype=int)
    r[order]=np.arange(1,len(order)+1)
    return r


def if_ensemble(train,test,features=FEATURES,clip_feature=None):
    tr=train[features].to_numpy(float).copy()
    te=test[features].to_numpy(float).copy()
    if clip_feature in features:
        idx=features.index(clip_feature)
        lo,hi=np.quantile(tr[:,idx],[0.01,0.99])
        tr[:,idx]=np.clip(tr[:,idx],lo,hi)
        te[:,idx]=np.clip(te[:,idx],lo,hi)
    scaler=RobustScaler().fit(tr)
    xtr=scaler.transform(tr);xte=scaler.transform(te)
    if not np.isfinite(xtr).all() or not np.isfinite(xte).all():
        raise ValueError('Nonfinite feature after scaling')
    s_tr=[];s_te=[];counts=np.zeros(len(test),dtype=int)
    for seed in SEEDS:
        model=IsolationForest(n_estimators=N_ESTIMATORS,contamination=.05,
                              random_state=seed,n_jobs=-1).fit(xtr)
        st=-model.decision_function(xtr);sv=-model.decision_function(xte)
        s_tr.append(st);s_te.append(sv)
        counts[top_order(sv,TOP_K)]+=1
    return np.mean(s_tr,axis=0),np.mean(s_te,axis=0),counts,scaler


def score_models(train,test):
    a=train[FEATURES].to_numpy(float);b=test[FEATURES].to_numpy(float)
    scaler=RobustScaler().fit(a)
    xtr=scaler.transform(a);xte=scaler.transform(b)
    lof=LocalOutlierFactor(n_neighbors=20,novelty=True,metric='minkowski',p=2).fit(xtr)
    svm=OneClassSVM(kernel='rbf',gamma='scale',nu=.05).fit(xtr)
    return {'Local Outlier Factor': -lof.score_samples(xte),
            'One-Class SVM':-svm.decision_function(xte)}


def eval_split(tr,te,meta,include_diagnostics=True):
    if len(te)<TOP_K:raise ValueError('Too few heldout rows for top-20')
    iftrain,ifscore,counts,scaler=if_ensemble(tr,te)
    comps=score_models(tr,te)
    scores={'Isolation Forest':ifscore,**comps}
    ref=set(top_order(ifscore,TOP_K))
    rows=[]
    test=te.copy()
    test['if_priority_score']=ifscore
    test['if_rank']=rank_vector(ifscore)
    test['if_seed_top20_count']=counts
    test['if_stable_8of10']=counts>=STABLE_MIN_SEEDS
    for name,values in scores.items():
        s=set(top_order(values,TOP_K))
        rho=float(spearmanr(rank_vector(ifscore),rank_vector(values)).statistic)
        rows.append({'first_test_week':meta['first_test_week'],
                     'model':name,'train_rows':len(tr),'test_rows':len(te),
                     'top_k':TOP_K,'overlap_with_if_top20':len(s&ref),
                     'jaccard_with_if_top20':jaccard(s,ref),
                     'spearman_with_if_all_test_rows':rho})
        if name!='Isolation Forest':
            safe=name.lower().replace(' ','_').replace('-','_')
            test[safe+'_rank']=rank_vector(values)
    temporal_feature='automation_change_abs'
    _,ablation,_,_=if_ensemble(tr,te,[x for x in FEATURES if x!=temporal_feature])
    _,clipped,_,_=if_ensemble(tr,te,FEATURES,clip_feature=temporal_feature)
    tab=set(top_order(ablation,TOP_K));tclip=set(top_order(clipped,TOP_K))
    sensitivity=[
      {'first_test_week':meta['first_test_week'],'variant':'Remove temporal feature',
       'overlap_with_primary_top20':len(ref&tab),
       'jaccard_with_primary_top20':jaccard(ref,tab),
       'spearman_with_primary_all_test_rows':float(spearmanr(rank_vector(ifscore),rank_vector(ablation)).statistic)},
      {'first_test_week':meta['first_test_week'],'variant':'Clip temporal to train 1st/99th pct',
       'overlap_with_primary_top20':len(ref&tclip),
       'jaccard_with_primary_top20':jaccard(ref,tclip),
       'spearman_with_primary_all_test_rows':float(spearmanr(rank_vector(ifscore),rank_vector(clipped)).statistic)},
    ]
    scaled=np.abs(scaler.transform(te[FEATURES].to_numpy(float)))
    test['largest_train_scaled_deviation_feature']=[FEATURES[i] for i in np.argmax(scaled,axis=1)]
    test['largest_train_scaled_deviation_size']=scaled.max(axis=1)
    test['temporal_ablation_rank']=rank_vector(ablation)
    test['temporal_clipped_rank']=rank_vector(clipped)
    test['in_if_test_top20']=[i in ref for i in range(len(te))]
    test['in_lof_test_top20']=[i in set(top_order(comps['Local Outlier Factor'],TOP_K)) for i in range(len(te))]
    test['in_ocsvm_test_top20']=[i in set(top_order(comps['One-Class SVM'],TOP_K)) for i in range(len(te))]
    for c,variable in [('volume','represented_sors'),('peer','weekly_peer_divergence_abs'),('temporal','automation_change_abs')]:
        rank=rank_vector(te[variable].to_numpy(float));test[f'{c}_baseline_rank']=rank
        test[f'in_{c}_baseline_top20']=rank<=TOP_K
    meta={**meta,'test_if_stable_8of10_count':int((counts>=8).sum()),
          'test_if_all10_count':int((counts==10).sum()),
          'min_train_date':str(tr.week_start.min().date()),'max_train_date':str(tr.week_start.max().date()),
          'min_test_date':str(te.week_start.min().date()),'max_test_date':str(te.week_start.max().date()),
          'train_model_contamination':.05,
          'test_stable_candidate_criteria':'Top20 in >=8 of 10 seeds; no labels'}
    return test,pd.DataFrame(rows),pd.DataFrame(sensitivity),meta


def check_frozen_metrics(metrics:dict, scored:pd.DataFrame,weekly:pd.DataFrame):
    checks={
      'weekly_processed_rows':(metrics['data_validation']['weekly_rows'],2228),
      'full_period_segments':(metrics['data_validation']['summary_rows'],86),
      'represented_SoRs':(metrics['data_validation']['weekly_represented_sors'],430511443),
      'RQ3_reliable_comparisons':(metrics['rq3']['reliable_observations'],1203),
      'RQ4_eligible':(metrics['rq4']['eligible_observations'],1130),
      'RQ4_stable':(metrics['rq4']['stable_anomalies'],18),
      'frozen_scored_csv_rows':(len(scored),1130),
      'frozen_scored_stable':(int(as_bool(scored['stable_anomaly']).sum()),18),
      'base_weekly_csv_rows':(len(weekly),2228),
    }
    out=[]
    for name,(actual,expected) in checks.items():
        out.append({'check':name,'expected':expected,'observed':actual,'pass':bool(actual==expected)})
    if not all(x['pass'] for x in out):raise ValueError('Frozen population mismatch: '+str(out))
    return pd.DataFrame(out)


def run(weekly_csv:Path=WEEKLY, frozen_csv:Path=FROZEN_RQ4,metrics_json:Path=FROZEN_METRICS,
        outdir:Path=DEFAULT_OUT,folds=(17,19),figures=True):
    """25 candidate weeks: 17/8 primary, 19/6 sensitivity.

    With 26 complete Monday-Sunday weeks, comparison weeks begin one week later
    because 7-day changes require a previous week. Primary test July 6-Aug 24.
    """
    weekly=pd.read_csv(weekly_csv)
    frozen=pd.read_csv(frozen_csv)
    metrics=json.loads(metrics_json.read_text())
    checks=check_frozen_metrics(metrics,frozen,weekly)
    cand=build_candidates(weekly)
    if cand.week_start.nunique()!=25:raise ValueError('Expected 25 observed adjacent-week candidate dates')
    outdir.mkdir(parents=True,exist_ok=True)
    frames=[];comps=[];sens=[];metas=[]
    for i,cut in enumerate(folds):
        tr,te,meta=chronological_features(cand,weeks_train=cut)
        scored,comparison,sensitivity,metadata=eval_split(tr,te,meta)
        scored['fold']=i+1
        frames.append(scored);comps.append(comparison);sens.append(sensitivity)
        metas.append({'fold':i+1,**metadata})
        print(f"Fold {i+1}: {metadata['first_test_week']}, train={len(tr)}, holdout={len(te)}, stable={metadata['test_if_stable_8of10_count']}")
    # Primary holdout is first fold.
    first=frames[0].reset_index(drop=True)
    checks.to_csv(outdir/'integrity_checks.csv',index=False)
    pd.concat(frames,ignore_index=True).to_csv(outdir/'temporal_holdout_scored.csv',index=False)
    pd.concat(comps,ignore_index=True).to_csv(outdir/'temporal_holdout_model_comparison.csv',index=False)
    pd.concat(sens,ignore_index=True).to_csv(outdir/'temporal_feature_sensitivity.csv',index=False)
    manifest={'analysis_type':'Exploratory retrospective temporal holdout, NOT supervised test accuracy',
              'source_sha256':{str(p):sha256(p) for p in [weekly_csv,frozen_csv,metrics_json]},
              'training_schedule':metas,'features':FEATURES,'random_seeds':SEEDS,
              'n_estimators':N_ESTIMATORS,'model_metrics':'Heldout ranking overlap, stability and sensitivity only',
              'scaling_fit':'Train-only RobustScaler',
              'peer_data':'Same-week peers in prequalified training-series cohort; batch-scoring assumption',
              'eligibility':'Training-prefix reliability count >=12; never full-period reliability filter',
              'robust_temporal_scores':'Train-only series median and MAD; zero MAD yields undefined',
              'interpretability':'Largest robust-scaled deviation = context, not feature attribution',
              'no_ground_truth':'No error, compliance or audit-usefulness labels; not accuracy',
              'library_versions':{'pandas':pd.__version__,'numpy':np.__version__,
                                  'scikit_learn':sklearn.__version__}}
    (outdir/'final_validation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if figures:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            comp=pd.concat(comps,ignore_index=True)
            pivot=comp.pivot(index='model',columns='first_test_week',values='jaccard_with_if_top20')
            fig,ax=plt.subplots(figsize=(8,4.1))
            pivot.plot(kind='bar',ax=ax,rot=15)
            ax.set_ylabel('Top-20 Jaccard with Isolation Forest')
            ax.set_xlabel('Held-out method')
            ax.set_ylim(0,1.05)
            fig.tight_layout();fig.savefig(outdir/'temporal_holdout_comparison.png',dpi=170)
            plt.close(fig)
            dom=first.loc[first.in_if_test_top20,'largest_train_scaled_deviation_feature'].value_counts()
            fig,ax=plt.subplots(figsize=(8,3.9));dom.plot(kind='barh',ax=ax)
            ax.set_xlabel('Count among holdout IF top-20')
            ax.set_ylabel('Largest absolute train-scaled feature (not attribution)')
            fig.tight_layout();fig.savefig(outdir/'holdout_feature_context.png',dpi=170)
            plt.close(fig)
        except ImportError:
            print('No matplotlib; charts omitted, CSV results saved.')
    print('PASS: frozen metrics, chronological no-leakage checks, two holdout splits')
    print('Generated:',outdir)
    return {'folds':metas,'comparison':pd.concat(comps,ignore_index=True),
            'sensitivity':pd.concat(sens,ignore_index=True)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weekly',type=Path,default=WEEKLY)
    p.add_argument('--frozen',type=Path,default=FROZEN_RQ4)
    p.add_argument('--metrics',type=Path,default=FROZEN_METRICS)
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    args=p.parse_args()
    result=run(args.weekly,args.frozen,args.metrics,args.output_dir)
    print('\nHeld-out model comparison:\n'+result['comparison'].round(4).to_string(index=False))
    print('\nFeature sensitivity:\n'+result['sensitivity'].round(4).to_string(index=False))

if __name__=='__main__':main()
