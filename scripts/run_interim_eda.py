#!/usr/bin/env python3
"""QM640 Phase 2: frozen-input, read-only exploratory analysis.

Usage: python scripts/run_interim_eda.py --stage data|univariate|bivariate|multivariate|temporal|rq|all
Only writes to outputs/interim/eda. Does not download sources, alter inputs or fit an anomaly model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler

FEATURES = ["log1p_represented_sors", "fully_automated_rate", "content_removed_rate",
            "account_or_service_restriction_rate", "weekly_peer_divergence_abs", "automation_change_abs"]
NUMERATORS = {"fully_automated_rate":"fully_auto_sors", "content_removed_rate":"content_removed_sors",
              "account_or_service_restriction_rate":"account_or_service_restriction_sors"}
WEEK_KEY = ["week_start", "platform_name", "category"]
SEG_KEY = ["platform_name", "category"]
OTHER = "STATEMENT_CATEGORY_OTHER_VIOLATION_TC"
STAGES = ["data", "univariate", "bivariate", "multivariate", "temporal", "rq"]
GROUPS = {"data": ["D01", "D02", "D03", "D04"], "univariate": [f"U{i:02d}" for i in range(1,6)],
          "bivariate": [f"B{i:02d}" for i in range(1,6)], "multivariate": [f"M{i:02d}" for i in range(1,5)],
          "temporal": [f"T{i:02d}" for i in range(1,5)], "rq": [f"R{i:02d}" for i in range(1,5)]}
POP = {"P0": "All weekly aggregate records; boundary weeks clipped to March–August in-period dates",
       "P1": "Complete Monday–Sunday study weeks only",
       "P2": "Full-period platform × category segments", "P3": "Reliable adjacent weekly changes, frozen RQ3",
       "P4": "Six-feature-complete weekly observations, frozen RQ4", "P5": "Stable Isolation Forest audit shortlist, frozen RQ4",
       "P6": "Full-period peer-divergence segments", "P7": "Full-period exposure/concentration segments"}
EXTRA_INPUTS = ["metadata/source_manifest.csv", "data/data_dictionary.csv", "outputs/synopsis/analysis_data_dictionary.csv",
                "outputs/synopsis/summary_metrics.json", "outputs/synopsis/data_missingness_report.csv",
                "outputs/synopsis/rq3_temporal_alerts.csv", "outputs/synopsis/rq4_baseline_comparison.csv",
                "outputs/synopsis/rq4_feature_set_sensitivity.csv", "outputs/synopsis/rq4_topk_sensitivity.csv",
                "outputs/synopsis/rq4_stability_threshold_sensitivity.csv", "outputs/synopsis/rq4_candidate_threshold_sensitivity.csv"]
INPUTS = {"P0":"data/dsa_weekly_segments.csv", "P1":"data/dsa_weekly_segments.csv",
          "P2":"data/dsa_segment_summary.csv", "P3":"outputs/synopsis/rq3_reliable_weekly.csv",
          "P4":"outputs/synopsis/rq4_scored_observations.csv", "P5":"outputs/synopsis/rq4_stable_shortlist.csv",
          "P6":"outputs/synopsis/rq2_peer_divergence.csv", "P7":"outputs/synopsis/rq1_segment_exposure.csv"}
# Each row: approved ID, description, priority, population, research-question link, primary evidence files
SPEC = [
("D01","Grain, counts, types, scope", "C","P0/P1/P2/P3/P4/P5","All","tables/d01_population_profile.csv;tables/d01_schema_audit.csv"),
("D02","Provenance and integrity reconciliation","C","P0/P2","All","tables/d02_quality_reconciliation.csv;tables/d02_provenance_audit.csv"),
("D03","Base missingness and structural unavailability","C","P0/P2/P3/P4","RQ2/RQ3/RQ4","tables/d03_missingness_by_population.csv;tables/d03_structural_unavailability.csv"),
("D04","Coverage and eligibility","C","P0/P1/P3/P4","All","tables/d04_coverage_and_eligibility.csv;tables/d04_exclusion_reasons.csv;figures/d04_coverage_eligibility.png"),
("U01","Univariate numerical summary","C","P0/P2/P4","All","tables/u01_descriptive_statistics.csv"),
("U02","SoR-volume tail","C","P0/P1","RQ1/RQ4","tables/u02_volume_tail.csv;figures/u02_volume_distribution.png"),
("U03","Bounded core rates and zero/one mass","C","P0/P4","RQ1/RQ4","tables/u03_boundary_mass.csv;figures/u03_rate_distributions.png"),
("U04","Platform/category representation","C","P0/P2","RQ1/RQ2","tables/u04_composition.csv;figures/u04_platform_category_composition.png"),
("U05","Weighted versus unweighted rates","C","P0/P1/P2","All","tables/u05_weighted_vs_unweighted.csv;figures/u05_weighting_comparison.png"),
("B01","Volume versus automation rate","C","P0/P1/P2","RQ1/RQ4","tables/b01_association_summaries.csv;figures/b01_volume_vs_automation.png"),
("B02","Automation versus enforcement rates","C","P4","RQ4","figures/b02_automation_enforcement.png"),
("B03","Platform/category automation distributions","C","P1/P2","RQ1/RQ2","tables/b03_category_rate_support.csv;figures/b03_platform_category_rates.png"),
("B04","Peer deviation versus support and volume","C","P6","RQ2","tables/b04_peer_support_summary.csv;figures/b04_peer_support_divergence.png"),
("B05","Platform-specific category composition","S","P2","RQ1/RQ2","tables/b05_category_mix.csv;figures/b05_category_mix.png"),
("M01","Six-feature population profile","C","P4","RQ4","tables/m01_rq4_feature_profile.csv"),
("M02","Six-feature correlation/dependence","C","P4","RQ4","tables/m02_feature_correlation.csv;figures/m02_feature_dependence.png"),
("M03","Robust scaling and transformation diagnostics","C","P4","RQ4","tables/m03_scaling_diagnostics.csv;figures/m03_transformation_checks.png"),
("M04","Joint features and stratified associations","S","P4","RQ4","tables/m04_stratified_dependence.csv;figures/m04_joint_feature_profiles.png"),
("T01","Complete-week volume and rate trends","C","P1","RQ1/RQ3","tables/t01_weekly_rate_volume.csv;figures/t01_weekly_trends.png"),
("T02","Temporal change distribution and MAD","C","P3","RQ3","tables/t02_series_volatility.csv;figures/t02_change_distribution.png"),
("T03","Eligibility and alert-component diagnostics","C","P0/P1/P3","RQ3/RQ4","tables/t03_alert_eligibility.csv;tables/t03_alert_component_matrix.csv"),
("T04","Selected alert neighbourhoods","S","P1/P3","RQ3","tables/t04_alert_context.csv;figures/t04_alert_context.png"),
("R01","Exposure concentration and OTHER context","C","P2/P7","RQ1","tables/r01_other_context.csv;figures/r01_exposure_concentration.png"),
("R02","Signed divergence and peer context","C","P6","RQ2","tables/r02_divergence_context.csv;figures/r02_peer_divergence.png"),
("R03","All reliable alert explanations","C","P3","RQ3","tables/r03_alert_explanations.csv"),
("R04","Stable shortlist context and frozen sensitivities","S","P4/P5","RQ4","tables/r04_shortlist_context.csv;tables/r04_frozen_sensitivity.csv;figures/r04_shortlist_feature_context.png")]
assert len(SPEC)==26


def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda:fh.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def label_category(s):
    return str(s).removeprefix('STATEMENT_CATEGORY_').replace('_VIOLATION_TC', ' OTHER').replace('_', ' ').title()


def grp_rate(df, keys, numerator):
    if not keys: return float(df[numerator].sum()/df['represented_sors'].sum()) if df['represented_sors'].sum() else np.nan
    d=df.groupby(keys,dropna=False,observed=True)[[numerator,'represented_sors']].sum().reset_index()
    d['weighted_rate']=d[numerator]/d['represented_sors'].replace(0,np.nan)
    return d


def describe(x):
    s=pd.to_numeric(pd.Series(x),errors='coerce');v=s.dropna(); n=len(v)
    def vfloat(z):return float(z) if pd.notna(z) else np.nan
    return {'n':len(s),'valid_n':n,'missing_n':int(s.isna().sum()),
            'mean':vfloat(v.mean()),'median':vfloat(v.median()),'std':vfloat(v.std()),
            'q05':vfloat(v.quantile(.05)),'q25':vfloat(v.quantile(.25)),
            'q75':vfloat(v.quantile(.75)),'q95':vfloat(v.quantile(.95)),
            'q99':vfloat(v.quantile(.99)), 'iqr':vfloat(v.quantile(.75)-v.quantile(.25)),
            'min':vfloat(v.min()),'max':vfloat(v.max()),'zero_count':int((v==0).sum()),
            'one_count':int((v==1).sum()),'unique_values':int(v.nunique()),
            'skew':vfloat(v.skew()) if n>2 else np.nan,
            'iqr_outlier_count':int(((v< v.quantile(.25)-1.5*(v.quantile(.75)-v.quantile(.25))) |
                                      (v> v.quantile(.75)+1.5*(v.quantile(.75)-v.quantile(.25)))).sum())
                         if n>0 and v.quantile(.75)>v.quantile(.25) else np.nan}


def spearman_checked(a,b):
    xy=pd.concat([pd.to_numeric(pd.Series(a),errors='coerce').reset_index(drop=True),
                  pd.to_numeric(pd.Series(b),errors='coerce').reset_index(drop=True)],axis=1).dropna()
    if len(xy)<3 or xy.iloc[:,0].nunique()<2 or xy.iloc[:,1].nunique()<2: return np.nan,len(xy)
    return float(stats.spearmanr(xy.iloc[:,0],xy.iloc[:,1]).statistic),len(xy)


class EDA:
    def __init__(self, root:Path, out:Path):
        self.root=root.resolve();self.out=out.resolve();self.out.mkdir(parents=True,exist_ok=True)
        for p in ('figures','tables','metadata'):(self.out/p).mkdir(exist_ok=True)
        self.figures=[]
        self.data={}
        for k,v in INPUTS.items():
            if k=='P1':continue
            p=self.root/v
            if not p.exists():raise FileNotFoundError(f'Missing frozen input: {p}')
            self.data[k]=pd.read_csv(p,low_memory=False)
        self.data['P1']=self.data['P0'].loc[self.data['P0']['complete_study_week'].astype(str).str.lower().eq('true')].copy()
        self.manifest=pd.read_csv(self.root/'metadata/source_manifest.csv')
        self.metrics=json.loads((self.root/'outputs/synopsis/summary_metrics.json').read_text())
        for name in ('P0','P1','P2','P3','P4','P5','P6','P7'):
            if name in ('P0','P1','P3','P4','P5','P6','P7'):
                if 'week_start' in self.data[name]:self.data[name]['week_start']=pd.to_datetime(self.data[name]['week_start'])
            if 'complete_study_week' in self.data[name]:self.data[name]['complete_study_week']=self.data[name]['complete_study_week'].astype(str).str.lower().eq('true')
        self.initial_checks()
        self.setup_register()

    def initial_checks(self):
        expected={'P0':2228,'P1':2083,'P2':86,'P3':1203,'P4':1130,'P5':18,'P6':86,'P7':86}
        for p,n in expected.items():
            if len(self.data[p])!=n:raise ValueError(f'INPUT BASELINE FAILED: {p} rows {len(self.data[p])} != {n}')
        if int(self.data['P0'].represented_sors.sum())!=430511443:raise ValueError('Input represented SoR total mismatch')
        if int(self.data['P2'].represented_sors.sum())!=430511443:raise ValueError('Segment total mismatch')
        if self.data['P0'].duplicated(WEEK_KEY).any() or self.data['P2'].duplicated(SEG_KEY).any():raise ValueError('Duplicate analytical key')
        if len(self.manifest)!=6 or not self.manifest.checksum_status.astype(str).eq('verified').all():raise ValueError('Source manifest does not record 6 checksums verified')
        if self.data['P4'][FEATURES].isna().any().any() or not np.isfinite(self.data['P4'][FEATURES].to_numpy(float)).all():raise ValueError('P4 features not finite')

    def setup_register(self):
        path=self.out/'metadata/evidence_register.csv';old={}
        if path.exists():
            prior=pd.read_csv(path).fillna('');old={x.analysis_id:x for x in prior.itertuples(index=False)}
        rows=[]
        for id,title,priority,pop,rq,outputs in SPEC:
            was=old.get(id)
            rows.append(dict(analysis_id=id,theme={'D':'Data quality','U':'Univariate','B':'Bivariate','M':'Multivariate','T':'Temporal','R':'RQ context'}[id[0]],
                             priority=priority,analysis_population=pop,source_paths=';'.join(sorted({INPUTS[p] for p in pop.split('/') if p in INPUTS})),
                             metric_or_diagnostic=title,output_paths=outputs,rq_link=rq,
                             validation_rule='Named outputs exist; declared population and frozen invariants reconciled',
                             status=getattr(was,'status','SPECIFIED'),review_notes=getattr(was,'review_notes','')))
        pd.DataFrame(rows).to_csv(path,index=False)

    def csv(self,rel,rows):
        dest=self.out/'tables'/rel
        pd.DataFrame(rows).to_csv(dest,index=False,float_format='%.15g')
        print(f'  saved {dest.relative_to(self.out)} ({len(rows)} rows)')

    def savefig(self,id,analysis_ids,source_table,pop,chart_type,rq,title,note,main=False):
        plt.gcf().suptitle(title,fontsize=13,y=0.995)
        plt.gcf().text(.015,.008,f'{pop}: {note}',fontsize=7.5,va='bottom',color='#555555')
        plt.tight_layout(rect=[0,.05,1,.95])
        png=self.out/'figures'/f'{id}.png'; pdf=self.out/'figures'/f'{id}.pdf'
        plt.savefig(png,dpi=320,bbox_inches='tight');plt.savefig(pdf,bbox_inches='tight');plt.close()
        self.figures.append(dict(figure_id=id,analysis_ids=analysis_ids,source_table=source_table,
            platform_category_week_scope=pop,chart_type=chart_type,main_or_appendix='CANDIDATE_MAIN' if main else 'APPENDIX',rq_link=rq,
            title=title,caption_note=note,output_png=f'figures/{id}.png',output_pdf=f'figures/{id}.pdf',status='GENERATED'))

    def group_wide(self,df,group):
        result=[]
        for key,part in df.groupby(group,dropna=False,observed=True):
            keys=key if isinstance(key,tuple) else (key,)
            result.append({**dict(zip(group,keys)), 'n_analytical_rows':len(part), 'represented_sors':int(part.represented_sors.sum()),
                           'fully_auto_sors':int(part.fully_auto_sors.sum()),
                           'fully_automated_rate_weighted':float(part.fully_auto_sors.sum()/part.represented_sors.sum()) if part.represented_sors.sum() else np.nan,
                           'fully_automated_rate_unweighted_mean':float(part.fully_automated_rate.mean())})
        return pd.DataFrame(result)

    def valid_pairs(self):
        """Audit P1 -> RQ3 eligibility without redefining frozen outputs."""
        p1=self.data['P1'].sort_values(WEEK_KEY)
        allchecks=[]
        for (p,c),g in p1.groupby(SEG_KEY,sort=True):
            g=g.sort_values('week_start')
            previous=None; per=[]
            for x in g.to_dict('records'):
                state=dict(week_start=x['week_start'],platform_name=p,category=c,represented_sors=x['represented_sors'],prev_week_volume=np.nan,
                           status='FIRST_OBSERVATION', delta_days=np.nan)
                if previous is not None:
                    days=(x['week_start']-previous['week_start']).days
                    state['delta_days']=days;state['prev_week_volume']=previous['represented_sors']
                    if days!=7:state['status']='NON_ADJACENT'
                    elif min(x['represented_sors'],previous['represented_sors'])<1000:state['status']='LOW_VOLUME'
                    else:state['status']='CANDIDATE'
                per.append(state);previous=x
            qualifying=sum(x['status']=='CANDIDATE' for x in per)
            for state in per:
                if state['status']=='CANDIDATE':state['status']='RQ3_ELIGIBLE' if qualifying>=12 else 'SERIES_LT_12'
                state['n_reliable_comparisons']=qualifying
                allchecks.append(state)
        return pd.DataFrame(allchecks)

    def data_stage(self):
        print('DATA D01–D04')
        P=self.data;p0=P['P0'];p2=P['P2'];p1=P['P1'];p3=P['P3'];p4=P['P4']
        self.csv('d01_population_profile.csv',[{'population_id':k,'grain':'week × platform × category' if k not in ('P2','P6','P7') else 'platform × category',
                    'n_analytical_rows':len(df),'represented_sors':int(df.represented_sors.sum()),
                    'distinct_platforms':df.platform_name.nunique(),'distinct_categories':df.category.nunique(),
                    'distinct_weeks':df.week_start.nunique() if 'week_start' in df else np.nan,'definition':POP[k]}
                    for k,df in P.items()])
        self.csv('d01_schema_audit.csv',[{'population_id':k,'variable':v,'dtype':str(dt),'n_missing':int(P[k][v].isna().sum()),
                                         'n_distinct':int(P[k][v].nunique(dropna=True))} for k in ('P0','P2','P3','P4') for v,dt in P[k].dtypes.items()])
        checks=[]
        def check(name,actual,expected=None):checks.append(dict(check=name,actual=actual,expected=expected,status='PASS' if expected is None or actual==expected else 'FAIL'))
        check('weekly unique keys',int(p0.duplicated(WEEK_KEY).sum()),0);check('segment unique keys',int(p2.duplicated(SEG_KEY).sum()),0)
        check('weekly represented SoRs',int(p0.represented_sors.sum()),430511443);check('summary represented SoRs',int(p2.represented_sors.sum()),430511443)
        check('source-manifest represented SoRs',int(self.manifest.represented_sors.sum()),430511443)
        for nm,df in [('weekly',p0),('summary',p2)]:
            for col in df.filter(regex='_sors$').columns:
                check(f'{nm} negative {col}',int((df[col]<0).sum()),0)
                if col!='represented_sors':check(f'{nm} numerator > denominator {col}',int((df[col]>df.represented_sors).sum()),0)
            for col in df.filter(regex='_rate$').columns:check(f'{nm} invalid {col}',int((~df[col].between(0,1) & df[col].notna()).sum()),0)
            autos=df[['fully_auto_sors','partially_auto_sors','not_auto_sors']].sum(axis=1)
            check(f'{nm} automation decisions exceed denominator',int((autos>df.represented_sors).sum()),0)
            checks.append(dict(check=f'{nm} unclassified automation remainder rows',actual=int((autos<df.represented_sors).sum()),expected='informational',status='INFO'))
        self.csv('d02_quality_reconciliation.csv',checks)
        self.csv('d02_provenance_audit.csv',self.manifest[['source_month','source_filename','source_url','sha1_url','official_sha1','downloaded_sha1','checksum_status','selected_aggregate_rows','represented_sors']])
        bad=[x for x in checks if x['status']=='FAIL']
        if bad:raise ValueError('Source-domain validity checks failed: '+repr(bad[:4]))
        self.csv('d03_missingness_by_population.csv',[dict(population_id=k,variable=v,missing_n=int(df[v].isna().sum()),
                                                          missing_pct=float(df[v].isna().mean()*100),n_analytical_rows=len(df))
                       for k,df in P.items() for v in df.columns])
        eligibility=self.valid_pairs();r3keys=set(map(tuple,p3[WEEK_KEY].to_numpy()));r4keys=set(map(tuple,p4[WEEK_KEY].to_numpy()))
        identified=set(map(tuple,eligibility.loc[eligibility.status=='RQ3_ELIGIBLE',WEEK_KEY].to_numpy()))
        if r3keys!=identified:raise ValueError('Rebuilt RQ3 eligibility keys diverge from frozen P3')
        if not r4keys.issubset(r3keys):raise ValueError('Frozen P4 not subset of P3')
        counts=eligibility.status.value_counts()
        extra=[]
        for x in eligibility.itertuples(index=False):
            if x.status=='RQ3_ELIGIBLE':
                extra.append({**x._asdict(),'rq4_eligible':(x.week_start,x.platform_name,x.category) in r4keys})
        peer=pd.DataFrame(extra)
        nopeer=int((~peer.rq4_eligible).sum())
        # RQ4 same-week/category peer support is drawn from P3, never from unrelated P0 rows.
        pool_size=p3.groupby(['week_start','category']).platform_name.transform('nunique')
        support_keys={tuple(v) for v in p3.loc[pool_size>1,WEEK_KEY].itertuples(index=False,name=None)}
        no_peer_support=int((pool_size<=1).sum())
        other_missing=len(support_keys-r4keys)
        if not r4keys.issubset(support_keys):
            raise ValueError('Frozen RQ4 includes a row with no P3 same-week/category peer support')
        if no_peer_support+other_missing!=nopeer:
            raise ValueError('RQ4 exclusion reasons do not reconcile')
        unavail=[dict(component='P1 to reliable RQ3',reason=why,observation_count=int(cnt),interpretation='Eligibility, not necessarily missing source data') for why,cnt in counts.items()]
        unavail+=[dict(component='P3 to eligible RQ4',reason='NO_SAME_WEEK_CATEGORY_ELIGIBLE_PEER',observation_count=no_peer_support,interpretation='Absence of another P3 reporting platform; not source missingness'),
                  dict(component='P3 to eligible RQ4',reason='OTHER_REQUIRED_FEATURE_UNAVAILABLE',observation_count=other_missing,interpretation='Feature availability, if any'),
                  dict(component='P3 robust score',reason='UNDEFINED_MAD_ZERO',observation_count=int(p3.robust_change_score.isna().sum()),interpretation='Undefined robust z, not a corrupt or zero score')]
        self.csv('d03_structural_unavailability.csv',unavail)
        scope=[]
        for platform in sorted(p0.platform_name.unique()):
            for category in sorted(p0.category.unique()):
                row={'platform_name':platform,'category':category}
                for pname,df in [('P0',p0),('P1',p1),('P3',p3),('P4',p4)]:
                    sub=df[(df.platform_name==platform)&(df.category==category)]
                    row[pname+'_rows']=len(sub);row[pname+'_represented_sors']=int(sub.represented_sors.sum())
                    row[pname+'_distinct_weeks']=sub.week_start.nunique()
                row['P1_observed']=row['P1_rows']>0
                row['P3_vs_P1_row_retention_pct']=100*row['P3_rows']/row['P1_rows'] if row['P1_rows'] else np.nan
                row['P4_vs_P3_row_retention_pct']=100*row['P4_rows']/row['P3_rows'] if row['P3_rows'] else np.nan
                scope.append(row)
        cover=pd.DataFrame(scope)
        self.csv('d04_coverage_and_eligibility.csv',cover)
        r3r4_peer=p3[WEEK_KEY].merge(p4[WEEK_KEY].assign(in_p4=True),how='left',on=WEEK_KEY)
        summary=[]
        for status,cnt in counts.items():summary.append({'transition':'P1 -> P3','reason':status,'rows':int(cnt)})
        summary.extend([{'transition':'P3 -> P4','reason':'IN_P4','rows':int(r3r4_peer.in_p4.fillna(False).sum())},
                        {'transition':'P3 -> P4','reason':'NO_SAME_WEEK_CATEGORY_ELIGIBLE_PEER','rows':no_peer_support},
                        {'transition':'P3 -> P4','reason':'OTHER_REQUIRED_FEATURE_UNAVAILABLE','rows':other_missing},
                        {'transition':'P0 -> P1','reason':'BOUNDARY_OR_INCOMPLETE_WEEK','rows':len(p0)-len(p1)}])
        self.csv('d04_exclusion_reasons.csv',summary)
        fig,(ax,ax2)=plt.subplots(1,2,figsize=(14,5.8),gridspec_kw={'width_ratios':[2.2,1]})
        observed=cover.pivot(index='platform_name',columns='category',values='P1_distinct_weeks')
        chart=observed.to_numpy(float); chart[chart==0]=np.nan
        im=ax.imshow(np.ma.masked_invalid(chart),vmin=0,vmax=26,aspect='auto',cmap='viridis')
        ax.set_yticks(range(len(observed.index)),observed.index)
        ax.set_xticks(range(len(observed.columns)),[label_category(c) for c in observed.columns],rotation=90,fontsize=7)
        ax.set_title('Observed full weeks by platform-category (blank = unobserved)')
        fig.colorbar(im,ax=ax,fraction=.04,pad=.02,label='Observed weeks')
        sizes=[len(P[k]) for k in ('P0','P1','P3','P4','P5')]
        ax2.barh(['P0','P1','P3','P4','P5'][::-1],sizes[::-1],color='steelblue')
        for i,num in enumerate(sizes[::-1]):ax2.text(num+16,i,f'{num:,}',va='center')
        ax2.set_xlim(0,max(sizes)*1.25);ax2.set_title('Different analytical populations')
        self.savefig('d04_coverage_eligibility','D04','tables/d04_coverage_and_eligibility.csv','P0/P1/P3/P4/P5',
                     'coverage heatmap and count bars','All','Observation support and RQ eligibility','Blank keys mean not observed, not zero activity; attrition rules differ.',main=True)

    def univariate_stage(self):
        print('UNIVARIATE U01–U05')
        P=self.data;p0=P['P0'];p1=P['P1'];p2=P['P2'];p4=P['P4']
        rows=[]
        for name,df in [('P0',p0),('P2',p2),('P4',p4)]:
            cols=FEATURES if name=='P4' else list(df.select_dtypes(include=np.number).columns)
            for col in cols:rows.append(dict(population_id=name,measure=col,grain=POP[name],**describe(df[col])))
        self.csv('u01_descriptive_statistics.csv',rows)
        tails=[]
        for name,df in [('P0',p0),('P1',p1)]:
            x=df.represented_sors
            for top in [1,5,10,20,50]:tails.append(dict(population_id=name,top_n=top,soRs_top=int(x.nlargest(top).sum()),
                                     total_sors=int(x.sum()),volume_share_top=float(x.nlargest(top).sum()/x.sum())))
        self.csv('u02_volume_tail.csv',tails)
        fig,axs=plt.subplots(1,2,figsize=(12,4.5))
        axs[0].hist(np.log1p(p0.represented_sors),bins=40)
        axs[0].set(xlabel='ln(1 + represented SoRs)',ylabel='Weekly analytical rows',title='All weekly rows P0')
        axs[1].hist(np.log1p(p1.represented_sors),bins=40)
        axs[1].set(xlabel='ln(1 + represented SoRs)',ylabel='Complete-week rows',title='Comparable full weeks P1')
        self.savefig('u02_volume_distribution','U02','tables/u02_volume_tail.csv','P0/P1','log-volume histograms','RQ1/RQ4',
                     'Represented SoR volume: distribution and tail','Each bar counts aggregated rows, not individual SoRs.',main=True)
        rates=list(NUMERATORS)
        boundary=[]
        for name,df in [('P0',p0),('P4',p4)]:
            for col in rates:
                d=describe(df[col]);boundary.append(dict(population_id=name,measure=col,n=len(df),
                    exact_zeros=d['zero_count'],exact_ones=d['one_count'],zero_pct=d['zero_count']/len(df)*100,
                    one_pct=d['one_count']/len(df)*100,median=d['median'],iqr=d['iqr']))
        self.csv('u03_boundary_mass.csv',boundary)
        fig,axs=plt.subplots(2,3,figsize=(15,7),sharex=True)
        for i,(pname,df) in enumerate([('P0',p0),('P4',p4)]):
            for j,f in enumerate(rates):
                axs[i,j].hist(df[f],bins=np.linspace(0,1,31))
                axs[i,j].set(xlim=(0,1),xlabel='Rate [0, 1]',ylabel='Analytical rows',title=f'{pname}: {f.replace("_rate", "").replace("_", " ")}')
        self.savefig('u03_rate_distributions','U03','tables/u03_boundary_mass.csv','P0/P4',
                     'bounded histograms','RQ1/RQ4','Bounded distributions of fixed moderation-rate features',
                     'Exact 0 and 1 values are valid reported aggregate rates.',main=True)
        composition=[]
        for gran,df in [('platform_name',p0),('category',p0),('platform_name',p2),('category',p2)]:
            for v,sub in df.groupby(gran):composition.append(dict(population_id='P0' if df is p0 else 'P2',dimension=gran,
                           value=v,analytical_rows=len(sub),row_share=len(sub)/len(df),represented_sors=int(sub.represented_sors.sum()),
                           sor_share=float(sub.represented_sors.sum()/df.represented_sors.sum())))
        self.csv('u04_composition.csv',composition)
        comp=pd.DataFrame(composition);s=comp[(comp.population_id=='P0')&(comp.dimension=='platform_name')].sort_values('value')
        fig,axs=plt.subplots(1,2,figsize=(11,4.4))
        axs[0].barh(s.value,s.row_share*100);axs[0].set(xlabel='% analytical rows',title='Row representation')
        axs[1].barh(s.value,s.sor_share*100);axs[1].set(xlabel='% represented SoRs',title='Volume representation')
        self.savefig('u04_platform_category_composition','U04','tables/u04_composition.csv','P0/P2',
                     'composition bars','RQ1/RQ2','Platform representation: row count versus represented SoRs',
                     'All rates are shares of different denominators; not platform performance.')
        res=[]
        for pname,df in [('P0',p0),('P1',p1),('P2',p2)]:
            for lvl in [None,'platform_name','category']:
                groups=[('ALL',df)] if lvl is None else list(df.groupby(lvl,observed=True))
                for key,part in groups:
                    for col,numerator in NUMERATORS.items():
                        weighted=float(part[numerator].sum()/part.represented_sors.sum()) if part.represented_sors.sum()>0 else np.nan
                        res.append(dict(population_id=pname,grouping=lvl or 'overall',group=str(key),measure=col,
                                 n_analytical_rows=len(part),represented_sors=int(part.represented_sors.sum()),
                                 record_mean=float(part[col].mean()),record_median=float(part[col].median()),sor_weighted_rate=weighted,
                                 weighting_definition='sum(numerator)/sum(represented_sors)'))
        self.csv('u05_weighted_vs_unweighted.csv',res)
        a=pd.DataFrame(res);a=a[(a.population_id=='P0')&(a.grouping=='platform_name')&(a.measure=='fully_automated_rate')].sort_values('group')
        y=np.arange(len(a));fig,ax=plt.subplots(figsize=(11,4.7))
        ax.barh(y-.18,a.record_mean,height=.35,label='Unweighted weekly-record mean')
        ax.barh(y+.18,a.sor_weighted_rate,height=.35,label='SoR-weighted ratio of sums')
        ax.set(yticks=y,yticklabels=a.group,xlim=(0,1),xlabel='Fully automated rate',title='')
        ax.legend(loc='lower right',fontsize=8)
        self.savefig('u05_weighting_comparison','U05','tables/u05_weighted_vs_unweighted.csv','P0',
                     'paired horizontal bars','All','Typical weekly cell versus aggregate exposure by platform',
                     'Both rates descriptive; weighted proportion does not imply independent SoR rows.',main=True)

    def bivariate_stage(self):
        print('BIVARIATE B01–B05');P=self.data;p0=P['P0'];p1=P['P1'];p2=P['P2'];p4=P['P4'];p6=P['P6']
        rows=[]
        for name,df in [('P0',p0),('P1',p1),('P2',p2)]:
            for label,part in [('ALL',df),*list(df.groupby('platform_name'))]:
                rho,n=spearman_checked(np.log1p(part.represented_sors),part.fully_automated_rate)
                rows.append(dict(population_id=name,platform_name=label,n=n,spearman_rho=rho,
                                 interpretation='Descriptive rank association; no independent-event or causal inference'))
        self.csv('b01_association_summaries.csv',rows)
        fig,axs=plt.subplots(1,2,figsize=(12,4.8))
        for p,sub in p1.groupby('platform_name'):
            axs[0].scatter(np.log1p(sub.represented_sors),sub.fully_automated_rate,s=12,alpha=.45,label=p)
        axs[0].set(xlabel='ln(1+ SoR volume)',ylabel='Fully automated rate [0,1]',ylim=(-.03,1.03),title='P1 complete weeks')
        axs[0].legend(fontsize=7,ncol=2)
        axs[1].scatter(np.log1p(p2.represented_sors),p2.fully_automated_rate,s=32,alpha=.7)
        axs[1].set(xlabel='ln(1+ SoR volume)',ylabel='Fully automated rate [0,1]',ylim=(-.03,1.03),title='P2 full-period segments')
        self.savefig('b01_volume_vs_automation','B01','tables/b01_association_summaries.csv','P1/P2',
                     'stratified scatterplots','RQ1/RQ4','Automation rate versus represented volume',
                     'Rates and absolute exposure are distinct; correlations are descriptive.',main=True)
        fig,axs=plt.subplots(1,2,figsize=(12,4.6))
        for p,sub in p4.groupby('platform_name'):
            axs[0].scatter(sub.fully_automated_rate,sub.content_removed_rate,s=16,alpha=.45,label=p)
            axs[1].scatter(sub.fully_automated_rate,sub.account_or_service_restriction_rate,s=16,alpha=.45,label=p)
        for ax,ylabel in zip(axs,['Content removed rate','Account/service restriction rate']):
            ax.set(xlabel='Fully automated rate',ylabel=ylabel,xlim=(-.03,1.03),ylim=(-.03,1.03))
        axs[1].legend(fontsize=7,ncol=2,loc='upper center')
        self.savefig('b02_automation_enforcement','B02','P4 six-feature inputs','P4',
                     'two joint scatterplots','RQ4','Automation rates and reported enforcement actions',
                     'Reported enforcement flags can overlap; no causal inference.')
        segment=p2.pivot(index='platform_name',columns='category',values='fully_automated_rate')
        support=p2.groupby('category').agg(n_platforms=('platform_name','nunique'),represented_sors=('represented_sors','sum'),
                    min_rate=('fully_automated_rate','min'),max_rate=('fully_automated_rate','max')).reset_index()
        support['rate_range']=support.max_rate-support.min_rate
        self.csv('b03_category_rate_support.csv',support)
        fig,axs=plt.subplots(1,2,figsize=(15,5.6),gridspec_kw={'width_ratios':[.9,2]})
        platforms=sorted(p1.platform_name.unique());axs[0].boxplot([p1.loc[p1.platform_name==p,'fully_automated_rate'] for p in platforms],tick_labels=platforms)
        axs[0].tick_params(axis='x',labelrotation=70);axs[0].set(ylabel='Weekly rate [0,1]',ylim=(-.03,1.03),title='P1: within-platform variation')
        vals=segment.to_numpy(float)
        im=axs[1].imshow(np.ma.masked_invalid(vals),vmin=0,vmax=1,cmap='viridis',aspect='auto')
        axs[1].set_xticks(range(len(segment.columns)),[label_category(x) for x in segment.columns],rotation=90,fontsize=7)
        axs[1].set_yticks(range(len(segment.index)),segment.index);axs[1].set_title('P2: observed full-period categories')
        fig.colorbar(im,ax=axs[1],label='Fully automated rate')
        self.savefig('b03_platform_category_rates','B03','tables/b03_category_rate_support.csv','P1/P2',
                     'boxplot and category heatmap','RQ1/RQ2','Automation-rate variation across platforms and categories',
                     'Blank cells denote no observed category; peer comparisons must remain within category.',main=True)
        summ=p6.groupby('peer_support_count').agg(n_segments=('platform_name','size'),median_abs_divergence=('peer_divergence_abs','median'),
                                      min_abs_divergence=('peer_divergence_abs','min'),max_abs_divergence=('peer_divergence_abs','max'),
                                      total_soRs=('represented_sors','sum')).reset_index()
        self.csv('b04_peer_support_summary.csv',summ)
        fig,axs=plt.subplots(1,2,figsize=(12,4.5))
        axs[0].scatter(p6.peer_support_count,p6.peer_divergence_abs,alpha=.6)
        axs[0].set(xlabel='Other reporting platforms',ylabel='Absolute peer deviation',ylim=(-.03,1.03))
        axs[1].scatter(np.log1p(p6.represented_sors),p6.peer_divergence_abs,alpha=.65,c=p6.peer_support_count,cmap='viridis')
        axs[1].set(xlabel='ln(1+ full-period SoRs)',ylabel='Absolute peer deviation',ylim=(-.03,1.03))
        self.savefig('b04_peer_support_divergence','B04','tables/b04_peer_support_summary.csv','P6',
                     'two support/divergence scatterplots','RQ2','Divergence magnitude versus peer support and segment size',
                     'Peer benchmark is descriptive, not a reference for correct decisions.',main=True)
        mix=p2.copy();den=mix.groupby('platform_name').represented_sors.transform('sum')
        mix['within_platform_share']=mix.represented_sors/den
        self.csv('b05_category_mix.csv',mix[SEG_KEY+['represented_sors','within_platform_share']])
        chart=mix.pivot(index='platform_name',columns='category',values='within_platform_share')
        fig,ax=plt.subplots(figsize=(13,5.4))
        im=ax.imshow(np.ma.masked_invalid(chart.to_numpy(float)),vmin=0,vmax=1,aspect='auto',cmap='viridis')
        ax.set_yticks(range(len(chart.index)),chart.index)
        ax.set_xticks(range(len(chart.columns)),[label_category(x) for x in chart.columns],rotation=90,fontsize=7)
        fig.colorbar(im,ax=ax,label='Share of platform-reported SoRs')
        self.savefig('b05_category_mix','B05','tables/b05_category_mix.csv','P2',
                     'within-platform composition heatmap','RQ1/RQ2','Category mix within each platform',
                     'Blank means unobserved, not a confirmed zero; within-platform observed shares sum to 1.')

    def multivariate_stage(self):
        print('MULTIVARIATE M01–M04');p4=self.data['P4'];data=p4[FEATURES]
        self.csv('m01_rq4_feature_profile.csv',[dict(feature=f,population_id='P4',**describe(data[f])) for f in FEATURES])
        cor=[]
        for f1 in FEATURES:
            for f2 in FEATURES:
                pair=pd.concat([data[f1].rename('first'),data[f2].rename('second')],axis=1).dropna();n=len(pair)
                rho,n2=spearman_checked(data[f1],data[f2]);pr=float(pair['first'].corr(pair['second'])) if pair['first'].nunique()>1 and pair['second'].nunique()>1 else np.nan
                cor.append(dict(feature_x=f1,feature_y=f2,population_id='P4',pair_n=n,spearman=rho,pearson=pr))
        self.csv('m02_feature_correlation.csv',cor)
        matrix=pd.DataFrame(cor).pivot(index='feature_x',columns='feature_y',values='spearman').reindex(index=FEATURES,columns=FEATURES)
        fig,ax=plt.subplots(figsize=(9,7))
        im=ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)),vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_xticks(range(6),[f.replace('_rate','').replace('_',' ') for f in FEATURES],rotation=65,ha='right',fontsize=8)
        ax.set_yticks(range(6),[f.replace('_rate','').replace('_',' ') for f in FEATURES],fontsize=8)
        for i in range(6):
            for j in range(6):
                x=matrix.iloc[i,j]
                if np.isfinite(x):ax.text(j,i,f'{x:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(x)>.6 else 'black')
        fig.colorbar(im,ax=ax,label='Spearman ρ')
        self.savefig('m02_feature_dependence','M02','tables/m02_feature_correlation.csv','P4',
                     'six-feature Spearman heatmap','RQ4','Dependence among the six frozen RQ4 features',
                     'n=1130 eligible rows; correlation is not feature importance or causality.',main=True)
        transformed=RobustScaler().fit_transform(data.to_numpy(float))
        scale=[]
        for i,f in enumerate(FEATURES):
            x=data[f];d=describe(x);sd=describe(transformed[:,i]);
            scale.append(dict(population_id='P4',feature=f,n=len(x),raw_median=d['median'],raw_iqr=d['iqr'],raw_min=d['min'],raw_max=d['max'],
               zero_iqr=bool(d['iqr']==0),near_zero_iqr=bool(abs(d['iqr'])<1e-8),scaled_min=sd['min'],scaled_max=sd['max'],
               scaled_median=sd['median'],max_abs_scaled=float(np.max(np.abs(transformed[:,i]))),
               finite_scaled=bool(np.isfinite(transformed[:,i]).all()),scaler_note='Diagnostic RobustScaler only; anomaly model NOT refitted'))
        self.csv('m03_scaling_diagnostics.csv',scale)
        fig,axs=plt.subplots(1,2,figsize=(12,4.5))
        axs[0].hist(p4.represented_sors,bins=35);axs[0].set(xlabel='Original SoR volume',ylabel='Rows',title='Raw volume')
        axs[0].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        axs[1].hist(p4.log1p_represented_sors,bins=35);axs[1].set(xlabel='ln(1 + SoRs)',ylabel='Rows',title='Frozen volume transformation')
        self.savefig('m03_transformation_checks','M03','tables/m03_scaling_diagnostics.csv','P4',
                     'raw/log1p volume histograms','RQ4','Volume transformation before frozen Isolation Forest',
                     'Scaler fitted only to describe numeric features; no Isolation Forest model fitted.')
        pairs=[('log1p_represented_sors','automation_change_abs'),('fully_automated_rate','content_removed_rate'),
               ('fully_automated_rate','weekly_peer_divergence_abs')]
        result=[]
        for varx,vary in pairs:
            for dim in ('platform_name','category'):
                for key,sub in p4.groupby(dim):
                    rho,n=spearman_checked(sub[varx],sub[vary])
                    result.append(dict(grouping=dim,group=str(key),feature_x=varx,feature_y=vary,population_id='P4',
                                       n=n,spearman=rho,available_for_interpretation=n>=5 and np.isfinite(rho)))
        self.csv('m04_stratified_dependence.csv',result)
        fig,axs=plt.subplots(1,3,figsize=(14,4.4))
        for j,(a,b) in enumerate(pairs):
            for p,sub in p4.groupby('platform_name'):axs[j].scatter(sub[a],sub[b],alpha=.4,s=10,label=p)
            axs[j].set(xlabel=a.replace('_',' '),ylabel=b.replace('_',' '))
            axs[j].tick_params(axis='x',labelsize=7,rotation=15)
        axs[2].legend(fontsize=7,ncol=2)
        self.savefig('m04_joint_feature_profiles','M04','tables/m04_stratified_dependence.csv','P4',
                     'stratified joint-feature panels','RQ4','Selected joint relationships in RQ4 feature space',
                     'Pairwise plots are descriptive; no additional model features are introduced.')

    def temporal_stage(self):
        print('TEMPORAL T01–T04');P=self.data;p1=P['P1'];p3=P['P3'];alerts=pd.read_csv(self.root/'outputs/synopsis/rq3_temporal_alerts.csv')
        alerts['week_start']=pd.to_datetime(alerts.week_start)
        parts=[]
        for group,df in [('OVERALL',p1),*list(p1.groupby('platform_name'))]:
            z=df.groupby('week_start')[['represented_sors','fully_auto_sors']].sum().reset_index()
            z['weighted_automation_rate']=z.fully_auto_sors/z.represented_sors.replace(0,np.nan)
            z['platform_name']=group;z['population_id']='P1';parts.append(z)
        trends=pd.concat(parts,ignore_index=True).sort_values(['platform_name','week_start'])
        self.csv('t01_weekly_rate_volume.csv',trends)
        fig,axs=plt.subplots(2,1,figsize=(13,8),sharex=True)
        for p,sub in trends.groupby('platform_name'):
            if p=='OVERALL':axs[0].plot(sub.week_start,sub.represented_sors,label=p,linewidth=2.5,color='black')
            else:axs[0].plot(sub.week_start,sub.represented_sors,label=p,alpha=.7)
            axs[1].plot(sub.week_start,sub.weighted_automation_rate,label=p,linewidth=2.5 if p=='OVERALL' else 1.2,
                        color='black' if p=='OVERALL' else None,alpha=1 if p=='OVERALL' else .7)
        axs[0].set(ylabel='Represented SoRs per complete week',title='Weekly volume')
        axs[1].set(ylabel='Fully automated rate [0,1]',ylim=(-.03,1.03),xlabel='Complete week beginning (Monday)',title='Weighted weekly automation rate')
        axs[1].legend(fontsize=7,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.5))
        self.savefig('t01_weekly_trends','T01','tables/t01_weekly_rate_volume.csv','P1',
                     'weekly time-series panels','RQ1/RQ3','Reported moderation volume and weighted automation over 26 complete weeks',
                     'Only full weeks are shown; rates are ratio-of-sums, not averages of cell rates.',main=True)
        vol=[]
        for (p,c),g in p3.groupby(SEG_KEY):
            ch=g.automation_change;med=ch.median();mad=(ch-med).abs().median()
            vol.append(dict(platform_name=p,category=c,population_id='P3',n_reliable_comparisons=len(g),
                            signed_delta_median=float(med),delta_mad=float(mad),median_abs_change=float(ch.abs().median()),
                            undefined_robust_score_count=int(g.robust_change_score.isna().sum()),
                            temporal_alert_count=int(g.temporal_alert.astype(str).str.lower().eq('true').sum())))
        self.csv('t02_series_volatility.csv',vol)
        fig,axs=plt.subplots(1,2,figsize=(12,4.6))
        axs[0].hist(p3.automation_change,bins=np.linspace(-1,1,51))
        axs[0].axvline(-.1,color='gray',ls='--');axs[0].axvline(.1,color='gray',ls='--')
        axs[0].set(xlabel='Current minus preceding automated rate (proportion points)',ylabel='Reliable changes')
        axs[1].hist([x['delta_mad'] for x in vol],bins=25)
        axs[1].set(xlabel='Within-series MAD of signed changes',ylabel='Eligible series')
        self.savefig('t02_change_distribution','T02','tables/t02_series_volatility.csv','P3',
                     'signed-change and MAD histograms','RQ3','Distribution of reliable weekly changes and within-series MAD',
                     'MAD=0 produces undefined robust z, not z=0.')
        eligibility=self.valid_pairs()
        self.csv('t03_alert_eligibility.csv',[dict(reason=s,observations=int(n),population_id='P1',selection_stage='P1 to P3')
                         for s,n in eligibility.status.value_counts().items()])
        z=p3.robust_change_score.abs().ge(3.5).fillna(False);m=p3.automation_change_abs.ge(.1).fillna(False)
        flag=p3.temporal_alert.astype(str).str.lower().eq('true')
        tab=[]
        for i,label in enumerate(['neither threshold','material change only','robust z only','both thresholds']):
            zz=bool(i&2);mm=bool(i&1)
            hit=(z==zz)&(m==mm)&p3.robust_change_score.notna()
            tab.append(dict(component=label,n_rows=int(hit.sum()),n_alerts=int((hit&flag).sum()),robust_criterion=zz,material_criterion=mm))
        tab.append(dict(component='undefined robust z (MAD=0)',n_rows=int(p3.robust_change_score.isna().sum()),
                        n_alerts=int((p3.robust_change_score.isna()&flag).sum()),robust_criterion=False,material_criterion=False))
        self.csv('t03_alert_component_matrix.csv',tab)
        if flag.sum()!=30 or (flag!=(z&m)).any():raise ValueError('Frozen temporal flags do not match original criteria')
        # representative cases: largest |delta|, distinct series, and 3-week context when observed
        examples=alerts.assign(abs_change=alerts.automation_change.abs()).sort_values(['abs_change','platform_name','category','week_start'],ascending=[False,True,True,True]).drop_duplicates(SEG_KEY).head(4)
        context=[]
        fig,axs=plt.subplots(2,2,figsize=(13,8),sharey=True)
        for ax,(_,example) in zip(axs.ravel(),examples.iterrows()):
            platform=example.platform_name; category=example.category;t=example.week_start
            seg=p1[(p1.platform_name==platform)&(p1.category==category)]
            local=seg[seg.week_start.between(t-pd.Timedelta(days=14),t+pd.Timedelta(days=14))].sort_values('week_start')
            ax.plot(local.week_start,local.fully_automated_rate,marker='o')
            ax.axvline(t,color='black',ls='--',alpha=.7)
            ax.set(title=f'{platform} • {label_category(category)[:37]}',ylabel='Rate [0,1]',ylim=(-.03,1.03))
            ax.tick_params(axis='x',labelrotation=40,labelsize=7)
            for r in local.itertuples():context.append(dict(selected_alert_week=t.date(),week_start=r.week_start.date(),
                platform_name=platform,category=category,represented_sors=int(r.represented_sors),
                fully_automated_rate=float(r.fully_automated_rate),is_alert_week=bool(r.week_start==t)))
        self.csv('t04_alert_context.csv',context)
        self.savefig('t04_alert_context','T04','tables/t04_alert_context.csv','P1/P3',
                     'four annotated local weekly series','RQ3','Context around selected large-magnitude reliable RQ3 alerts',
                     'Examples selected by highest absolute change among distinct series; context is descriptive.')

    def rq_stage(self):
        print('RQ CONTEXT R01–R04');P=self.data;p2=P['P2'];p3=P['P3'];p4=P['P4'];p5=P['P5'];p6=P['P6'];p7=P['P7']
        context=[]
        for s,df in [('ALL',p2),('EXCLUDING_OTHER',p2[p2.category!=OTHER]),('OTHER_ONLY',p2[p2.category==OTHER])]:
            num=df.fully_auto_sors.sum();den=df.represented_sors.sum()
            context.append(dict(category_scope=s,n_segments=len(df),represented_sors=int(den),fully_auto_sors=int(num),
                          weighted_automation_rate=float(num/den) if den else np.nan,
                          exposure_hhi=float(((df.fully_auto_sors/num)**2).sum()) if num else np.nan))
        self.csv('r01_other_context.csv',context)
        exposure=p7.sort_values('automated_exposure_share',ascending=False).head(20).copy()
        exposure['segment']=exposure.platform_name+' / '+exposure.category.map(label_category)
        fig,axs=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1.5,1]})
        bar=exposure.iloc[::-1]
        axs[0].barh(bar.segment,bar.automated_exposure_share*100)
        axs[0].set(xlabel='% of all fully automated SoRs',title='Highest segment shares')
        vals=p7.sort_values('automated_exposure_share',ascending=False).automated_exposure_share.cumsum().to_numpy()
        axs[1].plot(range(1,len(vals)+1),vals,linewidth=2);axs[1].set(xlabel='Segments (descending F)',ylabel='Cumulative automated exposure',ylim=(0,1.03),title='Concentration curve')
        self.savefig('r01_exposure_concentration','R01','tables/r01_other_context.csv;outputs/synopsis/rq1_segment_exposure.csv','P2/P7',
                     'exposure bars and cumulative curve','RQ1','Distribution of fully automated moderation exposure',
                     'Descriptive concentration, not a platform quality score; include OTHER sensitivity.')
        dev=p6.sort_values('peer_divergence_abs',ascending=False).head(18).copy()
        self.csv('r02_divergence_context.csv',p6.assign(large_flag=p6.peer_divergence_abs.gt(.25))[SEG_KEY+[
                          'represented_sors','peer_median_automation_rate','peer_support_count','peer_divergence_signed','peer_divergence_abs','large_flag']])
        dev['segment']=dev.platform_name+' / '+dev.category.map(label_category)
        fig,ax=plt.subplots(figsize=(12,7))
        y=np.arange(len(dev));ax.barh(y,dev.peer_divergence_signed*100)
        ax.axvline(25,color='gray',ls='--',alpha=.7);ax.axvline(-25,color='gray',ls='--',alpha=.7)
        ax.set(yticks=y,yticklabels=dev.segment,xlabel='Signed difference in fully automated rate (percentage points)')
        ax.invert_yaxis()
        self.savefig('r02_peer_divergence','R02','tables/r02_divergence_context.csv','P6',
                     'signed deviation bars','RQ2','Largest descriptive same-category leave-one-out peer deviations',
                     'Peer medians are non-normative; ±25pp denotes an operational threshold.')
        alerts=pd.read_csv(self.root/'outputs/synopsis/rq3_temporal_alerts.csv')
        alerts['previous_fully_automated_rate']=alerts.fully_automated_rate-alerts.automation_change
        alerts['previous_week_volume']=alerts.previous_week_volume.astype(float)
        self.csv('r03_alert_explanations.csv',alerts[['week_start','platform_name','category','previous_week_start','represented_sors',
            'previous_week_volume','previous_fully_automated_rate','fully_automated_rate','automation_change','automation_change_abs',
            'robust_change_score','reliable_comparisons','temporal_alert']].sort_values(['week_start','platform_name','category']))
        r04=[]
        for f in FEATURES:
            base=describe(p4[f]);chosen=describe(p5[f])
            r04.append(dict(feature=f,population_id='P4 vs P5',eligible_n=len(p4),stable_n=len(p5),
                            eligible_median=base['median'],eligible_q25=base['q25'],eligible_q75=base['q75'],
                            stable_median=chosen['median'],stable_q25=chosen['q25'],stable_q75=chosen['q75']))
        self.csv('r04_shortlist_context.csv',r04)
        sensitivity=[]
        mapping=[('baseline','rq4_baseline_comparison.csv'),('feature_ablation','rq4_feature_set_sensitivity.csv'),
                 ('top_k','rq4_topk_sensitivity.csv'),('stability_threshold','rq4_stability_threshold_sensitivity.csv'),
                 ('candidate_region','rq4_candidate_threshold_sensitivity.csv')]
        for typ,p in mapping:
            for r in pd.read_csv(self.root/'outputs/synopsis'/p).to_dict('records'):
                sensitivity.append(dict(sensitivity_type=typ,source_path='outputs/synopsis/'+p,settings=json.dumps(r,sort_keys=True)))
        self.csv('r04_frozen_sensitivity.csv',sensitivity)
        from collections import Counter
        topfeature=Counter(p5.largest_deviation_feature)
        baseline=pd.read_csv(self.root/'outputs/synopsis/rq4_baseline_comparison.csv')
        fig,axs=plt.subplots(1,2,figsize=(12,4.8))
        axs[0].bar(range(len(topfeature)),list(topfeature.values()))
        axs[0].set(xticks=range(len(topfeature)),xticklabels=[f.replace('_',' ') for f in topfeature],ylabel='Stable shortlist rows',title='Largest scaled-deviation feature')
        axs[0].tick_params(axis='x',rotation=15,labelsize=7)
        axs[1].barh(baseline.baseline,baseline.overlap_count)
        axs[1].set(xlabel='Overlap with 18-item stable shortlist',xlim=(0,20),title='Published top-20 baseline overlap')
        self.savefig('r04_shortlist_feature_context','R04','tables/r04_shortlist_context.csv;tables/r04_frozen_sensitivity.csv','P4/P5',
                     'feature-context and baseline bars','RQ4','Descriptive context of 18 stable audit-priority observations',
                     'Largest scaled deviation is not causal attribution; baselines are top-20 each.',main=True)

    def finalize(self,completed):
        path=self.out/'metadata/figure_catalog.csv'
        if path.exists():
            prior=pd.read_csv(path).fillna('').to_dict('records');by={x['figure_id']:x for x in prior}
        else:by={}
        for f in self.figures:by[f['figure_id']]=f
        fields=['figure_id','analysis_ids','source_table','platform_category_week_scope','chart_type','main_or_appendix','rq_link','title','caption_note','output_png','output_pdf','status']
        pd.DataFrame(list(by.values()),columns=fields).sort_values('figure_id').to_csv(path,index=False)
        regpath=self.out/'metadata/evidence_register.csv';reg=pd.read_csv(regpath).fillna('')
        for id in completed:
            row=reg.index[reg.analysis_id==id]
            if len(row):reg.loc[row,'status']='GENERATED'
        reg.to_csv(regpath,index=False)
        note=self.out/'metadata/interpretation_register.csv'
        if not note.exists():
            cols=['insight_id','analysis_ids','figure_or_table','evidence_metric','observation','what_it_means',
                  'what_it_does_not_mean','rq_link','limitations','report_location','review_status']
            pd.DataFrame(columns=cols).to_csv(note,index=False)
        exc=self.out/'metadata/data_exceptions.csv'
        if not exc.exists():pd.DataFrame(columns=['issue_id','analysis_id','input_population','observation_key','field',
                                                   'exception_type','observed_value','action_taken','rationale']).to_csv(exc,index=False)
        manifest_path=self.out/'metadata/eda_run_manifest.json'
        sha={}
        for p in sorted(set(INPUTS.values())|set(EXTRA_INPUTS)):
            fp=self.root/p
            if fp.exists():sha[p]=filehash(fp)
        generated={p.relative_to(self.out).as_posix():filehash(p) for p in self.out.rglob('*')
                   if p.is_file() and p!=manifest_path}
        try:commit=subprocess.run(['git','rev-parse','HEAD'],cwd=self.root,text=True,capture_output=True,check=True).stdout.strip()
        except Exception:commit='NOT_A_GIT_CHECKOUT'
        old=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        body=dict(study_version='QM640_Phase2_EDA_v1',repository_commit=commit,run_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                 python_version=platform.python_version(),package_versions={"pandas":pd.__version__,"numpy":np.__version__,
                 "scipy":__import__('scipy').__version__,"matplotlib":matplotlib.__version__,"scikit_learn":__import__('sklearn').__version__},
                 input_files_sha256=sha,input_row_counts={k:len(v) for k,v in self.data.items()},
                 source_provenance_reference='metadata/source_manifest.csv; Phase 1 independent six-archive SHA-1 audit',
                 analysis_ids=sorted(set(old.get('analysis_ids',[]))|set(completed)),generated_outputs_sha256=generated,
                 validation_summary=old.get('validation_summary',{'status':'NOT_YET_EXECUTED'}),mode='EDA_ONLY')
        manifest_path.write_text(json.dumps(body,indent=2,allow_nan=False)+'\n')
        print('Completed IDs:',', '.join(completed));print('Output root:',self.out)

    def run(self,stages):
        for s in stages:getattr(self,{'data':'data_stage','univariate':'univariate_stage','bivariate':'bivariate_stage',
                'multivariate':'multivariate_stage','temporal':'temporal_stage','rq':'rq_stage'}[s])()
        self.finalize([v for s in stages for v in GROUPS[s]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--stage',choices=STAGES+['all'],default='all')
    a=parser.parse_args();out=a.out or (a.root/'outputs/interim/eda')
    EDA(a.root,out).run(STAGES if a.stage=='all' else [a.stage])

if __name__=='__main__':main()
