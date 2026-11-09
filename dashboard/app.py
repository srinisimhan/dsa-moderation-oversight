"""DSA Oversight Radar — retrospective research prototype; no model fitting."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / 'outputs/synopsis'
B = ROOT / 'outputs/interim/model_benchmarks'
V = ROOT / 'outputs/final/research_validation'

FILES = {
 'segments':(S/'rq1_segment_exposure.csv',['platform_name','category','represented_sors','fully_auto_sors','automated_exposure_share']),
 'category':(S/'rq1_category_variation.csv',['category','rate_range']),
 'peers':(S/'rq2_peer_divergence.csv',['platform_name','category','peer_divergence_signed','peer_divergence_abs','peer_support_count']),
 'weekly':(S/'rq3_reliable_weekly.csv',['week_start','platform_name','category','fully_automated_rate','automation_change','robust_change_score','temporal_alert']),
 'alerts':(S/'rq3_temporal_alerts.csv',['week_start','platform_name','category','temporal_alert']),
 'scored':(S/'rq4_scored_observations.csv',['week_start','platform_name','category','anomaly_score','anomaly_rank','seed_inclusion_count','stable_anomaly','largest_deviation_feature']),
 'shortlist':(S/'rq4_stable_shortlist.csv',['week_start','platform_name','category','anomaly_score','seed_inclusion_count']),
 'challengers':(B/'rq4_challenger_model_comparison.csv',['model','eligible_rows','overlap_with_if_top_k']),
 'holdout':(V/'temporal_holdout_scored.csv',['fold','week_start','platform_name','category','if_priority_score','if_rank','if_seed_top20_count','if_stable_8of10','largest_train_scaled_deviation_feature']),
 'holdout_compare':(V/'temporal_holdout_model_comparison.csv',['first_test_week','model','test_rows','overlap_with_if_top20']),
 'sensitivity':(V/'temporal_feature_sensitivity.csv',['first_test_week','variant','overlap_with_primary_top20']),
}

def load_data():
    result = {}
    for key,(path,columns) in FILES.items():
        if not path.is_file(): raise FileNotFoundError(f'Required frozen output missing: {path}')
        frame = pd.read_csv(path)
        missing = set(columns)-set(frame.columns)
        if missing: raise ValueError(f'{path}: missing required columns {sorted(missing)}')
        result[key] = frame
    for key,path in [('metrics',S/'summary_metrics.json'),('validation',V/'final_validation_manifest.json'),('manifest',S/'analysis_run_manifest.json')]:
        if not path.is_file(): raise FileNotFoundError(path)
        result[key]=json.loads(path.read_text(encoding='utf-8'))
    return result

DATA = load_data()
PLATFORMS=sorted(DATA['segments'].platform_name.unique().tolist())
CATEGORIES=sorted(DATA['segments'].category.unique().tolist())
ALL='All'


def filtered(frame,platform,category,start=None,end=None,search=''):
    d=frame.copy()
    if platform!=ALL and 'platform_name' in d: d=d[d.platform_name==platform]
    if category!=ALL and 'category' in d: d=d[d.category==category]
    if start and 'week_start' in d: d=d[d.week_start.astype(str)>=start]
    if end and 'week_start' in d: d=d[d.week_start.astype(str)<=end]
    if search:
        term=str(search).casefold()
        d=d[d.astype(str).apply(lambda c:c.str.casefold().str.contains(term,regex=False)).any(axis=1)]
    return d.reset_index(drop=True)


def chart(frame,x,y,color=None,title='',kind='bar'):
    if frame.empty: return px.scatter(title='No rows match the current filters')
    if kind=='line': fig=px.line(frame,x=x,y=y,color=color,title=title,markers=True)
    elif kind=='scatter': fig=px.scatter(frame,x=x,y=y,color=color,title=title,hover_data=[c for c in ['category','platform_name','week_start'] if c in frame.columns and c not in [x,y,color]])
    else: fig=px.bar(frame,x=x,y=y,color=color,title=title)
    fig.update_layout(margin=dict(l=20,r=20,t=55,b=35),legend_title_text='',height=390)
    return fig


def export_frame(d,prefix):
    # Streamlit's download_button streams DataFrames directly. No temp files.
    return f'{prefix}.csv'


def selected_detail(d,evt):
    try:
        index=evt.index[0] if isinstance(evt.index,(list,tuple)) else evt.index
        if not isinstance(index,int) or index<0 or index>=len(d): return 'Select a data row to inspect its original values.'
        record=d.iloc[index].where(pd.notna(d.iloc[index]),None).to_dict()
        return '```json\n'+json.dumps(record,indent=2,default=str)+'\n```'
    except (IndexError,KeyError,TypeError): return 'Select a data row to inspect its original values.'


def overview(p,c):
    d=filtered(DATA['segments'],p,c)
    total=int(d.represented_sors.sum()); count=len(d)
    g=d.groupby('platform_name',as_index=False)[['represented_sors','fully_auto_sors']].sum()
    bycat=d.groupby('category',as_index=False)['represented_sors'].sum().sort_values('represented_sors',ascending=False)
    metrics=DATA['metrics']['data_validation']
    summary=(f'### Selected full-period scope\n**{total:,}** represented SoRs · **{count}** segments · '
             f'**{d.platform_name.nunique()}** platforms · **{d.category.nunique()}** categories.\n\n'
             f'Full unfiltered census: **{metrics["weekly_rows"]:,}** weekly observations, **{metrics["weekly_represented_sors"]:,}** SoRs, **{metrics["platforms"]}** platforms, **{metrics["categories"]}** categories, **{metrics["summary_rows"]}** segments (1 Mar–31 Aug 2026). '
             'Selected totals are summed from the frozen segment table, not an independent sample.')
    return summary,chart(g,'platform_name','represented_sors',title='Represented SoRs by platform'),chart(bycat,'category','represented_sors',title='Represented SoRs by category'),d,export_frame(d,'overview_segments')


def rq12(p,c,search):
    a=filtered(DATA['segments'],p,c,search=search)
    b=filtered(DATA['peers'],p,c,search=search)
    hhi=DATA['metrics']['rq1']['hhi']; hhi_no=DATA['metrics']['rq1']['hhi_excluding_other']
    summary=(f'### Frozen research metrics (unfiltered reference)\nRQ1 HHI **{hhi:.6f}** (excluding OTHER: **{hhi_no:.6f}**); '
             f'RQ2 median absolute peer divergence **{DATA["metrics"]["rq2"]["median_abs_divergence"]:.6f}**; '
             f'**{DATA["metrics"]["rq2"]["large_divergence_segments"]}** large-divergence segments out of **{DATA["metrics"]["rq2"]["comparable_segments"]}**. '
             'HHI is the frozen study-wide statistic, not recomputed for filtered subsets. '
             f'Visible: {len(a)} exposure rows and {len(b)} peer rows.')
    return summary,chart(a.sort_values('automated_exposure_share',ascending=False).head(25),'platform_name','automated_exposure_share',color='category',title='Top 25 segment exposure shares'),chart(b,'fully_automated_rate','peer_divergence_signed',color='platform_name',title='Same-category signed leave-one-out divergence',kind='scatter'),a,b,export_frame(a,'rq1_exposure'),export_frame(b,'rq2_peers')


def rq3(p,c,start,end,only_alerts,search):
    d=filtered(DATA['weekly'],p,c,start,end,search)
    if only_alerts: d=d[d.temporal_alert.astype(str).str.lower().isin(['true','1'])].reset_index(drop=True)
    plot=filtered(DATA['weekly'],p,c,start,end)
    plot=plot.sort_values('week_start')
    plot['series']=plot['platform_name'].astype(str)+' | '+plot['category'].astype(str)
    rq3metrics=DATA['metrics']['rq3']
    summary=(f'### Frozen RQ3: **{rq3metrics["reliable_observations"]:,}** eligible comparisons · **{rq3metrics["eligible_series"]}** reliable series · **{rq3metrics["alerts"]}** alerts across **{rq3metrics["affected_series"]}** series.\n'
             f'Visible records: **{len(d)}**; visible alerts: **{int(d.temporal_alert.sum())}**. '
             'Alert status and robust scores are loaded from the original research output. '
             'The line chart shows weekly automation rates for the selected series; choose a platform/category for a clear drill-down.')
    return summary,chart(plot,'week_start','fully_automated_rate',color='series',title='Eligible weekly automation rates by platform-category series',kind='line'),chart(d,'week_start','robust_change_score',color='platform_name',title='Robust change scores',kind='scatter'),d,export_frame(d,'rq3_visible')


def rq4(mode,p,c,start,end,stable,search):
    held=mode!='Original full-period (in-sample)'
    if held:
        fold=1 if mode=='Holdout: 6 July 2026 (primary)' else 2
        d=DATA['holdout'][DATA['holdout'].fold==fold].copy()
        score='if_priority_score';rank='if_rank';flag='if_stable_8of10';seed='if_seed_top20_count';feature='largest_train_scaled_deviation_feature'
        bounds='6 Jul–24 Aug 2026' if fold==1 else '20 Jul–24 Aug 2026'
        schedule=DATA['validation']['training_schedule'][fold-1]
        expected=schedule['test_model_rows']
        compare=DATA['holdout_compare'][DATA['holdout_compare'].first_test_week==('2026-07-06' if fold==1 else '2026-07-20')]
        sensitivity=DATA['sensitivity'][DATA['sensitivity'].first_test_week==('2026-07-06' if fold==1 else '2026-07-20')]
        info=(f'### Retrospective chronological holdout — {bounds}\n'
              f'**{expected}** test observations; **{schedule["train_model_rows"]}** earlier training observations; '
              f'**{schedule["test_if_stable_8of10_count"]}** stable test cases (≥8/10 seeds). '
              'Earlier weeks were used for fitting in the approved research; the dashboard only reads precomputed later-week scores. '
              'The two holdout windows overlap and are **not independent test samples**.')
    else:
        d=DATA['scored'].copy();score='anomaly_score';rank='anomaly_rank';flag='stable_anomaly';seed='seed_inclusion_count';feature='largest_deviation_feature'
        compare=DATA['challengers'];sensitivity=DATA['sensitivity'].iloc[0:0]
        info=('### Original full-period, retrospective **in-sample** Isolation Forest ranking\n'
              f'**{DATA["metrics"]["rq4"]["eligible_observations"]:,}** eligible records; **{DATA["metrics"]["rq4"]["stable_anomalies"]}** stable cases ({DATA["metrics"]["rq4"]["selected_10_of_10"]} selected in 10/10 seeds, {DATA["metrics"]["rq4"]["selected_9_of_10"]} in 9/10). '
              'Frozen 300-tree, six-feature, 10-seed model. This is an exploratory priority queue, not out-of-sample validation.')
    d=filtered(d,p,c,start,end,search)
    if stable: d=d[d[flag].astype(str).str.lower().isin(['true','1'])]
    d=d.sort_values([rank,'week_start'],na_position='last').reset_index(drop=True)
    viewcols=['week_start','platform_name','category','represented_sors',score,rank,seed,flag,feature]
    for col in ['largest_abs_scaled_deviation','largest_train_scaled_deviation_size','fully_automated_rate','weekly_peer_divergence_abs','automation_change_abs']:
        if col in d.columns: viewcols.append(col)
    view=d[viewcols]
    info+=f'\n\n**Visible cases: {len(d)}.** Scores and ranks are original, not renumbered after filtering. Largest robust-scaled deviation is diagnostic context, **not** SHAP, causal attribution, or proof of wrongdoing.'
    plot=d.sort_values(rank).head(30).copy()
    plot['case']=plot['platform_name']+' | '+plot['category'].str.replace('STATEMENT_CATEGORY_','',regex=False)+' | '+plot['week_start']
    return info,chart(plot,'case',score,title='Highest-priority visible records (frozen scores)'),view,compare,sensitivity,export_frame(view,'rq4_visible')


def validate():
    m=DATA['metrics']; v=DATA['validation']['training_schedule']
    assert len(DATA['segments'])==m['rq1']['segment_count']==86
    assert int(DATA['segments'].represented_sors.sum())==m['data_validation']['weekly_represented_sors']==430511443
    assert len(DATA['peers'])==m['rq2']['comparable_segments']==86
    assert len(DATA['weekly'])==m['rq3']['reliable_observations']==1203
    assert len(DATA['alerts'])==m['rq3']['alerts']==30
    assert int(DATA['weekly'].temporal_alert.sum())==30
    assert len(DATA['scored'])==m['rq4']['eligible_observations']==1130
    assert len(DATA['shortlist'])==m['rq4']['stable_anomalies']==18
    assert int(DATA['scored'].stable_anomaly.sum())==18
    assert int((DATA['shortlist'].seed_inclusion_count==10).sum())==17
    assert int((DATA['shortlist'].seed_inclusion_count==9).sum())==1
    keys=['week_start','platform_name','category']
    assert set(map(tuple,DATA['shortlist'][keys].to_numpy()))==set(map(tuple,DATA['scored'].loc[DATA['scored'].stable_anomaly,keys].to_numpy()))
    for fold,record in enumerate(v,start=1):
        subset=DATA['holdout'][DATA['holdout'].fold==fold]
        assert len(subset)==record['test_model_rows']
        assert int(subset.if_stable_8of10.sum())==record['test_if_stable_8of10_count']
        assert str(subset.week_start.min())==record['min_test_date']
        assert str(subset.week_start.max())==record['max_test_date']
    return True

validate()



st.set_page_config(page_title="DSA Oversight Radar",layout="wide")
st.title("DSA Oversight Radar")
st.caption("EU DSA Transparency Database | 1 March–31 August 2026 | Retrospective research prototype")
st.warning("Historical aggregate-level audit prioritisation only. Alerts are not evidence of wrongdoing, and model agreement is not predictive accuracy.")

def filters(key, dates=False, search=False):
    a,b=st.columns(2)
    p=a.selectbox('Platform',[ALL]+PLATFORMS,key=key+'_p')
    c=b.selectbox('Category',[ALL]+CATEGORIES,key=key+'_c')
    start=end=None
    if dates:
        x,y=st.columns(2)
        start=x.date_input('From',value=__import__('datetime').date(2026,3,1),key=key+'_start').isoformat()
        end=y.date_input('Through',value=__import__('datetime').date(2026,8,31),key=key+'_end').isoformat()
    q=st.text_input('Search rows',key=key+'_q') if search else ''
    return p,c,start,end,q

def table(d,key):
    st.dataframe(d,use_container_width=True,hide_index=True,on_select='ignore')
    st.download_button('Download filtered CSV',d.to_csv(index=False).encode('utf-8'),file_name=key+'.csv',mime='text/csv',key=key+'_download')
    if len(d):
        choices=[f"{i}: {row.get('platform_name','')} | {row.get('category','')} | {row.get('week_start','')}" for i,row in d.reset_index(drop=True).iterrows()]
        pick=st.selectbox('Inspect original row',range(len(choices)),format_func=lambda i:choices[i],key=key+'_detail')
        with st.expander('Case details',expanded=False):st.json(d.iloc[pick].where(pd.notna(d.iloc[pick]),None).to_dict())

one,two,three,four=st.tabs(['Overview','RQ1–RQ2 Exposure & Peers','RQ3 Temporal Instability','RQ4 Audit Triage'])
with one:
    p,c,_,_,_=filters('overview')
    summary,f,g,d,_=overview(p,c)
    st.markdown(summary)
    x,y=st.columns(2);x.plotly_chart(f,use_container_width=True);y.plotly_chart(g,use_container_width=True)
    table(d,'overview_segments')
    st.caption('Source: outputs/synopsis/rq1_segment_exposure.csv and summary_metrics.json. Filtered totals are not the full census.')
with two:
    p,c,_,_,q=filters('rq12',search=True)
    summary,f,g,a,b,_,_=rq12(p,c,q)
    st.markdown(summary)
    x,y=st.columns(2);x.plotly_chart(f,use_container_width=True);y.plotly_chart(g,use_container_width=True)
    st.subheader('RQ1: Exposure segments');table(a,'rq1_exposure')
    st.subheader('RQ2: Leave-one-platform-out peer divergence');table(b,'rq2_peers')
    st.caption('Sources: frozen rq1_segment_exposure.csv, rq2_peer_divergence.csv, summary_metrics.json. HHI is the original unfiltered study metric.')
with three:
    p,c,start,end,q=filters('rq3',dates=True,search=True)
    alerts_only=st.checkbox('Show alerts only',value=False)
    summary,f,g,d,_=rq3(p,c,start,end,alerts_only,q)
    st.markdown(summary)
    x,y=st.columns(2);x.plotly_chart(f,use_container_width=True);y.plotly_chart(g,use_container_width=True)
    table(d,'rq3_visible')
    st.caption('Source: frozen rq3_reliable_weekly.csv and rq3_temporal_alerts.csv. Alerts are precomputed, not recalculated.')
with four:
    mode=st.radio('Research population',['Original full-period (in-sample)','Holdout: 6 July 2026 (primary)','Holdout: 20 July 2026 (overlapping sensitivity)'],horizontal=False)
    p,c,_,_,q=filters('rq4',search=True)
    a,b=st.columns(2)
    start=a.text_input('From date (optional, YYYY-MM-DD)',key='rq4_start')
    end=b.text_input('Through date (optional, YYYY-MM-DD)',key='rq4_end')
    stable=st.checkbox('Stable cases only (≥8/10 seeds)',value=True)
    summary,f,d,compare,sensitivity,_=rq4(mode,p,c,start,end,stable,q)
    st.markdown(summary);st.plotly_chart(f,use_container_width=True)
    table(d,'rq4_visible')
    with st.expander('Precomputed model comparison (same population only)'):
        st.dataframe(compare,use_container_width=True,hide_index=True)
        st.download_button('Download comparison CSV',compare.to_csv(index=False).encode(),file_name='model_comparison.csv')
    if not sensitivity.empty:
        with st.expander('Chronological holdout feature sensitivity'):
            st.dataframe(sensitivity,use_container_width=True,hide_index=True)
            st.download_button('Download sensitivity CSV',sensitivity.to_csv(index=False).encode(),file_name='feature_sensitivity.csv')
    st.caption('Original and held-out rankings are separate historical populations. Largest robust-scaled deviation is diagnostic context, not SHAP or causal attribution.')
st.divider()
st.subheader('Provenance and limitations')
st.markdown('Frozen outputs from the approved QM640 research handoff; see `docs/APPROVED_RESEARCH_DECISIONS.md`, `outputs/synopsis/analysis_run_manifest.json`, `outputs/final/research_validation/final_validation_manifest.json` and `SHA256_MANIFEST.json`. The original source is the European Commission DSA Transparency Database (CC BY 4.0). The package excludes raw monthly Parquet archives. No model fitting or new experiments occur in this application. The observational census is bounded and not an independent random sample. Category peers may not be exchangeable. Same-week peer features assume availability before batch scoring. Holdout windows overlap and are not independent. There are no verified moderation-error or audit-yield labels. This is not a live regulatory decision system.')
