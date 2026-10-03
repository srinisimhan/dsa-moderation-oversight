#!/usr/bin/env python3
from __future__ import annotations
import gc, hashlib, shutil, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests

STUDY_START = pd.Timestamp("2026-03-01")
STUDY_END = pd.Timestamp("2026-08-31")
MONTHS = ["2026-03-01","2026-04-01","2026-05-01","2026-06-01","2026-07-01","2026-08-01"]
SELECTED_PLATFORMS = ["Instagram","TikTok","YouTube","X","Snapchat","LinkedIn","Roblox"]
DSA_BASE_URL = "https://d3vax7phxnku8l.cloudfront.net/agg/pqt/data/tdb_data/global___full/aggregations"
NEEDED_COLUMNS = [
    "platform_name","category","created_at","automated_decision",
    "decision_account","decision_provision",
    "DECISION_VISIBILITY_CONTENT_REMOVED","count"
]

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT/"data"
META_DIR = ROOT/"metadata"
CACHE_DIR = ROOT/".cache_dsa"

COUNT_COLUMNS = [
    "represented_sors","fully_auto_sors","partially_auto_sors","not_auto_sors",
    "content_removed_sors","account_terminated_sors","account_suspended_sors",
    "service_terminated_sors","service_suspended_sors",
    "account_or_service_restriction_sors"
]
RATE_NUMERATORS = {
    "fully_automated_rate":"fully_auto_sors",
    "partially_automated_rate":"partially_auto_sors",
    "not_automated_rate":"not_auto_sors",
    "content_removed_rate":"content_removed_sors",
    "account_terminated_rate":"account_terminated_sors",
    "account_suspended_rate":"account_suspended_sors",
    "service_terminated_rate":"service_terminated_sors",
    "service_suspended_rate":"service_suspended_sors",
    "account_or_service_restriction_rate":"account_or_service_restriction_sors",
}

def sha1_file(path):
    h = hashlib.sha1()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url, dest, timeout=300):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest,"wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk)

def fetch_sha1(url):
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        token = r.text.strip().split()[0].lower()
        return token if len(token)==40 else None
    except requests.RequestException:
        return None

def text_contains(s, pattern):
    return s.fillna("").astype(str).str.contains(pattern, case=False, regex=True)

def bool_true(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.fillna("").astype(str).str.strip().str.lower().isin(["true","1","yes"])

def weighted_count(df, mask):
    c = pd.to_numeric(df["count"], errors="coerce").fillna(0)
    return int(c.loc[mask].sum())

def reduce_period(df, period_col, out_name):
    rows=[]
    for (period, platform, category), g in df.groupby([period_col,"platform_name","category"], dropna=False):
        fully = g["automated_decision"].astype(str).eq("AUTOMATED_DECISION_FULLY")
        partial = g["automated_decision"].astype(str).eq("AUTOMATED_DECISION_PARTIALLY")
        not_auto = g["automated_decision"].astype(str).eq("AUTOMATED_DECISION_NOT_AUTOMATED")
        aterm = text_contains(g["decision_account"], "TERMINAT")
        asusp = text_contains(g["decision_account"], "SUSPEND")
        sterm = text_contains(g["decision_provision"], "TERMINAT")
        ssusp = text_contains(g["decision_provision"], "SUSPEND")
        crem = bool_true(g["DECISION_VISIBILITY_CONTENT_REMOVED"])
        anyrestr = aterm|asusp|sterm|ssusp
        rows.append({
            out_name: period, "platform_name":platform, "category":category,
            "represented_sors": int(pd.to_numeric(g["count"], errors="coerce").fillna(0).sum()),
            "fully_auto_sors": weighted_count(g, fully),
            "partially_auto_sors": weighted_count(g, partial),
            "not_auto_sors": weighted_count(g, not_auto),
            "content_removed_sors": weighted_count(g, crem),
            "account_terminated_sors": weighted_count(g, aterm),
            "account_suspended_sors": weighted_count(g, asusp),
            "service_terminated_sors": weighted_count(g, sterm),
            "service_suspended_sors": weighted_count(g, ssusp),
            "account_or_service_restriction_sors": weighted_count(g, anyrestr),
        })
    return pd.DataFrame(rows)

def add_rates(df):
    out=df.copy()
    den=out["represented_sors"].replace(0,np.nan)
    for rate,num in RATE_NUMERATORS.items():
        out[rate]=out[num]/den
    return out

def complete_week(ws):
    return bool(ws >= STUDY_START and ws + pd.Timedelta(days=6) <= STUDY_END)

def write_dictionary(path):
    rows = [
        ("week_start","date","Monday starting date of the reporting week.",""),
        ("complete_study_week","boolean","True when the full Monday-Sunday week falls within the study period.",""),
        ("platform_name","categorical","Platform name reported in the DSA data.",""),
        ("category","categorical","Broad DSA violation category.",""),
        ("represented_sors","integer","Number of Statements of Reasons represented by aggregate rows.","count"),
        ("fully_auto_sors","integer","Represented SoRs with a fully automated final decision.","count"),
        ("partially_auto_sors","integer","Represented SoRs with a partially automated final decision.","count"),
        ("not_auto_sors","integer","Represented SoRs with a non-automated final decision.","count"),
        ("content_removed_sors","integer","Represented SoRs where the content-removal flag is true.","count"),
        ("account_terminated_sors","integer","Represented SoRs with an account termination action.","count"),
        ("account_suspended_sors","integer","Represented SoRs with an account suspension action.","count"),
        ("service_terminated_sors","integer","Represented SoRs with a service/provision termination action.","count"),
        ("service_suspended_sors","integer","Represented SoRs with a service/provision suspension action.","count"),
        ("account_or_service_restriction_sors","integer","Represented SoRs with account or service suspension/termination.","count"),
        ("fully_automated_rate","float","fully_auto_sors / represented_sors.","0-1"),
        ("partially_automated_rate","float","partially_auto_sors / represented_sors.","0-1"),
        ("not_automated_rate","float","not_auto_sors / represented_sors.","0-1"),
        ("content_removed_rate","float","content_removed_sors / represented_sors.","0-1"),
        ("account_terminated_rate","float","account_terminated_sors / represented_sors.","0-1"),
        ("account_suspended_rate","float","account_suspended_sors / represented_sors.","0-1"),
        ("service_terminated_rate","float","service_terminated_sors / represented_sors.","0-1"),
        ("service_suspended_rate","float","service_suspended_sors / represented_sors.","0-1"),
        ("account_or_service_restriction_rate","float","account_or_service_restriction_sors / represented_sors.","0-1"),
        ("study_start","date","Inclusive study-period start date (summary file only).",""),
        ("study_end","date","Inclusive study-period end date (summary file only).",""),
    ]
    pd.DataFrame(rows,columns=["variable","type","description","unit_or_range"]).to_csv(path,index=False)

def main():
    DATA_DIR.mkdir(exist_ok=True); META_DIR.mkdir(exist_ok=True); CACHE_DIR.mkdir(exist_ok=True)
    weekly_parts=[]; summary_parts=[]; manifest=[]
    build_ts=datetime.now(timezone.utc).isoformat()

    for month in MONTHS:
        key=month[:7]
        fname=f"aggregated-simple-{month}-parquet.zip"
        url=f"{DSA_BASE_URL}/{fname}"
        sha_url=f"{url}.sha1"
        zpath=CACHE_DIR/fname
        edir=CACHE_DIR/key
        ppath=edir/"aggregated-simple.parquet"

        print(f"[{key}] downloading...")
        download(url,zpath)
        expected=fetch_sha1(sha_url)
        actual=sha1_file(zpath)
        status="not_available"
        if expected:
            status="verified" if expected==actual else "FAILED"
            if status=="FAILED":
                raise ValueError(f"SHA-1 mismatch for {fname}")

        edir.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(edir)
        if not ppath.exists():
            raise FileNotFoundError(ppath)

        df=pd.read_parquet(ppath,columns=NEEDED_COLUMNS)
        df=df[df["platform_name"].isin(SELECTED_PLATFORMS)].copy()
        df["created_at"]=pd.to_datetime(df["created_at"],errors="coerce",utc=True)
        if df["created_at"].isna().any():
            raise ValueError(f"{key}: unparseable created_at values")

        df["created_date"]=df["created_at"].dt.tz_localize(None).dt.normalize()
        df=df[(df["created_date"]>=STUDY_START)&(df["created_date"]<=STUDY_END)].copy()
        df["week_start"]=(df["created_at"].dt.tz_localize(None).dt.to_period("W-SUN")
                          .apply(lambda p:p.start_time))
        weekly_parts.append(reduce_period(df,"week_start","week_start"))

        df["study_period"]="2026-03-01_to_2026-08-31"
        summary_parts.append(reduce_period(df,"study_period","study_period"))

        manifest.append({
            "source_month":key,"source_filename":fname,"source_url":url,"sha1_url":sha_url,
            "official_sha1":expected or "","downloaded_sha1":actual,
            "checksum_status":status,"selected_aggregate_rows":len(df),
            "represented_sors":int(df["count"].sum()),"build_timestamp_utc":build_ts
        })

        del df; gc.collect()
        shutil.rmtree(edir,ignore_errors=True)
        if zpath.exists(): zpath.unlink()

    weekly=pd.concat(weekly_parts,ignore_index=True)
    weekly=(weekly.groupby(["week_start","platform_name","category"],as_index=False)[COUNT_COLUMNS].sum())
    assert not weekly.duplicated(["week_start","platform_name","category"]).any()
    weekly=add_rates(weekly)
    weekly["complete_study_week"]=weekly["week_start"].apply(complete_week)
    weekly=weekly.sort_values(["week_start","platform_name","category"]).reset_index(drop=True)

    summary=pd.concat(summary_parts,ignore_index=True)
    summary=(summary.groupby(["platform_name","category"],as_index=False)[COUNT_COLUMNS].sum())
    summary=add_rates(summary)
    summary["study_start"]=STUDY_START.date().isoformat()
    summary["study_end"]=STUDY_END.date().isoformat()
    summary=summary.sort_values(["platform_name","category"]).reset_index(drop=True)

    weekly.to_csv(DATA_DIR/"dsa_weekly_segments.csv",index=False)
    summary.to_csv(DATA_DIR/"dsa_segment_summary.csv",index=False)
    pd.DataFrame(manifest).to_csv(META_DIR/"source_manifest.csv",index=False)
    write_dictionary(DATA_DIR/"data_dictionary.csv")

    print("Build complete")
    print("weekly rows:",len(weekly))
    print("complete study weeks:",weekly.loc[weekly["complete_study_week"],"week_start"].nunique())
    print("summary rows:",len(summary))
    print("represented SoRs:",int(summary["represented_sors"].sum()))

if __name__=="__main__":
    main()
