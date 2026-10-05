#!/usr/bin/env python3

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "synopsis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEEKLY_FILE = DATA_DIR / "dsa_weekly_segments.csv"
SUMMARY_FILE = DATA_DIR / "dsa_segment_summary.csv"


# ============================================================
# FROZEN ANALYSIS CONFIGURATION
# ============================================================

OTHER_CATEGORY = "STATEMENT_CATEGORY_OTHER_VIOLATION_TC"

# RQ2
LARGE_DIVERGENCE_THRESHOLD = 0.25

# RQ3
MIN_WEEKLY_VOLUME = 1000
MIN_RELIABLE_COMPARISONS = 12
ROBUST_Z_THRESHOLD = 3.5
MIN_ABSOLUTE_CHANGE = 0.10

RQ3_VOLUME_SENSITIVITY = [500, 1000, 5000]

# RQ4
RQ4_FEATURES = [
    "log1p_represented_sors",
    "fully_automated_rate",
    "content_removed_rate",
    "account_or_service_restriction_rate",
    "weekly_peer_divergence_abs",
    "automation_change_abs",
]

N_ESTIMATORS = 300
CONTAMINATION = 0.05

SEEDS = [
    11,
    22,
    33,
    44,
    55,
    66,
    77,
    88,
    99,
    111,
]

TOP_K = 20
STABLE_MIN_SEEDS = 8

TOP_K_SENSITIVITY = [10, 20, 30]
STABILITY_THRESHOLD_SENSITIVITY = [7, 8, 9]
CONTAMINATION_SENSITIVITY = [0.03, 0.05, 0.10]


# ============================================================
# VALIDATION BENCHMARKS
# Comparison only - never tune parameters to force these.
# RQ1 uses the final full-period March 1-August 31 design.
# Earlier feasibility used complete weeks only and produced ~0.914.
# ============================================================

EXPECTED = {
    "rq1_categories_4plus": 13,
    "rq1_median_rate_range": 0.928,

    "rq2_comparable_segments": 86,
    "rq2_median_abs_divergence": 0.356,
    "rq2_large_divergence_segments": 50,

    "rq3_reliable_observations": 1203,
    "rq3_eligible_series": 50,
    "rq3_alerts": 30,
    "rq3_affected_series": 18,

    "rq4_eligible_observations": 1130,
    "rq4_stable_anomalies": 18,
    "rq4_selected_10_of_10": 17,
    "rq4_selected_9_of_10": 1,

    "rq4_overlap_volume": 0,
    "rq4_overlap_peer": 0,
    "rq4_overlap_temporal": 12,
}


# ============================================================
# HELPERS
# ============================================================

def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )


def jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def leave_one_out_median(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
) -> tuple[pd.Series, pd.Series]:

    medians = pd.Series(np.nan, index=df.index, dtype=float)
    support = pd.Series(0, index=df.index, dtype=int)

    for _, group in df.groupby(
        group_cols,
        sort=False,
        dropna=False,
    ):
        for idx in group.index:
            peers = (
                group.loc[group.index != idx, value_col]
                .dropna()
            )

            support.loc[idx] = len(peers)

            if len(peers) > 0:
                medians.loc[idx] = float(peers.median())

    return medians, support


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    weekly = pd.read_csv(WEEKLY_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    weekly["week_start"] = pd.to_datetime(
        weekly["week_start"]
    )

    weekly["complete_study_week"] = normalize_boolean(
        weekly["complete_study_week"]
    )

    return weekly, summary


# ============================================================
# DATA-LAYER VALIDATION
# ============================================================


def validate_data(
    weekly: pd.DataFrame,
    summary: pd.DataFrame,
):

    weekly_key = [
        "week_start",
        "platform_name",
        "category",
    ]

    summary_key = [
        "platform_name",
        "category",
    ]

    expected_platforms = {
        "Instagram",
        "TikTok",
        "YouTube",
        "X",
        "Snapchat",
        "LinkedIn",
        "Roblox",
    }

    required_weekly = [
        "week_start",
        "complete_study_week",
        "platform_name",
        "category",
        "represented_sors",
        "fully_automated_rate",
        "content_removed_rate",
        "account_or_service_restriction_rate",
    ]

    required_summary = [
        "platform_name",
        "category",
        "represented_sors",
        "fully_auto_sors",
        "fully_automated_rate",
    ]

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    missing_weekly_columns = [
        c for c in required_weekly
        if c not in weekly.columns
    ]

    missing_summary_columns = [
        c for c in required_summary
        if c not in summary.columns
    ]

    assert not missing_weekly_columns, (
        "Missing required weekly columns: "
        f"{missing_weekly_columns}"
    )

    assert not missing_summary_columns, (
        "Missing required summary columns: "
        f"{missing_summary_columns}"
    )

    # --------------------------------------------------------
    # Key uniqueness
    # --------------------------------------------------------

    weekly_duplicates = int(
        weekly.duplicated(
            weekly_key
        ).sum()
    )

    summary_duplicates = int(
        summary.duplicated(
            summary_key
        ).sum()
    )

    # --------------------------------------------------------
    # Rate bounds in BOTH published datasets
    # --------------------------------------------------------

    weekly_rate_columns = [
        c for c in weekly.columns
        if c.endswith("_rate")
    ]

    summary_rate_columns = [
        c for c in summary.columns
        if c.endswith("_rate")
    ]

    weekly_invalid_rates = int(
        sum(
            (
                ~weekly[c]
                .dropna()
                .between(0, 1)
            ).sum()
            for c in weekly_rate_columns
        )
    )

    summary_invalid_rates = int(
        sum(
            (
                ~summary[c]
                .dropna()
                .between(0, 1)
            ).sum()
            for c in summary_rate_columns
        )
    )

    # --------------------------------------------------------
    # Missingness in fields required for analysis
    # --------------------------------------------------------

    weekly_required_missing = int(
        weekly[
            required_weekly
        ].isna().sum().sum()
    )

    summary_required_missing = int(
        summary[
            required_summary
        ].isna().sum().sum()
    )

    # --------------------------------------------------------
    # Reconciliation
    # --------------------------------------------------------

    weekly_total = int(
        weekly[
            "represented_sors"
        ].sum()
    )

    summary_total = int(
        summary[
            "represented_sors"
        ].sum()
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    observed_platforms = set(
        weekly[
            "platform_name"
        ].dropna().unique()
    )

    platform_coverage_matches = (
        observed_platforms
        == expected_platforms
    )

    complete_week_starts = (
        weekly.loc[
            weekly[
                "complete_study_week"
            ],
            "week_start",
        ]
        .drop_duplicates()
        .sort_values()
    )

    results = {
        "weekly_rows":
            int(len(weekly)),

        "summary_rows":
            int(len(summary)),

        "platforms":
            int(
                weekly[
                    "platform_name"
                ].nunique()
            ),

        "categories":
            int(
                weekly[
                    "category"
                ].nunique()
            ),

        "complete_weeks":
            int(
                len(
                    complete_week_starts
                )
            ),

        "first_complete_week":
            str(
                complete_week_starts.min().date()
            ),

        "last_complete_week":
            str(
                complete_week_starts.max().date()
            ),

        "platform_coverage_matches":
            bool(
                platform_coverage_matches
            ),

        "weekly_duplicate_keys":
            weekly_duplicates,

        "summary_duplicate_keys":
            summary_duplicates,

        "weekly_required_missing":
            weekly_required_missing,

        "summary_required_missing":
            summary_required_missing,

        "weekly_invalid_rate_cells":
            weekly_invalid_rates,

        "summary_invalid_rate_cells":
            summary_invalid_rates,

        "weekly_represented_sors":
            weekly_total,

        "summary_represented_sors":
            summary_total,

        "totals_reconcile":
            bool(
                weekly_total
                == summary_total
            ),
    }

    assert weekly_duplicates == 0
    assert summary_duplicates == 0

    assert weekly_required_missing == 0
    assert summary_required_missing == 0

    assert weekly_invalid_rates == 0
    assert summary_invalid_rates == 0

    assert platform_coverage_matches

    assert len(
        complete_week_starts
    ) == 26

    assert (
        complete_week_starts.min()
        == pd.Timestamp(
            "2026-03-02"
        )
    )

    assert (
        complete_week_starts.max()
        == pd.Timestamp(
            "2026-08-24"
        )
    )

    assert weekly_total == summary_total

    return results
# ============================================================
# RQ1
# AUTOMATION EXPOSURE AND CONCENTRATION
# ============================================================

def run_rq1(summary: pd.DataFrame):

    rq1 = summary.copy()

    # Total fully automated represented SoRs across all
    # platform-category segments.
    total_fully_auto = rq1["fully_auto_sors"].sum()

    # Exposure share for each platform-category segment.
    rq1["automated_exposure_share"] = (
        rq1["fully_auto_sors"] / total_fully_auto
    )

    # Rank segments from greatest to least automated exposure.
    rq1 = (
        rq1.sort_values(
            "automated_exposure_share",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    rq1["cumulative_exposure_share"] = (
        rq1["automated_exposure_share"].cumsum()
    )

    # Herfindahl-Hirschman Index.
    hhi = float(
        np.square(
            rq1["automated_exposure_share"]
        ).sum()
    )

    # Cross-platform category coverage.
    category_coverage = (
        rq1.groupby("category")["platform_name"]
        .nunique()
        .rename("platform_count")
    )

    # Cross-platform range of fully automated rates.
    # This reproduces the original feasibility check.
    automation_range = (
        rq1.groupby("category")["fully_automated_rate"]
        .agg(
            min_rate="min",
            max_rate="max",
        )
    )

    automation_range["rate_range"] = (
        automation_range["max_rate"]
        - automation_range["min_rate"]
    )

    category_stats = (
        automation_range
        .join(category_coverage)
        .reset_index()
    )

    categories_4plus = int(
        (
            category_stats["platform_count"] >= 4
        ).sum()
    )

    median_rate_range = float(
        category_stats["rate_range"].median()
    )

    # --------------------------------------------------------
    # Sensitivity analysis excluding OTHER_VIOLATION_TC
    # --------------------------------------------------------

    non_other = rq1.loc[
        rq1["category"] != OTHER_CATEGORY
    ].copy()

    non_other_total = (
        non_other["fully_auto_sors"].sum()
    )

    non_other["automated_exposure_share"] = (
        non_other["fully_auto_sors"]
        / non_other_total
    )

    hhi_excluding_other = float(
        np.square(
            non_other["automated_exposure_share"]
        ).sum()
    )

    # --------------------------------------------------------
    # Save RQ1 outputs
    # --------------------------------------------------------

    rq1.to_csv(
        OUTPUT_DIR / "rq1_segment_exposure.csv",
        index=False,
    )

    category_stats.to_csv(
        OUTPUT_DIR / "rq1_category_variation.csv",
        index=False,
    )

    metrics = {
        "segment_count": int(len(rq1)),
        "categories_4plus": categories_4plus,
        "median_rate_range": median_rate_range,
        "hhi": hhi,
        "hhi_excluding_other": hhi_excluding_other,
    }

    return rq1, metrics


# ============================================================
# RQ2
# SAME-CATEGORY PEER DIVERGENCE
# ============================================================

def run_rq2(summary: pd.DataFrame):

    rq2 = summary.copy()

    # Leave-one-platform-out median among other platforms
    # reporting the same broad violation category.
    (
        rq2["peer_median_automation_rate"],
        rq2["peer_support_count"],
    ) = leave_one_out_median(
        rq2,
        ["category"],
        "fully_automated_rate",
    )

    rq2["peer_divergence_signed"] = (
        rq2["fully_automated_rate"]
        - rq2["peer_median_automation_rate"]
    )

    rq2["peer_divergence_abs"] = (
        rq2["peer_divergence_signed"].abs()
    )

    # Operational audit flag.
    # This is a review threshold, not ground truth.
    rq2["large_divergence_flag"] = (
        rq2["peer_divergence_abs"]
        > LARGE_DIVERGENCE_THRESHOLD
    )

    comparable = rq2.dropna(
        subset=["peer_divergence_signed"]
    ).copy()

    signed = comparable["peer_divergence_signed"]

    # --------------------------------------------------------
    # Secondary inferential check
    #
    # The synopsis deliberately treats effect magnitude
    # and confidence intervals as primary. The t-test is
    # supplementary because same-category observations are
    # not fully independent.
    # --------------------------------------------------------

    t_result = stats.ttest_1samp(
        signed,
        popmean=0.0,
        nan_policy="omit",
    )

    n = int(signed.notna().sum())
    mean_signed = float(signed.mean())

    if n > 1:

        sem = float(
            stats.sem(
                signed,
                nan_policy="omit",
            )
        )

        t_critical = float(
            stats.t.ppf(
                0.975,
                df=n - 1,
            )
        )

        ci_low = float(
            mean_signed
            - t_critical * sem
        )

        ci_high = float(
            mean_signed
            + t_critical * sem
        )

    else:

        ci_low = np.nan
        ci_high = np.nan

    # --------------------------------------------------------
    # Save RQ2 output
    # --------------------------------------------------------

    rq2.to_csv(
        OUTPUT_DIR / "rq2_peer_divergence.csv",
        index=False,
    )

    metrics = {
        "comparable_segments": int(len(comparable)),

        "median_abs_divergence": float(
            comparable[
                "peer_divergence_abs"
            ].median()
        ),

        "large_divergence_segments": int(
            comparable[
                "large_divergence_flag"
            ].sum()
        ),

        "mean_signed_divergence": mean_signed,

        "signed_divergence_ci95_low": ci_low,

        "signed_divergence_ci95_high": ci_high,

        "secondary_t_statistic": float(
            t_result.statistic
        ),

        "secondary_p_value": float(
            t_result.pvalue
        ),
    }

    return rq2, metrics


# ============================================================
# RQ3
# ROBUST TEMPORAL INSTABILITY
# ============================================================

def construct_rq3(
    weekly: pd.DataFrame,
    min_weekly_volume: int,
):

    # RQ3 uses only complete Monday-Sunday study weeks.
    temporal = weekly.loc[
        weekly["complete_study_week"]
    ].copy()

    temporal = temporal.sort_values(
        [
            "platform_name",
            "category",
            "week_start",
        ]
    )

    group_cols = [
        "platform_name",
        "category",
    ]

    grouped = temporal.groupby(
        group_cols,
        sort=False,
    )

    # Previous observation within the same platform-category series.
    temporal["previous_week_start"] = (
        grouped["week_start"].shift(1)
    )

    temporal["previous_week_volume"] = (
        grouped["represented_sors"].shift(1)
    )

    # Week-over-week change in fully automated rate.
    temporal["automation_change"] = (
        grouped["fully_automated_rate"].diff()
    )

    temporal["automation_change_abs"] = (
        temporal["automation_change"].abs()
    )

    temporal["days_since_previous"] = (
        temporal["week_start"]
        - temporal["previous_week_start"]
    ).dt.days

    # Reliable comparison:
    # current and previous week both meet volume threshold,
    # and observations are exactly seven days apart.
    candidate = temporal.loc[
        (
            temporal["represented_sors"]
            >= min_weekly_volume
        )
        &
        (
            temporal["previous_week_volume"]
            >= min_weekly_volume
        )
        &
        (
            temporal["days_since_previous"] == 7
        )
    ].copy()

    # Number of reliable comparisons per series.
    reliable_counts = (
        candidate.groupby(group_cols)
        .size()
        .rename("reliable_comparisons")
    )

    eligible = (
        reliable_counts.loc[
            reliable_counts
            >= MIN_RELIABLE_COMPARISONS
        ]
        .reset_index()
    )

    reliable = candidate.merge(
        eligible,
        on=group_cols,
        how="inner",
    )

    # --------------------------------------------------------
    # Robust within-series change score
    # --------------------------------------------------------

    median_change = (
        reliable.groupby(group_cols)[
            "automation_change"
        ]
        .transform("median")
    )

    absolute_deviation = (
        reliable["automation_change"]
        - median_change
    ).abs()

    mad = (
        absolute_deviation.groupby(
            [
                reliable["platform_name"],
                reliable["category"],
            ]
        )
        .transform("median")
    )

    reliable["robust_change_score"] = np.where(
        mad > 0,
        (
            0.6745
            * (
                reliable["automation_change"]
                - median_change
            )
            / mad
        ),
        np.nan,
    )

    # Dual threshold:
    # statistically unusual AND operationally material.
    reliable["temporal_alert"] = (
        (
            reliable[
                "robust_change_score"
            ].abs()
            >= ROBUST_Z_THRESHOLD
        )
        &
        (
            reliable[
                "automation_change_abs"
            ]
            >= MIN_ABSOLUTE_CHANGE
        )
    )

    return reliable


def run_rq3(
    weekly: pd.DataFrame,
):

    # Primary configuration: 1,000 SoRs.
    reliable = construct_rq3(
        weekly,
        MIN_WEEKLY_VOLUME,
    )

    alerts = reliable.loc[
        reliable["temporal_alert"]
    ].copy()

    affected_series = int(
        alerts[
            [
                "platform_name",
                "category",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # --------------------------------------------------------
    # Volume-threshold sensitivity
    # --------------------------------------------------------

    sensitivity_rows = []

    for threshold in RQ3_VOLUME_SENSITIVITY:

        temp = construct_rq3(
            weekly,
            threshold,
        )

        temp_alerts = temp.loc[
            temp["temporal_alert"]
        ]

        sensitivity_rows.append(
            {
                "minimum_weekly_volume":
                    threshold,

                "reliable_observations":
                    int(len(temp)),

                "eligible_series":
                    int(
                        temp[
                            [
                                "platform_name",
                                "category",
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),

                "alerts":
                    int(len(temp_alerts)),

                "affected_series":
                    int(
                        temp_alerts[
                            [
                                "platform_name",
                                "category",
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),
            }
        )

    sensitivity = pd.DataFrame(
        sensitivity_rows
    )

    # --------------------------------------------------------
    # Save RQ3 outputs
    # --------------------------------------------------------

    reliable.to_csv(
        OUTPUT_DIR
        / "rq3_reliable_weekly.csv",
        index=False,
    )

    alerts.to_csv(
        OUTPUT_DIR
        / "rq3_temporal_alerts.csv",
        index=False,
    )

    sensitivity.to_csv(
        OUTPUT_DIR
        / "rq3_volume_sensitivity.csv",
        index=False,
    )

    metrics = {
        "reliable_observations":
            int(len(reliable)),

        "eligible_series":
            int(
                reliable[
                    [
                        "platform_name",
                        "category",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),

        "reliable_weeks":
            int(
                reliable["week_start"].nunique()
            ),

        "alerts":
            int(len(alerts)),

        "affected_series":
            affected_series,
    }

    return reliable, metrics


# ============================================================
# RQ4
# INTEGRATED EXPLAINABLE AUDIT PRIORITISATION
# PRIMARY MODEL
# ============================================================

def prepare_rq4(
    reliable: pd.DataFrame,
):

    model_df = reliable.copy()

    # --------------------------------------------------------
    # Same-week, same-category leave-one-platform-out
    # peer benchmark.
    #
    # RQ4 starts from the reliable RQ3 analytical population.
    # --------------------------------------------------------

    (
        model_df[
            "weekly_peer_median_automation_rate"
        ],
        model_df[
            "weekly_peer_support_count"
        ],
    ) = leave_one_out_median(
        model_df,
        [
            "week_start",
            "category",
        ],
        "fully_automated_rate",
    )

    model_df[
        "weekly_peer_divergence_abs"
    ] = (
        model_df[
            "fully_automated_rate"
        ]
        - model_df[
            "weekly_peer_median_automation_rate"
        ]
    ).abs()

    # Absolute temporal change was already created by RQ3.
    model_df[
        "automation_change_abs"
    ] = (
        model_df[
            "automation_change"
        ].abs()
    )

    # Log-transform represented moderation volume.
    model_df[
        "log1p_represented_sors"
    ] = np.log1p(
        model_df[
            "represented_sors"
        ]
    )

    # Keep only rows with all six required model features.
    model_df = (
        model_df
        .dropna(
            subset=RQ4_FEATURES
        )
        .reset_index(drop=True)
    )

    return model_df


def fit_primary_rq4(
    model_df: pd.DataFrame,
):

    # --------------------------------------------------------
    # Robust scaling
    # --------------------------------------------------------

    X = model_df[
        RQ4_FEATURES
    ].copy()

    scaler = RobustScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    # --------------------------------------------------------
    # Isolation Forest across frozen seeds
    # --------------------------------------------------------

    selection_count = np.zeros(
        len(model_df),
        dtype=int,
    )

    seed_scores = []

    for seed in SEEDS:

        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=CONTAMINATION,
            random_state=seed,
            n_jobs=-1,
        )

        model.fit(
            X_scaled
        )

        # sklearn decision_function:
        # lower = more anomalous.
        decision_scores = (
            model.decision_function(
                X_scaled
            )
        )

        # Convert direction so:
        # higher = greater audit priority.
        priority_scores = (
            -decision_scores
        )

        seed_scores.append(
            priority_scores
        )

        # Select the 20 highest-priority observations.
        top_idx = np.argsort(
            -priority_scores,
            kind="stable",
        )[:TOP_K]

        selection_count[
            top_idx
        ] += 1

    # --------------------------------------------------------
    # Stable shortlist
    # --------------------------------------------------------

    result = model_df.copy()

    result[
        "seed_inclusion_count"
    ] = selection_count

    result[
        "stable_anomaly"
    ] = (
        result[
            "seed_inclusion_count"
        ]
        >= STABLE_MIN_SEEDS
    )

    # Average continuous priority score across seeds.
    score_matrix = np.vstack(
        seed_scores
    )

    result[
        "anomaly_score"
    ] = score_matrix.mean(
        axis=0
    )

    result[
        "anomaly_rank"
    ] = (
        result[
            "anomaly_score"
        ]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Simple feature-context explanation
    #
    # This is NOT causal attribution.
    # It identifies the feature with the largest absolute
    # robust-scaled deviation for the observation.
    # --------------------------------------------------------

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=RQ4_FEATURES,
        index=result.index,
    )

    result[
        "largest_deviation_feature"
    ] = (
        scaled_df.abs()
        .idxmax(axis=1)
    )

    result[
        "largest_abs_scaled_deviation"
    ] = (
        scaled_df.abs()
        .max(axis=1)
    )

    return result


def run_rq4_primary(
    reliable: pd.DataFrame,
):

    model_df = prepare_rq4(
        reliable
    )

    scored = fit_primary_rq4(
        model_df
    )

    stable = scored.loc[
        scored["stable_anomaly"]
    ].copy()

    # --------------------------------------------------------
    # Equal-sized simple baselines
    # --------------------------------------------------------

    stable_idx = set(
        stable.index
    )

    top_volume_idx = set(
        scored.nlargest(
            TOP_K,
            "represented_sors",
        ).index
    )

    top_peer_idx = set(
        scored.nlargest(
            TOP_K,
            "weekly_peer_divergence_abs",
        ).index
    )

    top_temporal_idx = set(
        scored.nlargest(
            TOP_K,
            "automation_change_abs",
        ).index
    )

    overlap_volume = len(
        stable_idx
        & top_volume_idx
    )

    overlap_peer = len(
        stable_idx
        & top_peer_idx
    )

    overlap_temporal = len(
        stable_idx
        & top_temporal_idx
    )

    baseline_comparison = pd.DataFrame(
        [
            {
                "baseline": "Top volume",
                "overlap_count":
                    overlap_volume,
                "jaccard_similarity":
                    jaccard(
                        stable_idx,
                        top_volume_idx,
                    ),
            },
            {
                "baseline":
                    "Top peer divergence",
                "overlap_count":
                    overlap_peer,
                "jaccard_similarity":
                    jaccard(
                        stable_idx,
                        top_peer_idx,
                    ),
            },
            {
                "baseline":
                    "Top temporal change",
                "overlap_count":
                    overlap_temporal,
                "jaccard_similarity":
                    jaccard(
                        stable_idx,
                        top_temporal_idx,
                    ),
            },
        ]
    )

    # --------------------------------------------------------
    # Save primary RQ4 outputs
    # --------------------------------------------------------

    scored.to_csv(
        OUTPUT_DIR
        / "rq4_scored_observations.csv",
        index=False,
    )

    stable.to_csv(
        OUTPUT_DIR
        / "rq4_stable_shortlist.csv",
        index=False,
    )

    baseline_comparison.to_csv(
        OUTPUT_DIR
        / "rq4_baseline_comparison.csv",
        index=False,
    )

    metrics = {
        "eligible_observations":
            int(len(scored)),

        "stable_anomalies":
            int(len(stable)),

        "selected_10_of_10":
            int(
                (
                    scored[
                        "seed_inclusion_count"
                    ]
                    == 10
                ).sum()
            ),

        "selected_9_of_10":
            int(
                (
                    scored[
                        "seed_inclusion_count"
                    ]
                    == 9
                ).sum()
            ),

        "overlap_volume":
            int(overlap_volume),

        "overlap_peer":
            int(overlap_peer),

        "overlap_temporal":
            int(overlap_temporal),
    }

    return scored, stable, baseline_comparison, metrics


# ============================================================
# RQ4 VALIDATION AND SENSITIVITY
# ============================================================

def fit_rq4_variant(
    model_df: pd.DataFrame,
    features: list[str],
    top_k: int = TOP_K,
    stable_min_seeds: int = STABLE_MIN_SEEDS,
    contamination: float = CONTAMINATION,
):

    X = model_df[features].copy()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    selection_count = np.zeros(
        len(model_df),
        dtype=int,
    )

    seed_selected = {}

    for seed in SEEDS:

        model = IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )

        model.fit(X_scaled)

        # Higher value = greater audit priority.
        priority = -model.decision_function(
            X_scaled
        )

        top_idx = np.argsort(
            -priority,
            kind="stable",
        )[:top_k]

        seed_selected[seed] = set(
            top_idx.tolist()
        )

        selection_count[top_idx] += 1

    result = model_df.copy()

    result["seed_inclusion_count"] = (
        selection_count
    )

    result["stable_anomaly"] = (
        selection_count
        >= stable_min_seeds
    )

    return result, seed_selected


def run_rq4_sensitivity(
    reliable: pd.DataFrame,
    primary_scored: pd.DataFrame,
):

    model_df = prepare_rq4(
        reliable
    )

    primary_stable_idx = set(
        primary_scored.index[
            primary_scored[
                "stable_anomaly"
            ]
        ]
    )

    # --------------------------------------------------------
    # A. TOP-K SENSITIVITY
    # --------------------------------------------------------

    topk_rows = []

    for k in TOP_K_SENSITIVITY:

        temp, _ = fit_rq4_variant(
            model_df,
            RQ4_FEATURES,
            top_k=k,
            stable_min_seeds=STABLE_MIN_SEEDS,
            contamination=CONTAMINATION,
        )

        temp_stable_idx = set(
            temp.index[
                temp["stable_anomaly"]
            ]
        )

        topk_rows.append(
            {
                "top_k": k,
                "stable_count":
                    len(temp_stable_idx),

                "overlap_with_primary":
                    len(
                        temp_stable_idx
                        & primary_stable_idx
                    ),

                "jaccard_vs_primary":
                    jaccard(
                        temp_stable_idx,
                        primary_stable_idx,
                    ),
            }
        )

    topk_sensitivity = pd.DataFrame(
        topk_rows
    )

    # --------------------------------------------------------
    # B. STABILITY-THRESHOLD SENSITIVITY
    #
    # Primary rule = >=8 of 10 seeds.
    # Test >=7, >=8, >=9.
    # --------------------------------------------------------

    threshold_rows = []

    for threshold in (
        STABILITY_THRESHOLD_SENSITIVITY
    ):

        candidate_idx = set(
            primary_scored.index[
                primary_scored[
                    "seed_inclusion_count"
                ]
                >= threshold
            ]
        )

        threshold_rows.append(
            {
                "minimum_seed_count":
                    threshold,

                "candidate_count":
                    len(candidate_idx),

                "overlap_with_primary":
                    len(
                        candidate_idx
                        & primary_stable_idx
                    ),

                "jaccard_vs_primary":
                    jaccard(
                        candidate_idx,
                        primary_stable_idx,
                    ),
            }
        )

    stability_threshold_sensitivity = (
        pd.DataFrame(
            threshold_rows
        )
    )

    # --------------------------------------------------------
    # C. FEATURE-SET SENSITIVITY
    #
    # Leave one feature out at a time.
    # This tests whether the shortlist depends heavily on one
    # individual feature.
    # --------------------------------------------------------

    feature_rows = []

    for feature_to_remove in RQ4_FEATURES:

        reduced_features = [
            feature
            for feature in RQ4_FEATURES
            if feature != feature_to_remove
        ]

        temp, _ = fit_rq4_variant(
            model_df,
            reduced_features,
            top_k=TOP_K,
            stable_min_seeds=STABLE_MIN_SEEDS,
            contamination=CONTAMINATION,
        )

        temp_stable_idx = set(
            temp.index[
                temp["stable_anomaly"]
            ]
        )

        feature_rows.append(
            {
                "feature_removed":
                    feature_to_remove,

                "stable_count":
                    len(temp_stable_idx),

                "overlap_with_primary":
                    len(
                        temp_stable_idx
                        & primary_stable_idx
                    ),

                "jaccard_vs_primary":
                    jaccard(
                        temp_stable_idx,
                        primary_stable_idx,
                    ),
            }
        )

    feature_sensitivity = pd.DataFrame(
        feature_rows
    )

    # --------------------------------------------------------
    # D. CONTAMINATION / CANDIDATE-REGION SENSITIVITY
    #
    # Contamination changes the binary Isolation Forest
    # candidate threshold. It is NOT interpreted as model
    # accuracy and is separate from the fixed top-k audit queue.
    # --------------------------------------------------------

    X = model_df[
        RQ4_FEATURES
    ].copy()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    contamination_rows = []

    for contamination in (
        CONTAMINATION_SENSITIVITY
    ):

        candidate_counts = np.zeros(
            len(model_df),
            dtype=int,
        )

        per_seed_counts = []

        for seed in SEEDS:

            model = IsolationForest(
                n_estimators=N_ESTIMATORS,
                contamination=contamination,
                random_state=seed,
                n_jobs=-1,
            )

            model.fit(X_scaled)

            predicted = model.predict(
                X_scaled
            )

            candidate_idx = np.where(
                predicted == -1
            )[0]

            candidate_counts[
                candidate_idx
            ] += 1

            per_seed_counts.append(
                len(candidate_idx)
            )

        stable_candidate_idx = set(
            np.where(
                candidate_counts
                >= STABLE_MIN_SEEDS
            )[0]
            .tolist()
        )

        contamination_rows.append(
            {
                "contamination":
                    contamination,

                "mean_candidates_per_seed":
                    float(
                        np.mean(
                            per_seed_counts
                        )
                    ),

                "min_candidates_per_seed":
                    int(
                        np.min(
                            per_seed_counts
                        )
                    ),

                "max_candidates_per_seed":
                    int(
                        np.max(
                            per_seed_counts
                        )
                    ),

                "stable_candidate_count":
                    len(
                        stable_candidate_idx
                    ),

                "primary_shortlist_inside_stable_candidate_region":
                    len(
                        primary_stable_idx
                        & stable_candidate_idx
                    ),
            }
        )

    contamination_sensitivity = (
        pd.DataFrame(
            contamination_rows
        )
    )

    # --------------------------------------------------------
    # E. SEED-TO-SEED STABILITY
    #
    # Pairwise Jaccard similarity between each seed's
    # top-20 shortlist.
    # --------------------------------------------------------

    _, seed_selected = fit_rq4_variant(
        model_df,
        RQ4_FEATURES,
        top_k=TOP_K,
        stable_min_seeds=STABLE_MIN_SEEDS,
        contamination=CONTAMINATION,
    )

    seed_rows = []

    seed_list = list(SEEDS)

    for i in range(len(seed_list)):

        for j in range(
            i + 1,
            len(seed_list),
        ):

            seed_a = seed_list[i]
            seed_b = seed_list[j]

            seed_rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "jaccard_similarity":
                        jaccard(
                            seed_selected[
                                seed_a
                            ],
                            seed_selected[
                                seed_b
                            ],
                        ),
                }
            )

    seed_stability = pd.DataFrame(
        seed_rows
    )

    # --------------------------------------------------------
    # SAVE VALIDATION OUTPUTS
    # --------------------------------------------------------

    topk_sensitivity.to_csv(
        OUTPUT_DIR
        / "rq4_topk_sensitivity.csv",
        index=False,
    )

    stability_threshold_sensitivity.to_csv(
        OUTPUT_DIR
        / "rq4_stability_threshold_sensitivity.csv",
        index=False,
    )

    feature_sensitivity.to_csv(
        OUTPUT_DIR
        / "rq4_feature_set_sensitivity.csv",
        index=False,
    )

    contamination_sensitivity.to_csv(
        OUTPUT_DIR
        / "rq4_candidate_threshold_sensitivity.csv",
        index=False,
    )

    seed_stability.to_csv(
        OUTPUT_DIR
        / "rq4_seed_pairwise_jaccard.csv",
        index=False,
    )

    sensitivity_metrics = {
        "mean_pairwise_seed_jaccard":
            float(
                seed_stability[
                    "jaccard_similarity"
                ].mean()
            ),

        "min_pairwise_seed_jaccard":
            float(
                seed_stability[
                    "jaccard_similarity"
                ].min()
            ),

        "max_pairwise_seed_jaccard":
            float(
                seed_stability[
                    "jaccard_similarity"
                ].max()
            ),
    }

    return {
        "topk":
            topk_sensitivity,

        "stability_threshold":
            stability_threshold_sensitivity,

        "feature_set":
            feature_sensitivity,

        "candidate_threshold":
            contamination_sensitivity,

        "seed_stability":
            seed_stability,

        "metrics":
            sensitivity_metrics,
    }


# ============================================================
# FINAL VALIDATION / REPRODUCIBILITY AUDIT
# ============================================================


def validate_source_manifest():

    manifest_file = (
        ROOT
        / "metadata"
        / "source_manifest.csv"
    )

    manifest = pd.read_csv(
        manifest_file
    )

    required_columns = [
        "source_month",
        "official_sha1",
        "downloaded_sha1",
        "checksum_status",
        "represented_sors",
    ]

    missing_columns = [
        c for c in required_columns
        if c not in manifest.columns
    ]

    assert not missing_columns, (
        "Missing source-manifest columns: "
        f"{missing_columns}"
    )

    status_verified = (
        manifest[
            "checksum_status"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("verified")
    )

    sha_matches = (
        manifest[
            "official_sha1"
        ]
        .astype(str)
        .str.lower()
        ==
        manifest[
            "downloaded_sha1"
        ]
        .astype(str)
        .str.lower()
    )

    represented_total = int(
        manifest[
            "represented_sors"
        ].sum()
    )

    results = {
        "manifest_rows":
            int(len(manifest)),

        "checksum_rows_verified":
            int(
                status_verified.sum()
            ),

        "sha1_rows_matching":
            int(
                sha_matches.sum()
            ),

        "all_checksums_verified":
            bool(
                status_verified.all()
                and sha_matches.all()
            ),

        "manifest_represented_sors":
            represented_total,
    }

    assert len(manifest) == 6
    assert status_verified.all()
    assert sha_matches.all()

    assert (
        represented_total
        == 430_511_443
    )

    return results


def write_missingness_report(
    weekly: pd.DataFrame,
    summary: pd.DataFrame,
):

    weekly_missing = (
        weekly.isna()
        .sum()
        .rename(
            "missing_count"
        )
        .reset_index()
        .rename(
            columns={
                "index": "variable"
            }
        )
    )

    weekly_missing[
        "dataset"
    ] = "weekly"

    summary_missing = (
        summary.isna()
        .sum()
        .rename(
            "missing_count"
        )
        .reset_index()
        .rename(
            columns={
                "index": "variable"
            }
        )
    )

    summary_missing[
        "dataset"
    ] = "summary"

    missingness = pd.concat(
        [
            weekly_missing,
            summary_missing,
        ],
        ignore_index=True,
    )

    missingness.to_csv(
        OUTPUT_DIR
        / "data_missingness_report.csv",
        index=False,
    )

    return missingness


def benchmark_check(
    rq1_metrics,
    rq2_metrics,
    rq3_metrics,
    rq4_metrics,
):

    actual = {
        "rq1_categories_4plus":
            rq1_metrics[
                "categories_4plus"
            ],

        "rq1_median_rate_range":
            rq1_metrics[
                "median_rate_range"
            ],

        "rq2_comparable_segments":
            rq2_metrics[
                "comparable_segments"
            ],

        "rq2_median_abs_divergence":
            rq2_metrics[
                "median_abs_divergence"
            ],

        "rq2_large_divergence_segments":
            rq2_metrics[
                "large_divergence_segments"
            ],

        "rq3_reliable_observations":
            rq3_metrics[
                "reliable_observations"
            ],

        "rq3_eligible_series":
            rq3_metrics[
                "eligible_series"
            ],

        "rq3_alerts":
            rq3_metrics[
                "alerts"
            ],

        "rq3_affected_series":
            rq3_metrics[
                "affected_series"
            ],

        "rq4_eligible_observations":
            rq4_metrics[
                "eligible_observations"
            ],

        "rq4_stable_anomalies":
            rq4_metrics[
                "stable_anomalies"
            ],

        "rq4_selected_10_of_10":
            rq4_metrics[
                "selected_10_of_10"
            ],

        "rq4_selected_9_of_10":
            rq4_metrics[
                "selected_9_of_10"
            ],

        "rq4_overlap_volume":
            rq4_metrics[
                "overlap_volume"
            ],

        "rq4_overlap_peer":
            rq4_metrics[
                "overlap_peer"
            ],

        "rq4_overlap_temporal":
            rq4_metrics[
                "overlap_temporal"
            ],
    }

    approximate_metrics = {
        "rq1_median_rate_range",
        "rq2_median_abs_divergence",
    }

    rows = []

    for metric, expected in EXPECTED.items():

        observed = actual[
            metric
        ]

        if metric in approximate_metrics:

            matches = (
                abs(
                    observed
                    - expected
                )
                <= 0.01
            )

        else:

            matches = (
                observed
                == expected
            )

        rows.append(
            {
                "metric":
                    metric,

                "validation_benchmark":
                    expected,

                "current_result":
                    observed,

                "matches_benchmark":
                    bool(matches),
            }
        )

    check = pd.DataFrame(
        rows
    )

    check.to_csv(
        OUTPUT_DIR
        / "feasibility_benchmark_check.csv",
        index=False,
    )

    return check


def write_run_manifest():

    manifest = {
        "python_version":
            platform.python_version(),

        "pandas_version":
            pd.__version__,

        "numpy_version":
            np.__version__,

        "scipy_version":
            scipy.__version__,

        "scikit_learn_version":
            sklearn.__version__,

        "rq3_min_weekly_volume":
            MIN_WEEKLY_VOLUME,

        "rq3_min_reliable_comparisons":
            MIN_RELIABLE_COMPARISONS,

        "rq3_robust_z_threshold":
            ROBUST_Z_THRESHOLD,

        "rq3_min_absolute_change":
            MIN_ABSOLUTE_CHANGE,

        "rq3_volume_sensitivity":
            RQ3_VOLUME_SENSITIVITY,

        "rq4_features":
            RQ4_FEATURES,

        "rq4_n_estimators":
            N_ESTIMATORS,

        "rq4_contamination":
            CONTAMINATION,

        "rq4_seeds":
            SEEDS,

        "rq4_top_k":
            TOP_K,

        "rq4_stable_min_seeds":
            STABLE_MIN_SEEDS,

        "rq4_top_k_sensitivity":
            TOP_K_SENSITIVITY,

        "rq4_stability_threshold_sensitivity":
            STABILITY_THRESHOLD_SENSITIVITY,

        "rq4_contamination_sensitivity":
            CONTAMINATION_SENSITIVITY,
    }

    with open(
        OUTPUT_DIR
        / "analysis_run_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    return manifest



# ============================================================
# ANALYSIS DATA DICTIONARY
#
# Documents variables generated by RQ1-RQ4.
# The deterministic data-layer dictionary remains separately
# maintained in data/data_dictionary.csv.
# ============================================================

def write_analysis_data_dictionary():

    rows = [

        # ----------------------------------------------------
        # RQ1
        # ----------------------------------------------------

        {
            "variable":
                "automated_exposure_share",
            "type":
                "float",
            "description":
                "Segment fully automated SoRs divided by total fully automated SoRs across all platform-category segments.",
            "unit_or_range":
                "0-1",
            "generated_in":
                "RQ1",
            "relevance":
                "Measures each platform-category segment's share of total fully automated moderation exposure.",
        },

        {
            "variable":
                "cumulative_exposure_share",
            "type":
                "float",
            "description":
                "Cumulative automated exposure share after sorting platform-category segments from highest to lowest exposure.",
            "unit_or_range":
                "0-1",
            "generated_in":
                "RQ1",
            "relevance":
                "Shows how concentrated fully automated moderation exposure is among the highest-volume segments.",
        },

        {
            "variable":
                "platform_count",
            "type":
                "integer",
            "description":
                "Number of selected platforms reporting the broad violation category.",
            "unit_or_range":
                "count",
            "generated_in":
                "RQ1",
            "relevance":
                "Describes cross-platform category coverage.",
        },

        {
            "variable":
                "min_rate",
            "type":
                "float",
            "description":
                "Minimum fully automated moderation rate observed across platforms for a category.",
            "unit_or_range":
                "0-1",
            "generated_in":
                "RQ1",
            "relevance":
                "Lower bound used to measure cross-platform variation within a category.",
        },

        {
            "variable":
                "max_rate",
            "type":
                "float",
            "description":
                "Maximum fully automated moderation rate observed across platforms for a category.",
            "unit_or_range":
                "0-1",
            "generated_in":
                "RQ1",
            "relevance":
                "Upper bound used to measure cross-platform variation within a category.",
        },

        {
            "variable":
                "rate_range",
            "type":
                "float",
            "description":
                "Difference between the maximum and minimum fully automated moderation rate across platforms within a category.",
            "unit_or_range":
                "0-1",
            "generated_in":
                "RQ1",
            "relevance":
                "Quantifies cross-platform variation in automation exposure.",
        },


        # ----------------------------------------------------
        # RQ2
        # ----------------------------------------------------

        {
            "variable":
                "peer_median_automation_rate",
            "type":
                "float",
            "description":
                "Median fully automated rate of other selected platforms reporting the same category, excluding the focal platform.",
            "unit_or_range":
                "0-1",
            "generated_in":
                "RQ2",
            "relevance":
                "Provides the leave-one-platform-out same-category peer benchmark.",
        },

        {
            "variable":
                "peer_support_count",
            "type":
                "integer",
            "description":
                "Number of peer platforms contributing to the leave-one-platform-out median.",
            "unit_or_range":
                "count",
            "generated_in":
                "RQ2",
            "relevance":
                "Reports the amount of peer support behind each divergence estimate.",
        },

        {
            "variable":
                "peer_divergence_signed",
            "type":
                "float",
            "description":
                "Focal platform fully automated rate minus the leave-one-platform-out same-category peer median.",
            "unit_or_range":
                "-1 to 1",
            "generated_in":
                "RQ2",
            "relevance":
                "Preserves the direction of deviation from same-category peers.",
        },

        {
            "variable":
                "peer_divergence_abs",
            "type":
                "float",
            "description":
                "Absolute value of signed peer divergence.",
            "unit_or_range":
                "0-1",
            "generated_in":
                "RQ2",
            "relevance":
                "Measures the magnitude of same-category peer divergence irrespective of direction.",
        },

        {
            "variable":
                "large_divergence_flag",
            "type":
                "boolean",
            "description":
                "True when absolute peer divergence exceeds 0.25.",
            "unit_or_range":
                "True/False",
            "generated_in":
                "RQ2",
            "relevance":
                "Operational flag for segments with more than 25 percentage-point divergence from peers.",
        },


        # ----------------------------------------------------
        # RQ3
        # ----------------------------------------------------

        {
            "variable":
                "previous_week_start",
            "type":
                "date",
            "description":
                "Start date of the preceding observation within the same platform-category series.",
            "unit_or_range":
                "date",
            "generated_in":
                "RQ3",
            "relevance":
                "Supports verification of exact seven-day temporal adjacency.",
        },

        {
            "variable":
                "previous_week_volume",
            "type":
                "float",
            "description":
                "Represented SoR volume of the previous weekly observation in the same platform-category series.",
            "unit_or_range":
                "count",
            "generated_in":
                "RQ3",
            "relevance":
                "Used with current-week volume to determine whether a week-to-week comparison is reliable.",
        },

        {
            "variable":
                "automation_change",
            "type":
                "float",
            "description":
                "Current week's fully automated rate minus the previous week's fully automated rate for the same platform-category series.",
            "unit_or_range":
                "-1 to 1",
            "generated_in":
                "RQ3",
            "relevance":
                "Primary signed measure of week-over-week automation-rate change.",
        },

        {
            "variable":
                "automation_change_abs",
            "type":
                "float",
            "description":
                "Absolute value of the week-over-week automation-rate change.",
            "unit_or_range":
                "0-1",
            "generated_in":
                "RQ3/RQ4",
            "relevance":
                "Measures operational magnitude of temporal change and is also an RQ4 model feature.",
        },

        {
            "variable":
                "days_since_previous",
            "type":
                "integer",
            "description":
                "Number of days between the current and previous weekly observations in the same series.",
            "unit_or_range":
                "days",
            "generated_in":
                "RQ3",
            "relevance":
                "Ensures temporal comparisons use exactly seven-day adjacency.",
        },

        {
            "variable":
                "reliable_comparisons",
            "type":
                "integer",
            "description":
                "Number of reliable week-to-week comparisons available for the platform-category series under the selected volume threshold.",
            "unit_or_range":
                "count",
            "generated_in":
                "RQ3",
            "relevance":
                "Determines whether a series satisfies the minimum 12-comparison eligibility rule.",
        },

        {
            "variable":
                "robust_change_score",
            "type":
                "float",
            "description":
                "Within-series robust temporal score calculated as 0.6745 times change minus median change divided by MAD.",
            "unit_or_range":
                "unbounded; NaN if MAD = 0",
            "generated_in":
                "RQ3",
            "relevance":
                "Identifies changes unusual relative to the platform-category series' own temporal behavior.",
        },

        {
            "variable":
                "temporal_alert",
            "type":
                "boolean",
            "description":
                "True when absolute robust change score is at least 3.5 and absolute automation change is at least 0.10.",
            "unit_or_range":
                "True/False",
            "generated_in":
                "RQ3",
            "relevance":
                "Operational temporal-instability signal requiring both statistical unusualness and material change.",
        },


        # ----------------------------------------------------
        # RQ4 FEATURE CONSTRUCTION
        # ----------------------------------------------------

        {
            "variable":
                "weekly_peer_median_automation_rate",
            "type":
                "float",
            "description":
                "Same-week same-category leave-one-platform-out median fully automated rate among eligible RQ4 observations.",
            "unit_or_range":
                "0-1",
            "generated_in":
                "RQ4",
            "relevance":
                "Provides the weekly peer benchmark used to construct RQ4 peer divergence.",
        },

        {
            "variable":
                "weekly_peer_support_count",
            "type":
                "integer",
            "description":
                "Number of peer platforms contributing to the same-week same-category peer median.",
            "unit_or_range":
                "count",
            "generated_in":
                "RQ4",
            "relevance":
                "Reports support behind the weekly peer comparison.",
        },

        {
            "variable":
                "weekly_peer_divergence_abs",
            "type":
                "float",
            "description":
                "Absolute difference between the focal observation's fully automated rate and the same-week same-category leave-one-platform-out peer median.",
            "unit_or_range":
                "0-1",
            "generated_in":
                "RQ4",
            "relevance":
                "One of the six Isolation Forest features.",
        },

        {
            "variable":
                "log1p_represented_sors",
            "type":
                "float",
            "description":
                "Natural logarithm of one plus represented SoR volume.",
            "unit_or_range":
                "non-negative transformed value",
            "generated_in":
                "RQ4",
            "relevance":
                "Volume feature transformed to reduce scale dominance in the multivariate model.",
        },


        # ----------------------------------------------------
        # RQ4 MODEL OUTPUT
        # ----------------------------------------------------

        {
            "variable":
                "seed_inclusion_count",
            "type":
                "integer",
            "description":
                "Number of the ten Isolation Forest runs in which the observation appears in the top-20 audit-priority list.",
            "unit_or_range":
                "0-10",
            "generated_in":
                "RQ4",
            "relevance":
                "Primary measure of selection stability across random seeds.",
        },

        {
            "variable":
                "stable_anomaly",
            "type":
                "boolean",
            "description":
                "True when an observation appears in the top-20 list in at least 8 of 10 Isolation Forest runs.",
            "unit_or_range":
                "True/False",
            "generated_in":
                "RQ4",
            "relevance":
                "Defines the final stable audit-priority shortlist.",
        },

        {
            "variable":
                "anomaly_score",
            "type":
                "float",
            "description":
                "Mean negative Isolation Forest decision-function score across the ten seeded runs, oriented so higher values indicate greater audit priority.",
            "unit_or_range":
                "continuous; higher = more anomalous",
            "generated_in":
                "RQ4",
            "relevance":
                "Provides a continuous audit-priority score for ranking observations.",
        },

        {
            "variable":
                "anomaly_rank",
            "type":
                "integer",
            "description":
                "Rank of the mean anomaly score, with rank 1 representing the highest audit priority.",
            "unit_or_range":
                "positive integer",
            "generated_in":
                "RQ4",
            "relevance":
                "Provides an operational ordering of candidate observations.",
        },

        {
            "variable":
                "largest_deviation_feature",
            "type":
                "categorical",
            "description":
                "RQ4 feature with the largest absolute robust-scaled deviation for the observation.",
            "unit_or_range":
                "one of six model features",
            "generated_in":
                "RQ4",
            "relevance":
                "Provides transparent feature context without claiming causal model attribution.",
        },

        {
            "variable":
                "largest_abs_scaled_deviation",
            "type":
                "float",
            "description":
                "Absolute robust-scaled magnitude of the observation's largest-deviation feature.",
            "unit_or_range":
                ">= 0",
            "generated_in":
                "RQ4",
            "relevance":
                "Quantifies the strength of the reported feature-level contextual deviation.",
        },
    ]

    dictionary = pd.DataFrame(rows)

    dictionary.to_csv(
        OUTPUT_DIR
        / "analysis_data_dictionary.csv",
        index=False,
    )

    return dictionary



# ============================================================
# END-TO-END RUNNER
# ============================================================

def main():

    print(
        "\nQM640 DSA MODERATION OVERSIGHT"
        "\nSYNOPSIS ANALYSIS\n"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    weekly, summary = load_data()

    # --------------------------------------------------------
    # Data validation
    # --------------------------------------------------------

    print(
        "1. Validating deterministic data layer..."
    )

    validation = validate_data(
        weekly,
        summary,
    )

    print(validation)

    source_validation = (
        validate_source_manifest()
    )

    print(
        "Source manifest:",
        source_validation,
    )

    write_missingness_report(
        weekly,
        summary,
    )

    write_analysis_data_dictionary()

    # --------------------------------------------------------
    # RQ1
    # --------------------------------------------------------

    print("\n2. Running RQ1...")

    _, rq1_metrics = run_rq1(
        summary
    )

    print(rq1_metrics)

    # --------------------------------------------------------
    # RQ2
    # --------------------------------------------------------

    print("\n3. Running RQ2...")

    _, rq2_metrics = run_rq2(
        summary
    )

    print(rq2_metrics)

    # --------------------------------------------------------
    # RQ3
    # --------------------------------------------------------

    print("\n4. Running RQ3...")

    rq3_reliable, rq3_metrics = (
        run_rq3(
            weekly
        )
    )

    print(rq3_metrics)

    # --------------------------------------------------------
    # RQ4
    # --------------------------------------------------------

    print("\n5. Running RQ4...")

    (
        rq4_scored,
        rq4_stable,
        rq4_baselines,
        rq4_metrics,
    ) = run_rq4_primary(
        rq3_reliable
    )

    print(rq4_metrics)

    # --------------------------------------------------------
    # Sensitivity
    # --------------------------------------------------------

    print(
        "\n6. Running RQ4 validation "
        "and sensitivity..."
    )

    rq4_sensitivity = (
        run_rq4_sensitivity(
            rq3_reliable,
            rq4_scored,
        )
    )

    print(
        rq4_sensitivity[
            "metrics"
        ]
    )

    # --------------------------------------------------------
    # Benchmark audit
    # --------------------------------------------------------

    print(
        "\n7. Comparing reproducible "
        "results with validation benchmarks..."
    )

    check = benchmark_check(
        rq1_metrics,
        rq2_metrics,
        rq3_metrics,
        rq4_metrics,
    )

    print(
        check.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Run manifest
    # --------------------------------------------------------

    write_run_manifest()

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    summary_metrics = {
        "data_validation":
            validation,

        "source_validation":
            source_validation,

        "rq1":
            rq1_metrics,

        "rq2":
            rq2_metrics,

        "rq3":
            rq3_metrics,

        "rq4":
            rq4_metrics,

        "rq4_sensitivity":
            rq4_sensitivity[
                "metrics"
            ],
    }

    with open(
        OUTPUT_DIR
        / "summary_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary_metrics,
            file,
            indent=2,
        )

    print(
        "\nAnalysis complete."
    )

    print(
        "Outputs written to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
