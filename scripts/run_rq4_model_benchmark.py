#!/usr/bin/env python3
"""Optional post-hoc RQ4 benchmark. NEVER rewrites the frozen primary RQ4 pipeline.

Fits LOF and One-Class SVM to the EXACT six-feature eligible RQ4 population;
compares their unusualness rankings against SAVED Isolation Forest scores.
No labels or supervised metrics are claimed. This script only writes to
outputs/interim/model_benchmarks/ by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
import sklearn
import scipy

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "outputs/synopsis/rq4_scored_observations.csv"
DEFAULT_DEST = ROOT / "outputs/interim/model_benchmarks"
KEYS = ["week_start", "platform_name", "category"]
FEATURES = [
    "log1p_represented_sors",
    "fully_automated_rate",
    "content_removed_rate",
    "account_or_service_restriction_rate",
    "weekly_peer_divergence_abs",
    "automation_change_abs",
]


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Deterministic ordering: larger means more anomalous; ties retain CSV order."""
    return np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:k]


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def read_frozen_source(path: Path, expected_n: int | None = 1130) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = KEYS + FEATURES + ["anomaly_score", "stable_anomaly"]
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"Frozen RQ4 CSV missing required columns: {missing}")
    if expected_n is not None and len(df) != expected_n:
        raise ValueError(f"Expected {expected_n} eligible RQ4 rows; found {len(df)}")
    if df[KEYS].isna().any().any() or df.duplicated(KEYS).any():
        raise ValueError("Frozen RQ4 observation keys are missing or duplicated")
    a = df[FEATURES + ["anomaly_score"]].to_numpy(dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("Missing or non-finite feature/Isolation Forest score")
    if (df["stable_anomaly"].astype(str).str.lower().isin(["true", "false"]) == False).any():
        raise ValueError("Unexpected stable_anomaly values")
    # Original frozen RQ4 shortlist: >=8/10 seeded top-20 selections.
    return df


def lof_scores(X: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    if n_neighbors < 2 or n_neighbors >= len(X):
        raise ValueError("LOF n_neighbors must be >=2 and less than sample size")
    model = LocalOutlierFactor(n_neighbors=n_neighbors, metric="minkowski", p=2,
                               contamination=0.05, novelty=False, n_jobs=-1)
    model.fit_predict(X)
    return -model.negative_outlier_factor_  # Higher = more unusual.


def svm_scores(X: np.ndarray, nu: float = 0.05) -> np.ndarray:
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    model.fit(X)
    return -model.decision_function(X).ravel()  # Higher = more unusual.


def evaluate(df: pd.DataFrame, top_k: int = 20) -> dict:
    if not 1 <= top_k < len(df):
        raise ValueError("top_k must be >=1 and less than population")
    X = RobustScaler().fit_transform(df[FEATURES].to_numpy(float))
    reference = df["anomaly_score"].to_numpy(dtype=float)
    scores = {
        "Isolation Forest (frozen)": reference,
        "Local Outlier Factor": lof_scores(X, 20),
        "One-Class SVM": svm_scores(X, 0.05),
    }
    reference_set = set(top_indices(reference, top_k).tolist())
    stable_set = set(np.flatnonzero(df["stable_anomaly"].astype(str).str.lower().eq("true")).tolist())
    if len(df) == 1130 and len(stable_set) != 18:
        raise ValueError(f"Frozen stable shortlist expected 18 rows; found {len(stable_set)}")
    comparisons, rankings, top_rows = [], [], []
    for model_name, values in scores.items():
        top = top_indices(values, top_k)
        top_set = set(top.tolist())
        rank = pd.Series(values).rank(ascending=False, method="average").to_numpy(dtype=float)
        rho = float(spearmanr(reference, values).statistic)
        comparisons.append({
            "model": model_name,
            "eligible_rows": len(df),
            "top_k": top_k,
            "overlap_with_if_top_k": len(top_set & reference_set),
            "jaccard_with_if_top_k": jaccard(top_set, reference_set),
            "spearman_with_if_all_ranks": rho,
            "overlap_with_if_stable_shortlist": len(top_set & stable_set),
            "jaccard_with_if_stable_shortlist": jaccard(top_set, stable_set),
        })
        rankings.append(pd.DataFrame({
            "model": model_name,
            **{key: df[key].values for key in KEYS},
            "anomaly_priority_score": values,
            "rank_full_population": rank,
            "in_model_top_k": [i in top_set for i in range(len(df))],
            "in_if_frozen_stable_shortlist": [i in stable_set for i in range(len(df))],
        }))
        for place, idx in enumerate(top.tolist(), 1):
            top_rows.append({"model": model_name, "rank": place,
                             **{key: df.at[idx, key] for key in KEYS},
                             "anomaly_priority_score": float(values[idx]),
                             "in_if_top_k": idx in reference_set,
                             "in_if_stable_shortlist": idx in stable_set})

    # Fixed, pre-declared parameter changes; agreement with each challenger's
    # own default top-k is a sensitivity diagnostic, NOT predictive accuracy.
    variants = [
        ("Local Outlier Factor", "n_neighbors=10", lof_scores(X, 10)),
        ("Local Outlier Factor", "n_neighbors=35", lof_scores(X, 35)),
        ("One-Class SVM", "nu=0.03", svm_scores(X, 0.03)),
        ("One-Class SVM", "nu=0.10", svm_scores(X, 0.10)),
    ]
    sensitivity = []
    for model_name, variation, alt_scores in variants:
        default_set = set(top_indices(scores[model_name], top_k).tolist())
        alt_set = set(top_indices(alt_scores, top_k).tolist())
        sensitivity.append({"model": model_name, "parameter_variant": variation,
                            "overlap_with_model_default_top_k": len(default_set & alt_set),
                            "jaccard_with_model_default_top_k": jaccard(default_set, alt_set)})
    return {
        "comparison": pd.DataFrame(comparisons),
        "rankings": pd.concat(rankings, ignore_index=True),
        "top_k": pd.DataFrame(top_rows),
        "sensitivity": pd.DataFrame(sensitivity),
        "metadata": {
            "source": str(DEFAULT_SOURCE.relative_to(ROOT)) if df.attrs.get("canonical_source", False) else "explicit-input",
            "eligible_rows": len(df), "features": FEATURES, "top_k": top_k,
            "if_primary": "Imported frozen 10-seed mean anomaly_score; no IF refit",
            "if_frozen_stable_shortlist_size": len(stable_set),
            "preprocessing": "RobustScaler fitted to same six features on same eligible population",
            "library_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                                 "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
            "LOF": "n_neighbors=20; p=2; contamination=.05; in-sample negative_outlier_factor_",
            "OneClassSVM": "RBF; gamma=scale; nu=.05; in-sample negative decision_function",
            "evaluation": "unlabelled ranking agreement and parameter sensitivity; no accuracy/F1",
            "caution": "Correlated scores/overlap neither prove correctness nor moderation errors",
        },
    }


def save_outputs(results: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("comparison", "rq4_challenger_model_comparison.csv"),
        ("top_k", "rq4_challenger_top20.csv"),
        ("sensitivity", "rq4_challenger_parameter_sensitivity.csv"),
        ("rankings", "rq4_challenger_all_rankings.csv"),
    ]:
        results[key].to_csv(destination / filename, index=False)
    (destination / "rq4_challenger_metadata.json").write_text(
        json.dumps(results["metadata"], indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--expected-n", type=int, default=1130,
                        help="Frozen RQ4 population guard; set to your test fixture size for tests")
    args = parser.parse_args()
    df = read_frozen_source(args.input, expected_n=args.expected_n)
    if args.input.resolve() == DEFAULT_SOURCE.resolve():
        df.attrs["canonical_source"] = True
    results = evaluate(df, top_k=args.top_k)
    save_outputs(results, args.output_dir)
    print("\nRQ4 challenger comparison (unsupervised; no accuracy/F1):")
    print(results["comparison"].round(4).to_string(index=False))
    print("\nSensitivity to predefined parameter alternatives:")
    print(results["sensitivity"].round(4).to_string(index=False))
    print(f"\nSaved five files to {args.output_dir}")


if __name__ == "__main__":
    main()
