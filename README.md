# DSA Moderation Oversight

QM640 Data Analytics Capstone project on explainable oversight and audit prioritisation for automated content moderation using European Commission Digital Services Act (DSA) Transparency Database data.

## Project objective

The project develops a reproducible framework for identifying which **platform × violation-category × week** observations warrant human audit attention when oversight capacity is limited.

The analysis does not determine whether a moderation decision is correct, whether one platform performs better than another, or whether observed differences imply policy non-compliance. The outputs are prioritisation signals for further human review.

## Data source

The analytical data are derived from the European Commission DSA Transparency Database monthly **aggregated-simple** Parquet files.

Official resources:

- DSA Transparency Database: https://transparency.dsa.ec.europa.eu/
- Data download documentation: https://dsa.pages.code.europa.eu/transparency-database/dsa-tdb/data_sources.html
- DSA data citation DOI: https://doi.org/10.2906/134353607485211

The repository does not redistribute the large raw monthly archives. The deterministic build script downloads them directly from the official aggregate-data endpoint and verifies the published SHA-1 checksums.

## Study scope

- Study period: **March 1 through August 31, 2026**
- Platforms: Instagram, TikTok, YouTube, X, Snapchat, LinkedIn, Roblox
- Category level: broad DSA violation category
- Primary weekly unit: week × platform × category
- Full-period unit: platform × category
- Total represented Statements of Reasons: **430,511,443**

## Research questions

**RQ1 - Automation exposure and concentration**
Measures fully automated moderation rates, exposure shares, cumulative concentration and HHI across platform-category segments.

**RQ2 - Peer divergence**
Compares each platform-category segment with the leave-one-platform-out median of other selected platforms reporting the same category.

**RQ3 - Temporal instability**
Identifies unusual week-over-week changes in fully automated moderation using exact seven-day adjacency, volume support and a robust median/MAD change score.

**RQ4 - Integrated audit prioritisation**
Uses an explainable, stability-tested Isolation Forest to combine volume, automation, enforcement, peer-divergence and temporal-change information into a human-audit priority queue.

## Repository structure

```text
dsa-moderation-oversight/
├── README.md
├── requirements.txt
├── scripts/
│   ├── build_dataset.py
│   └── run_synopsis_analysis.py
├── data/
│   ├── dsa_weekly_segments.csv
│   ├── dsa_segment_summary.csv
│   └── data_dictionary.csv
├── metadata/
│   └── source_manifest.csv
└── outputs/
    └── synopsis/
        ├── analysis_data_dictionary.csv
        ├── analysis_run_manifest.json
        ├── data_missingness_report.csv
        ├── feasibility_benchmark_check.csv
        ├── rq1_category_variation.csv
        ├── rq1_segment_exposure.csv
        ├── rq2_peer_divergence.csv
        ├── rq3_reliable_weekly.csv
        ├── rq3_temporal_alerts.csv
        ├── rq3_volume_sensitivity.csv
        ├── rq4_baseline_comparison.csv
        ├── rq4_candidate_threshold_sensitivity.csv
        ├── rq4_feature_set_sensitivity.csv
        ├── rq4_scored_observations.csv
        ├── rq4_seed_pairwise_jaccard.csv
        ├── rq4_stability_threshold_sensitivity.csv
        ├── rq4_stable_shortlist.csv
        ├── rq4_topk_sensitivity.csv
        └── summary_metrics.json
```

## Reproducibility architecture

The repository deliberately separates deterministic data preparation from research analysis.

### Layer 1 - deterministic data preparation

`scripts/build_dataset.py`:

- downloads the six official monthly aggregate archives;
- verifies official SHA-1 checksums;
- selects the seven study platforms and study period;
- aggregates represented Statements of Reasons;
- consolidates duplicate week-platform-category keys created at month boundaries;
- calculates published rates;
- flags complete Monday-Sunday study weeks;
- creates the full-period platform-category summary;
- writes source provenance and the base data dictionary.

It intentionally does **not** apply RQ-specific audit thresholds or machine-learning models.

### Layer 2 - research analysis

`scripts/run_synopsis_analysis.py` consumes the published processed datasets and implements RQ1-RQ4.

It performs:

- data-quality and reconciliation checks;
- RQ1 exposure and concentration analysis;
- RQ1 sensitivity excluding `STATEMENT_CATEGORY_OTHER_VIOLATION_TC`;
- RQ2 leave-one-platform-out peer-divergence analysis;
- RQ3 robust temporal-change detection;
- RQ3 500 / 1,000 / 5,000 SoR volume sensitivity;
- RQ4 six-feature Isolation Forest prioritisation;
- ten-seed stability testing;
- feature-context reporting;
- top-volume, top-peer and top-temporal baseline comparisons;
- Jaccard similarity;
- top-k sensitivity;
- stability-threshold sensitivity;
- feature-set sensitivity;
- candidate-threshold sensitivity;
- pairwise seed-stability analysis.

## Direct data access

The processed datasets are available directly in the repository:

- [Weekly platform × category dataset](data/dsa_weekly_segments.csv)
- [Six-month platform × category summary](data/dsa_segment_summary.csv)
- [Base data dictionary](data/data_dictionary.csv)
- [Source provenance manifest](metadata/source_manifest.csv)

The base dictionary documents variables created by the deterministic data layer.

`outputs/synopsis/analysis_data_dictionary.csv` separately documents variables derived during RQ1-RQ4 analysis.

## Rebuild the datasets

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
python scripts/build_dataset.py
```

The build recreates:

```text
data/dsa_weekly_segments.csv
data/dsa_segment_summary.csv
data/data_dictionary.csv
metadata/source_manifest.csv
```

## Run the complete analysis

After the processed datasets are available:

```bash
python scripts/run_synopsis_analysis.py
```

The analysis outputs are written to:

```text
outputs/synopsis/
```

For a complete reproduction beginning with the European Commission source files:

```bash
python -m pip install -r requirements.txt
python scripts/build_dataset.py
python scripts/run_synopsis_analysis.py
```

## Data-quality and reproducibility checks

Before RQ analysis, the pipeline verifies:

- required schema;
- uniqueness of week-platform-category and platform-category keys;
- platform coverage;
- complete-week coverage;
- required-field missingness;
- rate bounds;
- reconciliation of weekly and full-period represented counts;
- source-manifest coverage;
- official versus downloaded SHA-1 checksums.

The validated dataset contains:

- **2,228** weekly platform-category records;
- **86** full-period platform-category segments;
- **7** platforms;
- **14** broad violation categories;
- **26** complete Monday-Sunday study weeks;
- **430,511,443** represented Statements of Reasons.

## Primary RQ4 configuration

The primary audit-prioritisation model uses six features:

1. log1p represented SoR volume;
2. fully automated rate;
3. content-removal rate;
4. account/service-restriction rate;
5. absolute same-week peer divergence;
6. absolute automation-rate change.

Features are median-centred and IQR-scaled using `RobustScaler`.

Isolation Forest uses:

- 300 trees;
- 5% contamination;
- fixed top-20 audit queue;
- 10 random seeds: 11, 22, 33, 44, 55, 66, 77, 88, 99, 111;
- stable selection threshold of at least 8 of 10 runs.

Higher reported `anomaly_score` values indicate greater audit priority.

Feature-context reporting identifies the largest absolute robust-scaled feature deviation. It is descriptive and is not interpreted as causal attribution.

## Validation approach

Because RQ4 is unsupervised and no authoritative anomaly labels exist, evaluation does not use classification accuracy, F1, ROC-AUC or similar supervised metrics.

Validation instead emphasizes:

- seed stability;
- continuous anomaly ranking;
- feature context;
- baseline overlap;
- Jaccard similarity;
- top-k sensitivity;
- feature-set sensitivity;
- stability-rule sensitivity;
- candidate-threshold sensitivity.

The operational shortlist is intended to help prioritize human review, not to classify moderation behavior as correct or incorrect.

## Source citation

European Commission-DG CONNECT. (2023). *Digital Services Act Transparency Database*. Directorate-General for Communications Networks, Content and Technology. https://doi.org/10.2906/134353607485211
