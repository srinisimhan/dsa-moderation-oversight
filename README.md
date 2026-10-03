# DSA Moderation Oversight

QM640 capstone project on explainable oversight and audit prioritisation for automated content moderation using European Commission Digital Services Act (DSA) Transparency Database data.

## Data source
The analytical data are derived from the European Commission DSA Transparency Database monthly **aggregated-simple** Parquet files.

Official resources:
- DSA Transparency Database: https://transparency.dsa.ec.europa.eu/
- Data download documentation: https://dsa.pages.code.europa.eu/transparency-database/dsa-tdb/data_sources.html
- DSA data citation DOI: https://doi.org/10.2906/134353607485211

The repository does not redistribute the large raw monthly archives. The build script downloads them directly from the official aggregate-data endpoint and verifies the official SHA-1 checksum when available.

## Study scope
- Study period: **2026-03-01 through 2026-08-31**
- Platforms: Instagram, TikTok, YouTube, X, Snapchat, LinkedIn, Roblox
- Category level: broad DSA violation category

## Repository structure
```text
dsa-moderation-oversight/
├── README.md
├── requirements.txt
├── scripts/
│   └── build_dataset.py
├── data/
│   ├── dsa_weekly_segments.csv
│   ├── dsa_segment_summary.csv
│   └── data_dictionary.csv
└── metadata/
    └── source_manifest.csv
```


## Direct data access

The processed analytical datasets used in this project are available directly in this repository:

- [Weekly platform × category dataset](data/dsa_weekly_segments.csv)
- [Six-month platform × category summary](data/dsa_segment_summary.csv)
- [Data dictionary](data/data_dictionary.csv)
- [Source provenance manifest](metadata/source_manifest.csv)

The same analytical datasets can be independently reproduced from the official European Commission source files using the build instructions below.

## Rebuild the dataset
Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
python scripts/build_dataset.py
```

## Generated datasets
`data/dsa_weekly_segments.csv` contains one row per week × platform × category. Edge partial weeks are retained and flagged with `complete_study_week`.

`data/dsa_segment_summary.csv` contains one row per platform × category for the full March 1-August 31, 2026 study period.

`data/data_dictionary.csv` defines published analytical variables.

`metadata/source_manifest.csv` records source URLs, checksum information, counts, and build timestamp.

## Reproducibility boundary
The build script performs source retrieval and deterministic preparation only. It intentionally does not apply RQ-specific minimum-volume thresholds, temporal anomaly thresholds, peer-divergence thresholds, robust z-scores, or ML models.

## Source citation
European Commission-DG CONNECT. (2023). *Digital Services Act Transparency Database*. Directorate-General for Communications Networks, Content and Technology. https://doi.org/10.2906/134353607485211
