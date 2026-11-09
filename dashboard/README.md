# DSA Oversight Radar — Streamlit Community Cloud

This app lives inside the **existing** repository `srinisimhan/dsa-moderation-oversight`.
It uses saved CSV/JSON outputs only. No model fitting occurs inside the dashboard.

## Repository integration

The integration ZIP contains **new files only** relative to the pre-dashboard `main` branch.
Open the existing GitHub repo. Choose **Add file → Upload files**, then drag the **contents of the extracted integration folder** into the uploader (preserving nested paths), and commit to a feature branch or `main` after review. Do **not** create another repository or delete existing folders.

Required entrypoint: `dashboard/app.py`. Its standalone dependency file is `dashboard/requirements.txt`; the original root `requirements.txt` is kept for research. Source paths are resolved from the repository root with `Path(__file__).resolve().parent.parent`.

## Deploy (no local commands)

1. Open https://share.streamlit.io/ and sign in using GitHub.
2. Click **Create app → Yup, I have an app**.
3. Repository: `srinisimhan/dsa-moderation-oversight`; branch: the branch containing all integration files (prefer `main` after review); main file path: `dashboard/app.py`.
4. Deploy on Streamlit Community Cloud, choose a memorable free subdomain if available.
5. Verify all four sections, original 18 stable cases, primary holdout n=344, second holdout n=262 and CSV exports.

## Data scope

Frozen European Commission DSA Transparency Database aggregations, 1 March–31 August 2026. Historical research prioritisation only; neither an adjudication tool nor a moderation-error classifier. Original in-sample IF and chronological holdout views remain separate. Models are *not* retrained in the app.

## Automated tests

`python dashboard/test_dashboard_data.py` verifies exact source counts, source integrity and holdout metrics. No Streamlit import is needed for this test. Existing `scripts/` and `tests/` are unchanged or extended by new files only.
