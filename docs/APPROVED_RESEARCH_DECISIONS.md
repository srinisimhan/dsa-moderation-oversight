# QM640 research evidence: approved scope and interpretation

**Project:** From Transparency to Triage: Explainable Audit Prioritisation for Automated Content Moderation Using EU DSA Transparency Data.

**Status:** Existing original results reproduced, source results reconciled and supplementary chronological holdout executed. This handoff consolidates outputs and makes *no change* to the frozen Interim model or its numerical findings. No external human-review assessment is part of the study.

## Research design (unchanged)

- Study period: 1 March–31 August 2026; seven selected platforms; 14 broad categories.
- Base data: 2,228 weekly platform-category observations, 86 full-period platform-category segments, representing 430,511,443 Statements of Reasons (SoRs).
- RQ1: Exposure and concentration of fully automated moderation. Report segment exposure and HHI with category sensitivity.
- RQ2: Same-category leave-one-platform-out automation-rate divergence. Report signed and absolute divergences, with interpretation by platform and category.
- RQ3: Robust adjacent-week automation-rate changes under fixed volume and series reliability requirements; 1,203 eligible comparisons from 50 series, 30 alerts in 18 series.
- RQ4: Six-feature Isolation Forest audit-priority ranking. 1,130 eligible records and 18 selections stable in >=8/10 seeds (17 in 10/10 and one in 9/10), compared with simple baselines and unsupervised challengers.

## Methodological consistency decisions

1. **RQ1–RQ3 randomisation tests are supplementary/exploratory.** The nulls condition on exchangeability within broad categories, which is not assured across platform practices. Report their statistic, p value and assumptions. They address *platform-level concentration/heterogeneity*, not whether individual cases are moderation errors.
2. **RQ2's original one-sample signed-mean t-test and the newer category-blocked maximum-platform-divergence permutation test have different null hypotheses.** The original p=0.05762 and supplementary permutation p=0.00024994 are not interchangeable or contradictory. Do not replace the primary descriptive RQ2 magnitude with either significance result.
3. **RQ4 is not a supervised classification task.** There are no verified error, illegality, non-compliance, or human audit-yield labels. Do not report precision/recall/F1/ROC-AUC, accuracy or an effect on actual regulatory decisions. RQ4 tests ranking repeatability, agreement and sensitivity, not decision correctness.
4. **Temporal holdout is retrospective, but training is chronological.** Earlier eligible weeks are used to fit scaler and models; later weeks are scored without refitting. Same-week peer features require that selected peers' weekly aggregates are available before batch scoring. The two test windows overlap; their metrics are sensitivity comparisons, not two independent test samples.
5. **Largest robust-scaled deviation is diagnostic context, not SHAP, feature attribution, or a causal explanation.** Removing the temporal feature changes selection, but temporal clipping at the examined train-percentile thresholds does not.
6. **Sample size is an eligible observational census of a bounded period**, not random independent sampling. No single textbook power calculation establishes adequacy across all four dependent analytical populations. Where sample-size formulas are requested, explain the illustrative assumptions and the limitations of dependence and cluster sizes.
7. **No new evaluation scope**: no human-review study, AI-assisted case assessment, ad-hoc random-selection statistical benchmark, or diversified-queue result is included in this approved package.

## Exploratory randomisation results (do not overinterpret)

| RQ | One-sided Monte Carlo p (4,000 permutations) | α=.05 result | Test target |
|---|---:|---|---|
| RQ1 | 0.555111 | Do not reject | HHI relative to category-permuted automation rates on observed exposure slots |
| RQ2 | 0.00024994 | Reject conditional null | Largest absolute platform-mean signed leave-one-out divergence |
| RQ3 | 0.145964 | Do not reject | Largest platform-mean series-level upper-tail change |

Neither non-rejection establishes equivalence nor rejection establishes moderation wrongdoing. These exploratory p-values are conditional on the documented assumptions. The report can retain the primary RQ descriptions and add these as clearly secondary analyses.

## RQ4 evidence and approved model choice

- Frozen full-period reference: 300-tree Isolation Forest, 10 fixed seeds, RobustScaler on six features, top 20 per seed and stable shortlist threshold >=8 seeds.
- Primary chronological holdout: 721 training records and 344 later records, beginning 6 July 2026; 19 holdout cases selected in >=8 seeds.
- Alternative *overlapping* sensitivity cutoff: 819/262, beginning 20 July 2026; 18 stable heldout cases.
- In primary holdout, the LOF and One-Class SVM top-20 selections overlap IF by 2 and 9 cases, respectively. Volume/peer/temporal baselines overlap by 1/0/12. Removing temporal feature retains 11/20, while clipping it at train-based 1st/99th percentiles retains 20/20.
- Isolation Forest remains the **preselected reference model** supported by reproducible stability and ranking outputs, not a scientifically proven accuracy winner. Challenger disagreement and sensitivity belong in limitations.

## Practical implementation and claims

A dashboard may present historical platform/category/time trends, RQ1–RQ3 data, RQ4 rank-and-filter queue, feature context and model comparison. It should say **retrospective analysis prototype** if no live operational deployment is demonstrated. The analyst may flag a weekly aggregate for investigation; the system cannot adjudicate a moderation decision or infer regulatory violation.

For full source provenance, see the repository README and frozen `outputs/synopsis/analysis_run_manifest.json`. This package does not contain raw European Commission monthly Parquet archives. No GitHub commit/push is implied by generation of this handoff.
