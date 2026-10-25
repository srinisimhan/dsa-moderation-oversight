# RQ4 model extension — Local Outlier Factor and One-Class SVM

**Status:** Optional *post-hoc* comparison for the Interim/Final Report. **Not a change** to the frozen RQ4 primary analysis.

## Purpose and boundaries

Use the exact 1,130 eligible observations and six features in the existing `outputs/synopsis/rq4_scored_observations.csv`. Compare the *saved* 10-seed mean Isolation Forest score against two challengers:

- **Local Outlier Factor (LOF)**: 20 neighbors; Euclidean distances; score = negative of `negative_outlier_factor_` (larger = more unusual).
- **One-Class SVM**: RBF kernel; `gamma='scale'`, `nu=0.05`; score = negative `decision_function` (larger = more unusual).

Each challenger uses the **same six frozen features** and `RobustScaler` as the original model. Do **not** use the saved IF score, IF rank, shortlist flag, peer baseline labels, or other output fields as predictor features. All analyses are exploratory and in-sample ranking comparisons, not a ground-truth test set.

## Execution (from repository root)

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_rq4_model_benchmark.py' -v
python scripts/run_rq4_model_benchmark.py
```

The existing base files and the frozen script stay unchanged. Only `outputs/interim/model_benchmarks/` receives new outputs:

1. `rq4_challenger_model_comparison.csv` — equal-sized top-20 overlap/Jaccard vs IF, full-population Spearman agreement, and separate overlap with original 18 stable IF selections.
2. `rq4_challenger_top20.csv` — each model's top-20 cases, with keys and cross-reference flags.
3. `rq4_challenger_parameter_sensitivity.csv` — LOF neighbors 10/35 and SVM nu 0.03/0.10 compared with their respective defaults.
4. `rq4_challenger_all_rankings.csv` — optional reproducibility evidence, one record per observation per model.
5. `rq4_challenger_metadata.json` — declared features, parameter settings and evaluation interpretation.

## Methodological safeguards

- The **same 1,130-row population**, observation keys and transformations are required. The script stops on missing/non-finite features, duplicates or an unexpected population size.
- Each **top-20 comparison uses 20 vs 20**; the frozen 18-item stable IF shortlist is assessed in separate columns. These denominators must not be confused.
- LOF/One-Class SVM scores are only **within-model rankings**; raw magnitudes across algorithms are not comparable.
- Spearman rank correlation and Jaccard similarity measure **agreement**, not correctness, detection accuracy or a statistically validated gold standard. Challengers need **not** agree with IF; disagreements are useful audit signals.
- LOF is deterministic in this setup. LOF neighbor sensitivity and SVM nu sensitivity are **parameter robustness**, not ten-seed stability or external generalization.
- **Do not calculate accuracy, F1, recall, precision, ROC-AUC**, or claim moderation errors identified: the DSA aggregate data has no independently verified audit-relevance labels.
- The existing IF model and its original seeded stability and sensitivity findings remain the primary RQ4 results.
- Record exact library versions and generated output values when reporting the extension. Never claim the synthetic tests below are DSA findings.

## Suggested report sentence after real results have been computed

> "As a supplementary, post-hoc robustness check, we compared the frozen Isolation Forest audit-priority ranking against Local Outlier Factor and One-Class SVM using the same 1,130 eligible observations and six features. We assessed equal-sized top-20 Jaccard agreement, full-population Spearman rank correlation, and sensitivity to predeclared parameter changes. These metrics describe cross-method ranking consistency rather than predictive accuracy, because no verified ground-truth labels are available."

Insert numeric results **only from the newly generated CSVs**. Label this as an extension rather than a preregistered study design.
