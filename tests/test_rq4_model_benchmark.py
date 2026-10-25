"""Synthetic smoke/contract tests only; not scientific results from DSA records."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_rq4_model_benchmark import FEATURES, evaluate, jaccard, read_frozen_source, save_outputs, top_indices


class BenchmarkContractTests(unittest.TestCase):
    @staticmethod
    def fixture(n=80):
        rng = np.random.default_rng(19)
        df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
        df["week_start"] = pd.date_range("2026-03-02", periods=n, freq="D").astype(str)
        df["platform_name"] = "SyntheticPlatform"
        df["category"] = "SYNTHETIC"
        df["anomaly_score"] = (df[FEATURES].abs().sum(axis=1) +
                               rng.normal(0, .05, size=n))
        df["stable_anomaly"] = False
        df.loc[[2, 10, 50], "stable_anomaly"] = True
        return df

    def test_rank_and_jaccard(self):
        self.assertEqual(top_indices(np.array([1., 9., 9., 2.]), 2).tolist(), [1, 2])
        self.assertEqual(jaccard({1, 2}, {2, 3}), 1/3)

    def test_models_and_saved_outputs(self):
        fixture = self.fixture()
        with tempfile.TemporaryDirectory() as path:
            path = Path(path)
            fixture.to_csv(path/"input.csv", index=False)
            df = read_frozen_source(path/"input.csv", expected_n=80)
            res = evaluate(df, top_k=10)
            self.assertEqual(len(res["comparison"]), 3)
            self.assertEqual(len(res["top_k"]), 30)
            self.assertEqual(len(res["sensitivity"]), 4)
            self.assertEqual(len(res["rankings"]), 240)
            self.assertEqual(res["comparison"].iloc[0]["overlap_with_if_top_k"], 10)
            self.assertEqual(res["comparison"].iloc[0]["jaccard_with_if_top_k"], 1.)
            self.assertTrue(np.isfinite(res["comparison"]["spearman_with_if_all_ranks"]).all())
            self.assertFalse((res["comparison"].columns.str.contains("accuracy|f1")).any())
            save_outputs(res, path/"outputs")
            self.assertEqual(len(list((path/"outputs").iterdir())), 5)

    def test_reject_changed_population(self):
        with tempfile.TemporaryDirectory() as path:
            p = Path(path)/"input.csv"
            self.fixture().to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, "Expected 1130"):
                read_frozen_source(p)

    def test_reject_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as path:
            p = Path(path)/"input.csv"
            df = self.fixture()
            df.loc[1, ["week_start", "platform_name", "category"]] = df.loc[0, ["week_start", "platform_name", "category"]]
            df.to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, "duplicated"):
                read_frozen_source(p, expected_n=80)


if __name__ == "__main__":
    unittest.main()
