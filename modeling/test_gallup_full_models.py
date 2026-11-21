"""Tests for gallup_full_models.py. Run from the modeling folder:
    python -m unittest test_gallup_full_models
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import gallup_full_models as g


def answers(*values):
    return pd.Series(values, dtype=object)


class TierRuleTests(unittest.TestCase):
    def test_tiers_are_cumulative_and_add_each_column_once(self):
        added = [c for _, num, cat in g.TIERS.values() for c in num + cat]
        self.assertEqual(len(added), len(set(added)))
        previous = set()
        for tier in g.TIERS:
            numeric, categorical = g.tier_columns(tier)
            current = set(numeric + categorical)
            self.assertTrue(previous < current, tier)
            previous = current

    def test_target_and_wellbeing_never_reach_usable_tiers(self):
        wellbeing = set(g.TIERS["F"][2])
        self.assertEqual(wellbeing, {"Q1", "Q2", "Q3", "Q42", "Q43", "Q59"})
        for tier in g.USABLE_TIERS:
            numeric, categorical = g.tier_columns(tier)
            used = set(numeric + categorical)
            self.assertFalse(used & set(g.NEVER_USED), tier)
            self.assertFalse(used & wellbeing, tier)
        numeric, categorical = g.tier_columns("F")
        self.assertFalse(set(numeric + categorical) & set(g.NEVER_USED))

    def test_factors_partition_the_tier_e_answers(self):
        numeric, categorical = g.tier_columns("E")
        columns = numeric + categorical
        items = [c for cols in g.FACTORS.values() for c in cols]
        self.assertEqual(len(items), len(set(items)), "an item is in two factors")
        self.assertTrue(set(items) <= set(columns), set(items) - set(columns))
        groups = g.factor_groups(columns)
        grouped = [c for cols in groups.values() for c in cols]
        self.assertCountEqual(grouped, columns)
        # The profile group holds only tier A and B answers.
        a_b = set(sum(g.tier_columns("B"), []))
        self.assertTrue(set(groups[g.PROFILE_GROUP]) <= a_b)

    def test_factor_groups_drop_factors_without_columns(self):
        groups = g.factor_groups(["Q33F", "Q51"])
        self.assertEqual(groups, {"Feeling supported (Support)": ["Q33F"], g.PROFILE_GROUP: ["Q51"]})


class AnswerCleaningTests(unittest.TestCase):
    def test_numeric_answers(self):
        result = g.numeric_answers(answers("5", " ", "-98", 3.0, "40.5", -98))
        np.testing.assert_allclose(result.to_numpy(dtype=float), [5, np.nan, np.nan, 3, 40.5, np.nan], equal_nan=True)

    def test_coded_answers_keep_special_codes_as_their_own_categories(self):
        result = g.coded_answers(answers("1", "6", "7", "-98", "-99", " ", 2.0, 1))
        self.assertEqual(result.dtype, "category")
        self.assertEqual(list(result.astype(str)), ["1", "6", "7", "-98", "-99", g.NOT_ASKED, "2", "1"])

    def test_likert_keeps_only_the_1_to_5_scale(self):
        result = g.likert(answers(1, 5, 6, 7, -98, " ", "3"))
        np.testing.assert_allclose(result.to_numpy(dtype=float), [1, 5, np.nan, np.nan, np.nan, np.nan, 3], equal_nan=True)


class JobLevelFeatureTests(unittest.TestCase):
    def test_cell_means_are_smoothed_and_unseen_cells_get_the_overall_mean(self):
        source = pd.DataFrame({"Q6": [1, 1, 2], "Q7": [5, 5, 5], "Q33F": [1, 3, 5]})
        target = pd.DataFrame({"Q6": [1, 9], "Q7": [5, 9]})
        result = g.cell_means(source, target, ["Q33F"])["job_avg_Q33F"].to_numpy()
        overall, k = 3.0, g.SMOOTHING
        np.testing.assert_allclose(result, [(1 + 3 + k * overall) / (2 + k), overall])

    def test_not_sure_answers_do_not_count(self):
        source = pd.DataFrame({"Q6": [1, 1], "Q7": [5, 5], "Q33F": [2, 6]})
        result = g.cell_means(source, source.iloc[:1], ["Q33F"])["job_avg_Q33F"].iloc[0]
        self.assertAlmostEqual(result, 2.0)

    def test_own_answer_never_feeds_own_feature(self):
        rng = np.random.default_rng(0)
        n = 200
        data = pd.DataFrame({"Q6": rng.integers(1, 4, n), "Q7": rng.integers(1, 3, n), "Q33F": rng.integers(1, 6, n)})
        train_idx, test_idx = np.arange(150), np.arange(150, n)
        before = g.job_level_features(data, train_idx, test_idx, ["Q33F"])
        changed = data.copy()
        changed.loc[7, "Q33F"] = 1 if data.loc[7, "Q33F"] != 1 else 5
        after = g.job_level_features(changed, train_idx, test_idx, ["Q33F"])
        self.assertEqual(before.loc[7, "job_avg_Q33F"], after.loc[7, "job_avg_Q33F"])
        # Test rows do use every training answer, including row 7's.
        self.assertFalse(np.allclose(before.iloc[test_idx], after.iloc[test_idx]))
        self.assertFalse(before.isna().any().any())

    def test_test_answers_never_feed_any_feature(self):
        rng = np.random.default_rng(1)
        data = pd.DataFrame({"Q6": rng.integers(1, 3, 100), "Q7": 1, "Q33F": rng.integers(1, 6, 100)})
        train_idx, test_idx = np.arange(80), np.arange(80, 100)
        changed = data.copy()
        changed.loc[test_idx, "Q33F"] = 1
        pd.testing.assert_frame_equal(
            g.job_level_features(data, train_idx, test_idx, ["Q33F"]),
            g.job_level_features(changed, train_idx, test_idx, ["Q33F"]),
        )


class EvaluationTests(unittest.TestCase):
    def test_best_f1_threshold(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.4, 0.35, 0.8])
        # 0.35 -> predictions [0, 1, 1, 1]: precision 2/3, recall 1, F1 0.8 (the best).
        self.assertAlmostEqual(g.best_f1_threshold(y, prob), 0.35)

    def test_evaluate_uses_threshold_from_training_predictions(self):
        y = np.array([0, 1, 0, 1, 1, 0, 1, 1])
        prob = np.linspace(0.1, 0.9, 8)
        draws = np.random.default_rng(0).integers(0, 8, size=(50, 8))
        draws = np.array([d for d in draws if 0 < y[d].mean() < 1])
        row = g.evaluate(y, prob, draws, np.ones(8), oof_y=np.array([0, 1]), oof_prob=np.array([0.2, 0.7]))
        self.assertAlmostEqual(row["threshold"], 0.7)
        self.assertLessEqual(row["roc_auc_ci_low"], row["roc_auc"])
        self.assertGreaterEqual(row["roc_auc_ci_high"], row["roc_auc"])
        self.assertAlmostEqual(row["roc_auc_survey_weighted"], row["roc_auc"])

    def test_permutation_importance_keeps_category_dtype(self):
        class UsesCategory:
            def predict_proba(self, X):
                assert X["signal"].dtype == "category"
                return (X["signal"].astype(str) == "1").astype(float).to_numpy() + 0.01 * X["noise"].to_numpy()

        rng = np.random.default_rng(0)
        signal = rng.integers(0, 2, 300)
        X = pd.DataFrame({"signal": pd.Series(signal.astype(str), dtype="category"), "noise": rng.random(300)})
        table = g.grouped_permutation_importance(UsesCategory(), X, signal, {"signal": ["signal"], "noise": ["noise"]})
        drops = table.set_index("factor")["auc_drop_mean"]
        self.assertGreater(drops["signal"], 0.3)
        self.assertLess(abs(drops["noise"]), 0.01)


class DataAndModelTests(unittest.TestCase):
    def write(self, folder, fit_ids, raw_ids):
        fit = pd.DataFrame({"ENTITY_ID": fit_ids, "satisfied": 1, **{c: 0.5 for c in g.ALL_FIT}})
        raw = pd.DataFrame({"ENTITY_ID": raw_ids, "Q4": 8, "Q33F": 1})
        fit.to_csv(folder / "fit.csv", index=False)
        raw.to_csv(folder / "raw.csv", index=False, encoding="cp1252")
        return folder / "fit.csv", folder / "raw.csv"

    def test_load_data_keeps_modelling_rows_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = g.load_data(*self.write(Path(tmp), [1, 2], [1, 2, 3]))
        self.assertEqual(list(data["ENTITY_ID"]), [1, 2])
        self.assertIn("Q33F", data.columns)

    def test_load_data_rejects_unmatched_or_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                g.load_data(*self.write(Path(tmp), [1, 4], [1, 2]))
            with self.assertRaises(pd.errors.MergeError):
                g.load_data(*self.write(Path(tmp), [1, 2], [1, 1, 2]))

    def test_models_learn_from_coded_answers(self):
        rng = np.random.default_rng(0)
        n = 600
        code = rng.integers(1, 6, n)
        hours = rng.normal(40, 5, n)
        y = (rng.random(n) < np.where(code <= 2, 0.85, 0.25)).astype(int)
        raw = pd.DataFrame({"Q33F": code.astype(str), "Q20": hours})
        raw.loc[:20, "Q20"] = -98
        X = g.prepare(raw, ["Q20"], ["Q33F"])
        self.assertTrue(X["Q20"].isna().sum() > 0)
        train, test = np.arange(450), np.arange(450, n)
        for make in (g.logistic, g.boosting):
            model = make(["Q20"], ["Q33F"]).fit(X.iloc[train], y[train])
            prob = model.predict_proba(X.iloc[test])
            self.assertTrue(((prob >= 0) & (prob <= 1)).all())
            self.assertGreater(roc_auc_score(y[test], prob), 0.7, make.__name__)


if __name__ == "__main__":
    unittest.main()
