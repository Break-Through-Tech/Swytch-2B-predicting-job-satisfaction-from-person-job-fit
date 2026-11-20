import unittest

import numpy as np
import pandas as pd

from engineer_fit_features import (
    advancement_fit,
    advancement_onet_fit,
    hours_fit,
    merge_questions,
    remote_fit,
    voice_fit,
)


def series(*values):
    return pd.Series(values, dtype=float)


class FitFeatureTests(unittest.TestCase):
    def assertSeries(self, actual, expected):
        np.testing.assert_allclose(actual.to_numpy(), np.array(expected, dtype=float), equal_nan=True)

    def test_remote_fit(self):
        q18 = series(2, 0, 5, 3, np.nan, 2)
        q19 = series(2, 7, 1, 8, 2, -98)
        # match, full mismatch, 4 days apart, "not possible", missing actual, missing wanted
        self.assertSeries(remote_fit(q18, q19), [1, 0, 1 - 4 / 7, np.nan, np.nan, np.nan])

    def test_advancement_fit(self):
        q33h = series(1, 5, 2, 4, 3, 6, 7, 1, np.nan)
        q34 = series(1, 2, 2, 1, 1, 1, 2, 3, 1)
        self.assertSeries(advancement_fit(q33h, q34), [1, 1, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan])

    def test_voice_fit(self):
        have = series(1, 4, 4, 2, 5, np.nan)
        should = series(1, 1, 3, 3, 1, 1)
        self.assertSeries(voice_fit(have, should), [1, 0, 2 / 3, 2 / 3, np.nan, np.nan])

    def test_hours_fit(self):
        self.assertSeries(hours_fit(series(2, 1, 3, np.nan, -98)), [1, 0, 0, np.nan, np.nan])

    def test_advancement_onet_fit(self):
        q33h = series(1, 5, 3, 6)
        recognition = series(7, 7, 4, 4)
        # strongly interested + top recognition, not interested + top, neutral + middle, not sure
        self.assertSeries(advancement_onet_fit(q33h, recognition), [1, 0, 1, np.nan])

    def test_merge_keeps_rows_and_rejects_duplicates(self):
        model = pd.DataFrame({"ENTITY_ID": [1, 2, 3], "satisfied": [1, 0, 1]})
        raw = pd.DataFrame({"ENTITY_ID": [3, 1, 2, 9], "Q18": [0, 1, -98, 2], "Q19": [1, 1, 8, 0], "Q33H": [1, 2, 3, 4], "Q34": [1, 2, 3, 1]})
        merged = merge_questions(model, raw)
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged["ENTITY_ID"].tolist(), [1, 2, 3])
        self.assertSeries(merged["Q18"], [1, np.nan, 0])  # -98 becomes NaN
        with self.assertRaises(ValueError):
            merge_questions(model, pd.concat([raw, raw.iloc[[0]]]))


if __name__ == "__main__":
    unittest.main()
