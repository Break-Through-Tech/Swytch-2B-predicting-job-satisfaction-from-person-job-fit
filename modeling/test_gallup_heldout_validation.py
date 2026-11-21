import unittest

import numpy as np
import pandas as pd

import gallup_heldout_validation as heldout


class HeldOutValidationTests(unittest.TestCase):
    def make_data(self):
        rows = []
        for occupation in range(1, 11):
            for industry in range(1, 4):
                for respondent in range(20):
                    rows.append(
                        {
                            "Q6": occupation,
                            "Q7": industry,
                            "Q20": 35 + respondent,
                            "Q51": 30 + occupation,
                            "satisfied": (respondent + occupation + industry) % 2,
                        }
                    )
        data = pd.DataFrame(rows)
        for column in heldout.BACKGROUND_CATEGORICAL:
            data[column] = 1
        for column in heldout.FIT_FEATURES:
            data[column] = np.random.default_rng(42).random(len(data))
        return data

    def test_group_keys_cover_all_three_strategies(self):
        data = self.make_data()
        occupation = heldout.group_values(data, "occupation")
        industry = heldout.group_values(data, "industry")
        combined = heldout.group_values(data, "occupation_industry")
        self.assertEqual(len(np.unique(occupation)), 10)
        self.assertEqual(len(np.unique(industry)), 3)
        self.assertEqual(len(np.unique(combined)), 30)

    def test_training_and_test_groups_never_overlap(self):
        data = self.make_data()
        for strategy in ("occupation", "industry", "occupation_industry"):
            splits = heldout.split_groups(data, strategy)
            groups = heldout.group_values(data, strategy)
            self.assertGreater(len(splits), 0)
            for train_idx, test_idx in splits:
                self.assertTrue(set(groups[train_idx]).isdisjoint(set(groups[test_idx])))
                self.assertEqual(
                    set(data["satisfied"].to_numpy()[train_idx]),
                    {0, 1},
                )
                self.assertEqual(
                    set(data["satisfied"].to_numpy()[test_idx]),
                    {0, 1},
                )


if __name__ == "__main__":
    unittest.main()
