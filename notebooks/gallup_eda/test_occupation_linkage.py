import pandas as pd
import unittest

from occupation_linkage import (
    link_respondents,
    normalize_occupation_code,
    prepare_crosswalk,
)


class OccupationLinkageTests(unittest.TestCase):
    def test_normalize_occupation_code(self):
        cases = [
            ("0012", None, "0012"),
            ("12.0", None, "12"),
            ("12", 4, "0012"),
            ("No response", None, None),
            (-98, None, None),
            (-98.0, None, None),
            ("12.5", None, None),
        ]
        for raw, width, expected in cases:
            with self.subTest(raw=raw, width=width):
                self.assertEqual(
                    normalize_occupation_code(raw, width=width), expected
                )

    def test_prepare_crosswalk_aggregates_one_to_many_mappings(self):
        crosswalk = pd.DataFrame(
            {
                "source": ["0012", "0012"],
                "onet": ["11-1011.00", "11-1012.00"],
                "score": [2.0, 4.0],
            }
        )

        prepared = prepare_crosswalk(
            crosswalk,
            source_code_col="source",
            onet_code_col="onet",
            numeric_attributes=["score"],
        )

        row = prepared.iloc[0]
        self.assertEqual(row["candidate_count"], 2)
        self.assertEqual(row["onet_code"], "11-1011.00|11-1012.00")
        self.assertEqual(row["score"], 3.0)

    def test_link_preserves_unmatched_rows_and_audit_fields(self):
        respondents = pd.DataFrame({"Q6_CODED": ["12.0", "999", None]})
        crosswalk = pd.DataFrame(
            {"source": ["12", "12"], "onet": ["11-1011.00", "11-1012.00"]}
        )

        linked = link_respondents(
            respondents,
            crosswalk,
            respondent_code_col="Q6_CODED",
            source_code_col="source",
            onet_code_col="onet",
        )

        self.assertEqual(len(linked), 3)
        self.assertEqual(
            linked["match_status"].tolist(),
            ["matched", "unmatched", "unmatched"],
        )
        self.assertEqual(linked["candidate_count"].tolist(), [2, 0, 0])
        self.assertEqual(linked["match_method"].tolist(), ["exact_code", "none", "none"])


if __name__ == "__main__":
    unittest.main()
