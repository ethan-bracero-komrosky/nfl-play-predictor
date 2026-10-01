import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.proe import build_proe_features


def play(date, game, team="A", label=1, down=1, distance=2, two_point=0, season=2021):
    return dict(game_date=date, game_id=game, posteam=team, play_type=label,
                down=down, ydstogo=distance, two_point_attempt=two_point, season=season)


class ProeTests(unittest.TestCase):
    def build(self, rows, start=2021, end=2025):
        return build_proe_features(pd.DataFrame(rows), history_start_year=start,
                                   training_end_year=end)

    def test_known_values_and_current_game_exclusion(self):
        rows = [play("2021-01-01", "g1", label=1),
                play("2021-01-02", "g2", label=0),
                play("2021-01-03", "g3", label=1)]
        result = self.build(rows)
        np.testing.assert_allclose(result.league_pass_rate, [0.5, 1, 0.5])
        np.testing.assert_allclose(result.proe, [0, 0.5, -0.25])
        rows[-1]["play_type"] = 0
        assert_frame_equal(result, self.build(rows))

    def test_actual_and_expected_use_the_same_historical_situations(self):
        rows = [play("2021-01-01", "g1", "B", 1, 1, 2),
                play("2021-01-01", "g1", "B", 0, 3, 10),
                play("2021-01-02", "g2", "A", 1, 1, 2),
                play("2021-01-03", "g3", "A", 0, 3, 10),
                play("2021-01-04", "g4", "A", 1, 1, 2)]
        result = self.build(rows)
        self.assertEqual(result.iloc[-1].rolling_pass_rate, 0.5)
        self.assertEqual(result.iloc[-1].rolling_expected_pass_rate, 0.5)
        self.assertEqual(result.iloc[-1].proe, 0)

    def test_window_is_previous_five_games_and_carries_across_seasons(self):
        rows = [play(f"2021-01-{i+1:02d}", f"g{i}", label=i % 2 == 0)
                for i in range(7)]
        rows[-1].update(game_date="2022-01-01", season=2022)
        result = self.build(rows)
        self.assertAlmostEqual(result.iloc[-1].proe,
                               2/5 - (1 + 1/2 + 2/3 + 1/2 + 3/5)/5)

    def test_same_date_games_do_not_inform_each_other(self):
        rows = [play("2021-01-01", "g1", "A", 1),
                play("2021-01-01", "g2", "B", 0)]
        result = self.build(rows)
        np.testing.assert_allclose(result.league_pass_rate, [0.5, 0.5])
        rows[0]["play_type"] = 0
        assert_frame_equal(result, self.build(rows))

    def test_all_test_outcomes_are_excluded(self):
        rows = [play("2025-12-01", "g1", label=1, season=2025),
                play("2026-09-01", "g2", label=1, season=2026),
                play("2026-09-08", "g3", label=0, season=2026)]
        result = self.build(rows)
        rows[1]["play_type"] = 0
        rows[2]["play_type"] = None  # inference does not need test labels
        assert_frame_equal(result, self.build(rows))
        self.assertEqual(result.iloc[1].proe, result.iloc[2].proe)

    def test_future_training_labels_do_not_change_earlier_features(self):
        rows = [play(f"2021-01-{i+1:02d}", f"g{i}", label=i % 2) for i in range(4)]
        result = self.build(rows)
        rows[2]["play_type"] = 1 - rows[2]["play_type"]
        rows[3]["play_type"] = 1 - rows[3]["play_type"]
        assert_frame_equal(result.iloc[:3], self.build(rows).iloc[:3])

    def test_history_start_excludes_older_seasons(self):
        rows = [play("2016-01-01", "old", label=1, season=2016),
                play("2021-01-01", "g1", label=0),
                play("2026-09-01", "test", label=1, season=2026)]
        result = self.build(rows)
        self.assertEqual(result.iloc[1].league_pass_rate, 0.5)
        self.assertEqual(result.iloc[2].league_pass_rate, 0)
        self.assertEqual(result.iloc[2].proe, -0.5)

    def test_conversions_have_a_separate_expected_rate(self):
        rows = [play("2021-01-01", "g1", label=1),
                play("2021-01-01", "g1", label=0, down=None, distance=0, two_point=1),
                play("2021-01-02", "g2", label=1),
                play("2021-01-02", "g2", label=1, down=None, distance=0, two_point=1)]
        result = self.build(rows)
        np.testing.assert_allclose(result.league_pass_rate, [0.5, 0.5, 1, 0])
        self.assertFalse(result.isna().any().any())

    def test_shuffled_rows_keep_alignment(self):
        df = pd.DataFrame([play(f"2021-01-{i+1:02d}", f"g{i}", label=i % 2)
                           for i in range(5)], index=[11, 7, 31, 13, 5])
        expected = build_proe_features(df, history_start_year=2021, training_end_year=2025)
        shuffled = build_proe_features(df.sample(frac=1, random_state=42),
                                      history_start_year=2021, training_end_year=2025)
        assert_frame_equal(expected.sort_index(), shuffled.sort_index())

    def test_invalid_distances_and_missing_metadata_fail(self):
        for distance in [-1, 101, None, 0]:
            with self.assertRaises(ValueError):
                self.build([play("2021-01-01", "g1", distance=distance)])
        with self.assertRaises(ValueError):
            self.build([play(None, "g1")])


if __name__ == "__main__":
    unittest.main()
