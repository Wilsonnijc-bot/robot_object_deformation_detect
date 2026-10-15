"""Rigidity diagnostics respond to bending but not to rigid motion."""

import unittest

import numpy as np

from robot.scoring.mose3 import rigidity


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class RigidityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.points = rng.uniform(-1, 1, (40, 3)) * [3, 1, 0.5] + [0, 0, 5]
        self.count = 6

    def rigid(self):
        frames = [self.points @ rotation_z(0.05 * t).T + [0.1 * t, 0, 0] for t in range(self.count)]
        rotations = np.stack([np.repeat(rotation_z(0.05 * t)[None], len(self.points), 0) for t in range(self.count)])
        return np.stack(frames), rotations

    def bent(self):
        frames = []
        for t in range(self.count):
            moved = self.points.copy()
            far = moved[:, 0] > 0
            moved[far] = (moved[far] - [0, 0, 5]) @ rotation_z(0.15 * t).T + [0, 0, 5]
            frames.append(moved)
        rotations = np.stack([np.stack([rotation_z(0.15 * t if x > 0 else 0.0) for x in self.points[:, 0]])
                              for t in range(self.count)])
        return np.stack(frames), rotations

    def test_rigid_motion_is_near_zero_and_bending_is_not(self):
        visible = np.ones((self.count, len(self.points)), bool)
        embeddings = np.ones((self.count, len(self.points), 16))
        rigid = rigidity.summarize(*self.rigid()[:1], visible, self.rigid()[1], visible, embeddings)['scores']
        bent = rigidity.summarize(*self.bent()[:1], visible, self.bent()[1], visible, embeddings)['scores']
        # The PDI median/MAD formula is robust by design, so a half-link bend moves it least.
        for name, floor in (('rigid_fit_residual', 1e-2), ('pair_ratio_mad', 1e-3), ('se3_rotation_spread_deg', 1.0)):
            self.assertLess(rigid[name], 1e-6, name)
            self.assertGreater(bent[name], floor, name)
        self.assertLess(rigid['embedding_dispersion'], 1e-9)

    def test_pair_ratio_carries_previous_value_without_three_pairs(self):
        trajectories, _ = self.bent()
        visible = np.ones((self.count, len(self.points)), bool)
        visible[3, 2:] = False
        series, carried = rigidity.pair_ratio_mad(trajectories, visible)
        self.assertEqual(carried, [3])
        self.assertEqual(series[3], series[2])
        self.assertEqual(series[0], 0.0)

    def test_empty_subset_reports_no_scores(self):
        empty = rigidity.summarize(np.zeros((4, 0, 3)), np.zeros((4, 0), bool), np.zeros((4, 0, 3, 3)),
                                   np.zeros((4, 0), bool), np.zeros((4, 0, 16)))
        self.assertEqual(empty['points'], 0)
        for name in ('rigid_fit_residual', 'pair_ratio_mad', 'se3_rotation_spread_deg', 'embedding_dispersion'):
            self.assertIsNone(empty['scores'][name], name)

    def test_scale_change_does_not_raise_pair_ratio_mad(self):
        trajectories = np.stack([self.points * (1 + 0.1 * t) for t in range(self.count)])
        visible = np.ones(trajectories.shape[:2], bool)
        series, _ = rigidity.pair_ratio_mad(trajectories, visible)
        self.assertLess(series.max(), 1e-9)


if __name__ == '__main__':
    unittest.main()
