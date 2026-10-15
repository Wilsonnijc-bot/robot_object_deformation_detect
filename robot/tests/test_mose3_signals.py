"""The three link signals on a toy link, as in the plan's interactive demo."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from robot.scoring.mose3.rigidity import kabsch
from robot.scoring.mose3.signals import dense_residual, link_signals

FRAMES = 6


def toy(motion, embeddings=None):
    """A 4 x 0.6 x 0.3 link along x at depth 5; ``motion(t, x)`` moves the query points to frame t."""
    rng = np.random.default_rng(0)
    xq = np.column_stack([rng.uniform(0, 4, 400), rng.uniform(-0.3, 0.3, 400), rng.uniform(5, 5.3, 400)])
    trajectories = np.stack([motion(t, xq) for t in range(FRAMES)])
    n = len(xq)
    # MoSE3 keeps one body: every point gets the rigid motion of the whole link (here identity).
    quat = np.zeros((FRAMES, n, 4))
    quat[..., 0] = 1
    return dict(trajectories_world=trajectories.astype(np.float32), visible=np.ones((FRAMES, n), bool),
                in_frame=np.ones((FRAMES, n), bool), se3_valid=np.ones((FRAMES, n), bool),
                se3_quat_wxyz=quat.astype(np.float32), se3_trans=np.zeros((FRAMES, n, 3), np.float32),
                on_depth_edge=np.zeros(n, bool),
                rigidity_embedding_tracked=(np.ones((FRAMES, n, 16)) if embeddings is None else embeddings(xq)).astype(np.float32))


def toy_dense(motion, leak=0, visible_fraction=1.0):
    """Mask pixels of the toy link (3000) plus ``leak`` mask pixels on a static surface past its end."""
    rng = np.random.default_rng(1)
    link = np.column_stack([rng.uniform(0, 4, 3000), rng.uniform(-0.3, 0.3, 3000), rng.uniform(5, 5.3, 3000)])
    stray = np.column_stack([rng.uniform(4, 5, leak), rng.uniform(-0.3, 0.3, leak), rng.uniform(5, 5.3, leak)])
    tracks = np.concatenate([np.stack([motion(t, link) for t in range(FRAMES)]),
                             np.repeat(stray[None], FRAMES, axis=0)], axis=1)
    count = len(link) + leak
    logit = np.ones((FRAMES, count), np.float32)
    logit[1:, int(visible_fraction * count):] = -1          # later frames hide the rest
    return dict(tracks=tracks.astype(np.float32), visibility_logit=logit, on_depth_edge=np.zeros(count, bool),
                mask_signed_distance_px=np.full(count, 5, np.float16))


def rigid_motion(t, x):
    return Rotation.from_rotvec([0, 0.04 * t, 0.06 * t]).apply(x - [2, 0, 5]) + [2 + 0.1 * t, 0, 5]


def rotate_part(xq, select, rotvec, pivot):
    moved = xq.copy()
    moved[select] = Rotation.from_rotvec(rotvec).apply(xq[select] - pivot) + pivot
    return moved


class SignalTests(unittest.TestCase):
    def test_rigid_motion_is_quiet(self):
        def motion(t, x):
            return Rotation.from_rotvec([0, 0.04 * t, 0.06 * t]).apply(x - [2, 0, 5]) + [2 + 0.1 * t, 0, 5]
        result, _, _ = link_signals(toy(motion))
        self.assertLess(result['rigid_residual_refit']['max'], 1e-6)
        for part in ('stretch', 'shear', 'bend_deg', 'twist_deg'):
            self.assertLess(result[f'seam_{part}']['max'], 1e-4, part)
        self.assertLess(result['rigidity_discrepancy']['max'], 1e-9)
        # MoSE3's SE(3) here is identity, so its residual sees the whole rigid motion.
        self.assertGreater(result['rigid_residual']['max'], 0.05)

    def test_stretching_one_slice_opens_a_seam(self):
        def motion(t, x):
            moved = x.copy()
            inside, beyond = (x[:, 0] >= 2) & (x[:, 0] < 3), x[:, 0] >= 3
            moved[inside, 0] = 2 + (x[inside, 0] - 2) * (1 + 0.1 * t)
            moved[beyond, 0] += 0.1 * t
            return moved
        result, _, _ = link_signals(toy(motion))
        self.assertGreater(result['seam_stretch']['max'], 0.01)
        self.assertGreater(result['rigid_residual']['max'], 0.02)

    def test_bend_at_a_seam_is_a_bend_not_a_twist(self):
        angle = np.radians(12)

        def motion(t, x):
            return rotate_part(x, x[:, 0] >= 3, [0, angle * t / (FRAMES - 1), 0], [3, 0, 5.15])
        result, _, _ = link_signals(toy(motion))
        self.assertAlmostEqual(result['seam_bend_deg']['max'], 12, delta=4)
        self.assertLess(result['seam_twist_deg']['max'], 2)

    def test_twist_about_the_long_axis_is_a_twist(self):
        angle = np.radians(15)

        def motion(t, x):
            return rotate_part(x, x[:, 0] >= 3, [angle * t / (FRAMES - 1), 0, 0], [3, 0, 5.15])
        result, _, _ = link_signals(toy(motion))
        self.assertAlmostEqual(result['seam_twist_deg']['max'], 15, delta=5)
        self.assertLess(result['seam_bend_deg']['max'], 3)

    def test_split_embedding_raises_discrepancy(self):
        def embeddings(xq):
            e = np.zeros((FRAMES, len(xq), 16))
            e[:, :, 0] = 1
            e[3:, xq[:, 0] > 3, 0], e[3:, xq[:, 0] > 3, 1] = 0.6, 0.8      # one end leaves the body at frame 3
            return e
        result, frames, _ = link_signals(toy(lambda t, x: x, embeddings))
        self.assertLess(frames['rigidity_discrepancy'][1], 1e-9)
        self.assertGreater(result['rigidity_discrepancy']['max'], 0.1)
        self.assertGreaterEqual(result['rigidity_discrepancy']['max_frame'], 3)


class DenseFitTests(unittest.TestCase):
    def test_rigid_motion_is_quiet(self):
        result, frames, points = link_signals(toy(rigid_motion), toy_dense(rigid_motion))
        self.assertLess(result['rigid_residual_dense']['max'], 1e-5)
        self.assertLess(result['rigid_residual_dense_all']['max'], 1e-5)
        self.assertEqual(result['dense_frames_defined'], 1.0)
        self.assertAlmostEqual(result['link_length_dense'], 4 * 0.96, delta=0.05)
        self.assertEqual(points['rigid_residual_dense'].shape, (FRAMES, 400))

    def test_leaked_mask_pixels_lose_their_pull(self):
        data, dense = toy(rigid_motion), toy_dense(rigid_motion, leak=600)
        trajectories = data['trajectories_world'].astype(np.float64)
        visible = np.ones((FRAMES, trajectories.shape[1]), bool)
        robust = np.nanmax(dense_residual(trajectories, visible, dense)[0])
        # One unweighted fit of the same pixels is dragged toward the static surface.
        rotation, translation = kabsch(dense['tracks'][0].astype(np.float64), dense['tracks'][-1].astype(np.float64))
        plain = np.linalg.norm(trajectories[0] @ rotation.T + translation - trajectories[-1], axis=1).max()
        self.assertGreater(plain, 0.05)
        self.assertLess(robust * 4, 1e-3)

    def test_bent_end_stands_out_against_the_rest(self):
        angle = np.radians(10)

        def motion(t, x):
            return rotate_part(x, x[:, 0] >= 3, [0, angle * t / (FRAMES - 1), 0], [3, 0, 5.15])
        data = toy(motion)
        result, _, points = link_signals(data, toy_dense(motion))
        residual, x = points['rigid_residual_dense'][-1], data['trajectories_world'][0, :, 0]
        self.assertGreater(np.median(residual[x >= 3.5]), 0.01)
        self.assertLess(np.median(residual[x < 2.5]), 0.2 * np.median(residual[x >= 3.5]))
        self.assertGreater(result['rigid_residual_dense']['max'], 0.01)

    def test_frames_with_too_little_of_the_link_visible_are_undefined(self):
        result, frames, _ = link_signals(toy(rigid_motion), toy_dense(rigid_motion, visible_fraction=0.2))
        self.assertIsNone(result['rigid_residual_dense']['max'])
        self.assertEqual(result['dense_frames_defined'], 0.0)


if __name__ == '__main__':
    unittest.main()
