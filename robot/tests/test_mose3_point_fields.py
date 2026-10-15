"""Contracts at the MoSE3 dense-field to explicit-query boundary."""

from pathlib import Path
import sys
import unittest

import numpy as np

from robot.workflows.mose3.point_fields import (
    bilinear_taps, grid_to_source, mask_band, project, sample, sample_per_frame, source_to_grid, touches_mask,
)

VENDOR = Path(__file__).resolve().parents[2] / 'infrastructure/vendor/MoSE3'

try:
    import torch
except ImportError:  # the analysis environment has no Torch
    torch = None


class CoordinateTests(unittest.TestCase):
    def test_source_grid_round_trip_and_pixel_centres(self):
        xy = np.array([[0.0, 0.0], [1279.0, 703.0], [640.25, 351.5]])
        grid = source_to_grid(xy, (704, 1280), (280, 518))
        self.assertTrue(np.allclose(grid_to_source(grid, (704, 1280), (280, 518)), xy))
        # The centre of the image stays the centre of the grid.
        centre = source_to_grid(np.array([[639.5, 351.5]]), (704, 1280), (280, 518))
        self.assertTrue(np.allclose(centre, [[258.5, 139.5]]))

    def test_bilinear_is_exact_on_linear_fields_and_nearest_on_request(self):
        h, w = 5, 7
        yy, xx = np.indices((h, w), dtype=np.float64)
        field = np.stack([2 * xx + 3 * yy, xx - yy], axis=-1)[None]            # [1, H, W, 2]
        xy = np.array([[1.25, 2.5], [5.9, 0.1], [3.0, 4.0]])
        index, weight = bilinear_taps(xy, (h, w))
        values = sample(field, index, weight)[0]
        self.assertTrue(np.allclose(values[:, 0], 2 * xy[:, 0] + 3 * xy[:, 1]))
        self.assertTrue(np.allclose(values[:, 1], xy[:, 0] - xy[:, 1]))
        index, weight = bilinear_taps(xy, (h, w), nearest=np.array([True, False, False]))
        self.assertTrue(np.allclose(sample(field, index, weight)[0, 0], field[0, 2, 1]))

    def test_edge_flag_ignores_zero_weight_taps(self):
        edge = np.zeros((4, 4), bool)
        edge[2, 3] = True
        index, weight = bilinear_taps(np.array([[2.0, 1.0], [2.5, 1.5]]), (4, 4))
        self.assertTrue(np.array_equal(touches_mask(edge, index, weight), [False, True]))

    def test_projection_returns_frame_specific_pixels(self):
        k = np.array([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]])
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[1, :3, 3] = [0.5, 0, 0]
        points = np.array([[[0.0, 0.0, 2.0]], [[0.5, 0.0, 2.0]]])
        uv, depth = project(points, poses, k)
        self.assertTrue(np.allclose(uv, [[[50, 40]], [[50, 40]]]))
        self.assertTrue(np.allclose(depth, 2.0))

    def test_mask_band_membership_and_signed_distance(self):
        mask = np.zeros((40, 60), bool)
        mask[8:32, 10:50] = True                       # 2x the grid: rows 4..15, columns 5..24 on 20 x 30
        mask[8:32:2, 50] = True                        # a quarter-covered grid column stays outside
        index, fraction, signed = mask_band(mask, (20, 30), 2)
        rows, columns = np.divmod(index, 30)
        inside = signed > 0
        self.assertEqual(inside.sum(), 12 * 20)
        self.assertTrue(np.all(fraction[inside] >= 0.5) and np.all(fraction[~inside] < 0.5))
        self.assertTrue(np.all((rows[inside] >= 4) & (rows[inside] <= 15) & (columns[inside] >= 5) & (columns[inside] <= 24)))
        self.assertEqual(signed[(rows == 4) & (columns == 12)][0], 1)       # border ring
        self.assertEqual(signed[(rows == 7) & (columns == 12)][0], 4)
        self.assertEqual(signed[(rows == 2) & (columns == 12)][0], -2)      # band
        self.assertFalse(np.any((rows == 1) & (columns == 12)))             # beyond the band

    def test_per_frame_sampling(self):
        field = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4, 1)
        index, weight = bilinear_taps(np.array([[1.0, 2.0], [3.0, 0.0]]), (3, 4))
        values = sample_per_frame(field, index.reshape(2, 1, 4), weight.reshape(2, 1, 4))
        self.assertTrue(np.allclose(values[:, 0, 0], [field[0, 2, 1, 0], field[1, 0, 3, 0]]))


@unittest.skipIf(torch is None or not VENDOR.joinpath('mose3').is_dir(), 'Torch and vendored MoSE3 required')
class QuerySE3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(VENDOR))

    def test_integer_queries_match_upstream_per_pixel_fit(self):
        from mose3.utils.per_point_se3 import fit_per_pixel_se3
        from robot.workflows.mose3.point_fields import fit_query_se3

        # Two rigid parts with distinct embeddings; random data makes Horn fits
        # ill-conditioned, where upstream itself changes with its chunk size.
        generator = torch.Generator().manual_seed(3)
        count, h, w = 4, 9, 11
        yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing='ij')
        base = torch.stack([xx, yy, 4 + 0.2 * torch.sin(xx) + 0.1 * yy * yy], -1)
        right = (xx >= 5)[..., None]
        frames = [base]
        for t in range(1, count):
            angle = 0.1 * t
            rotation = torch.tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]],
                                    dtype=torch.float32)
            moved = torch.where(right, base @ rotation.T + torch.tensor([0.1 * t, 0, 0]), base + 0.05 * t)
            frames.append(moved + 1e-3 * torch.randn(h, w, 3, generator=generator))
        tracks = torch.stack(frames)
        part = torch.where(right, torch.ones(16), -torch.ones(16))
        emb = (part + 0.05 * torch.randn(count, h, w, 16, generator=generator))
        pixels = np.array([[2, 3], [8, 0], [5, 10]])                          # (h, w)
        for sigma in (0.0, 8.0):
            dense = fit_per_pixel_se3(tracks, emb, num_ref_pts=50, sigma_px=sigma, chunk_size=7)
            ours = fit_query_se3(tracks, emb, pixels[:, ::-1].astype(float),
                                 emb[0, pixels[:, 0], pixels[:, 1]], num_ref_pts=50, sigma_px=sigma)
            for name, key in (('quat', 'quat'), ('trans', 'trans'), ('ok', 'ok')):
                expected = dense[key][:, pixels[:, 0], pixels[:, 1]]
                actual = ours[name]
                if name == 'ok':
                    self.assertTrue(torch.equal(expected, actual))
                else:
                    self.assertTrue(torch.allclose(expected, actual, atol=1e-4), (sigma, name))

    def test_rigid_scene_gives_the_motion_back(self):
        from robot.workflows.mose3.point_fields import fit_query_se3, quat_to_matrix

        h, w = 6, 8
        yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing='ij')
        base = torch.stack([xx, yy, 3 + 0.1 * xx * yy], -1)
        angle = 0.3
        rotation = torch.tensor([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
                                dtype=torch.float32)
        shift = torch.tensor([0.2, -0.1, 0.4])
        tracks = torch.stack([base, base @ rotation.T + shift])
        emb = torch.ones(2, h, w, 16)
        result = fit_query_se3(tracks, emb, np.array([[3.5, 2.5]]), torch.ones(1, 16), num_ref_pts=48)
        self.assertTrue(np.allclose(quat_to_matrix(result['quat'][1, 0].numpy()), rotation.numpy(), atol=1e-4))
        self.assertTrue(np.allclose(result['trans'][1, 0].numpy(), shift.numpy(), atol=1e-4))


if __name__ == '__main__':
    unittest.main()
