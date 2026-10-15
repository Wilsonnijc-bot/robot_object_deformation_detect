"""Minimal external adapter to the pinned official Real3D-AD inference path."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/Reg2Inv"))
sys.path.insert(0, str(ROOT / "third_party/KNN_CUDA"))

from config_real3dad import make_cfg
from model import create_model
from geotransformer.datasets.registration.real3dad.dataset import (
    norm_pcd, preprocess_pcd, get_downsample_pcd,
)
from geotransformer.utils.data import calibrate_neighbors_stack_mode, registration_collate_fn_stack_mode
from geotransformer.modules.ops import apply_transform, grid_subsample
from patchcore.patchcore import PatchCore
from patchcore.common import FaissNN
from patchcore.sampler import ApproximateGreedyCoresetSampler
from utils.cpu_knn import KNN
from utils.riconv_utils import index_points

from reg2invbaselinecode.geometry import camera_to_cad_transform


class InvalidCloud(ValueError):
    pass


class LinkBank:
    def __init__(self, link, cad_path, bank_dir, checkpoint):
        self.link = link
        self.cfg = make_cfg()
        self.voxel = self.cfg.data.voxel_size
        self.cad_m = np.load(cad_path)["points_m"]
        cad_norm, self.radius_m = norm_pcd(self.cad_m, self.cfg.data.num_points)
        self.cad_center = np.mean(self.cad_m / self.radius_m - cad_norm, axis=0)
        self.reference = get_downsample_pcd(cad_norm, self.voxel).astype(np.float32)
        sample = dict(ref_points=self.reference, src_points=self.reference,
                      transform=np.eye(4, dtype=np.float32), voxel_size=self.voxel)
        # Upstream calibrator, normal CAD only; labels and test geometry absent.
        self.neighbors, _ = calibrate_neighbors_stack_mode(
            [sample] * 25, registration_collate_fn_stack_mode, self.cfg.backbone.num_stages)
        self.model = create_model(self.cfg, self.voxel, self.neighbors).cuda().eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state['model'], strict=True)
        self.scorer = PatchCore(torch.device('cuda'))
        # Preserve actual official test.py behavior: the upstream keyword
        # anomaly_scorer_num_nn is swallowed by **kwargs, leaving 1-NN in load.
        # Use CPU FAISS IndexFlatL2, the official supported non-GPU backend.
        self.scorer.load(device=torch.device('cuda'), target_embed_dimension=1024,
                         featuresampler=ApproximateGreedyCoresetSampler(0.1, torch.device('cuda')),
                         nn_method=FaissNN(False, 4))
        bank_dir.mkdir(parents=True, exist_ok=True)
        path = bank_dir / f"{link}.npz"
        if path.exists():
            saved = np.load(path)
            assert np.allclose(saved['reference'], self.reference)
            self.position_norm = float(saved['position_norm'])
            self.shape_norm = float(saved['shape_norm'])
            self.scorer.anomaly_scorer.fit(detection_features=[saved['features']])
        else:
            # One normal full-surface CAD prototype, as allowed by upstream's
            # template loader. No query-derived points or learned robot tuning.
            output = self.forward(self.reference)
            position = apply_transform(output['src_points_f'], output['estimated_transform']).cpu().numpy()
            shape = output['src_feats_i'].cpu().numpy()
            self.position_norm = float(np.linalg.norm(position, axis=1).max())
            self.shape_norm = float(np.linalg.norm(shape, axis=1).max())
            if min(self.position_norm, self.shape_norm) <= 0:
                raise RuntimeError(f"Invalid CAD feature norms: {link}")
            features = np.column_stack((position / self.position_norm, shape / self.shape_norm)).astype(np.float32)
            # Never request more coreset entries than actual normal features.
            memory = self.scorer.build_memory_bank(features, min(10000, len(features)))
            np.savez_compressed(path, features=memory, position_norm=self.position_norm,
                                shape_norm=self.shape_norm, reference=self.reference,
                                radius_m=self.radius_m, cad_center=self.cad_center,
                                neighbors=self.neighbors,
                                normal_registration=output['estimated_transform'].cpu().numpy())
        # This is the exact surface point cloud fed into normal feature extraction.
        input_path = bank_dir / f"{link}_input.npz"
        if not input_path.exists():
            np.savez_compressed(input_path, normalized_points=self.reference,
                                points_m=(self.reference + self.cad_center) * self.radius_m,
                                radius_m=self.radius_m, cad_center=self.cad_center)

    @torch.inference_mode()
    def forward(self, query):
        return self.model(dict(ref_points=torch.from_numpy(self.reference).cuda(),
                               src_points=torch.from_numpy(query.astype(np.float32)).cuda(),
                               transform=torch.eye(4, device='cuda'), voxel_size=self.voxel))

    @torch.inference_mode()
    def predict(self, points_m):
        if len(points_m) < 128:
            raise InvalidCloud('insufficient_valid_points')
        raw = preprocess_pcd(points_m, self.radius_m).astype(np.float32)
        center = np.mean(points_m / self.radius_m - raw, axis=0)
        query = get_downsample_pcd(raw, self.voxel).astype(np.float32)
        if len(query) < 128:
            raise InvalidCloud('insufficient_voxel_points')
        # RIConv uses 32 neighbors only at the finest stage (already >=128).
        # The coarse angular embedding needs angle_k+1 nodes. Use the exact
        # upstream grid subsampler so this guard matches the model's counts.
        stage = torch.from_numpy(query)
        lengths = torch.LongTensor([len(query)])
        for voxel in (2 * self.voxel, 4 * self.voxel):
            stage, lengths = grid_subsample(stage, lengths, voxel_size=voxel)
            if len(stage) < self.cfg.geotransformer.angle_k + 1:
                raise InvalidCloud('insufficient_coarse_points')
        output = self.forward(query)
        transform = output['estimated_transform'].cpu().numpy()
        if not np.isfinite(transform).all() or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3):
            raise InvalidCloud('nonfinite_or_nonrigid_registration')
        source = output['src_points_f']
        aligned = apply_transform(source, output['estimated_transform']).cpu().numpy()
        shape = output['src_feats_i'].cpu().numpy()
        features = np.column_stack((aligned / self.position_norm, shape / self.shape_norm))

        # Verbatim extraction logic of upstream test_real3dad.py.
        check = source.cpu().unsqueeze(0)
        _, group_idx = KNN(64)(check, check)
        neighborhood = index_points(check, group_idx).squeeze(0)
        group_centers = torch.mean(neighborhood, dim=1)
        center_distance = torch.norm(neighborhood - group_centers.unsqueeze(1), dim=2)
        _, top = torch.topk(center_distance, k=1, dim=1, largest=False)
        keep = torch.any(top == 0, dim=1)
        xyz_sampled = source[keep].cpu()
        features = np.asarray(features[keep.numpy()], order='C', dtype=np.float32)

        t_m = camera_to_cad_transform(transform, self.radius_m, center, self.cad_center)
        aligned_m = points_m @ t_m[:3, :3].T + t_m[:3, 3]
        distance = cKDTree(self.reference).query(raw @ transform[:3, :3].T + transform[:3, 3])[0]
        fitness = float(np.mean(distance <= self.cfg.eval.acceptance_radius))
        median_m = float(np.median(distance) * self.radius_m)
        status = 'ok' if fitness >= 0.5 else 'poor_alignment'
        result = dict(points_m=points_m, aligned_points_m=aligned_m.astype(np.float32),
                      R=transform[:3, :3], t=transform[:3, 3], transform_normalized=transform,
                      transform_camera_to_cad_m=t_m, query_center_normalized=center,
                      radius_m=self.radius_m, voxel_points=len(query),
                      selected_feature_points=len(features), registration_status=status,
                      registration_fitness=fitness, registration_median_error_m=median_m,
                      score_status='ok')
        if len(features) < 64:
            # Upstream predict_score uses k=64 pooling. Do not silently replace
            # its scoring rule or densify observations to manufacture a score.
            result.update(score_status='insufficient_score_support', link_score=np.nan,
                          point_scores=np.full(len(points_m), np.nan, np.float32))
        else:
            scores, point_scores = self.scorer.predict_score([features], [xyz_sampled], [torch.from_numpy(raw)])
            result.update(link_score=float(scores[0]), point_scores=np.asarray(point_scores, np.float32))
            if not np.isfinite(result['point_scores']).all() or not np.isfinite(result['link_score']):
                raise InvalidCloud('nonfinite_anomaly_scores')
            assert len(result['point_scores']) == len(points_m)
        return result


def seed_all(seed=7351):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    o3d.utility.random.seed(seed)
    torch.set_num_threads(4)
