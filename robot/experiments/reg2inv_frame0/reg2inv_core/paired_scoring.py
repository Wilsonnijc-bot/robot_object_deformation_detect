"""Registration ablation: frozen official Reg2Inv features/support/scorer."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from reg2invbaselinecode.reg2inv import (
    LinkBank, InvalidCloud, preprocess_pcd, get_downsample_pcd,
    grid_subsample, apply_transform, KNN, index_points, camera_to_cad_transform,
)


def rigid(pose):
    pose = np.asarray(pose, np.float64)
    return (pose.shape == (4, 4) and np.isfinite(pose).all()
            and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
            and np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-3))


def normalized_pose(camera_to_cad, radius, query_center, cad_center):
    pose = np.asarray(camera_to_cad, np.float64).copy()
    if not rigid(pose):
        raise InvalidCloud('nonfinite_or_nonrigid_registration')
    pose[:3, 3] = pose[:3, 3] / radius - cad_center + pose[:3, :3] @ query_center
    return pose.astype(np.float32)


class PairedBank(LinkBank):
    @torch.inference_mode()
    def extract(self, points_m, retain_output=False):
        if len(points_m) < 128:
            raise InvalidCloud('insufficient_valid_points')
        raw = preprocess_pcd(points_m, self.radius_m).astype(np.float32)
        center = np.mean(points_m / self.radius_m - raw, axis=0)
        query = get_downsample_pcd(raw, self.voxel).astype(np.float32)
        if len(query) < 128:
            raise InvalidCloud('insufficient_voxel_points')
        stage = torch.from_numpy(query)
        lengths = torch.LongTensor([len(query)])
        for voxel in (2 * self.voxel, 4 * self.voxel):
            stage, lengths = grid_subsample(stage, lengths, voxel_size=voxel)
            if len(stage) < self.cfg.geotransformer.angle_k + 1:
                raise InvalidCloud('insufficient_coarse_points')
        output = self.forward(query)
        source = output['src_points_f']
        check = source.cpu().unsqueeze(0)
        _, group_idx = KNN(64)(check, check)
        neighborhood = index_points(check, group_idx).squeeze(0)
        centers = torch.mean(neighborhood, dim=1)
        distance = torch.norm(neighborhood - centers.unsqueeze(1), dim=2)
        _, top = torch.topk(distance, k=1, dim=1, largest=False)
        keep = torch.any(top == 0, dim=1)
        return dict(raw=raw, center=center, query=query, source=source,
                    shape=output['src_feats_i'].cpu().numpy(), keep=keep.numpy(),
                    native_transform=output['estimated_transform'].cpu().numpy(),
                    model_output=output if retain_output else None)

    @torch.inference_mode()
    def score(self, shared, camera_to_cad=None):
        transform = shared['native_transform'] if camera_to_cad is None else normalized_pose(
            camera_to_cad, self.radius_m, shared['center'], self.cad_center)
        if not rigid(transform):
            raise InvalidCloud('nonfinite_or_nonrigid_registration')
        metric = camera_to_cad_transform(transform, self.radius_m, shared['center'], self.cad_center)
        source = shared['source']
        aligned = apply_transform(source, torch.as_tensor(transform, device=source.device)).cpu().numpy()
        features = np.column_stack((aligned / self.position_norm, shared['shape'] / self.shape_norm))
        keep = shared['keep']
        features = np.asarray(features[keep], order='C', dtype=np.float32)
        distance = cKDTree(self.reference).query(
            shared['raw'] @ transform[:3, :3].T + transform[:3, 3])[0]
        fitness = float(np.mean(distance <= self.cfg.eval.acceptance_radius))
        result = dict(transform_camera_to_cad_m=metric, transform_normalized=transform,
                      registration_fitness=fitness,
                      registration_median_error_m=float(np.median(distance) * self.radius_m),
                      registration_status='ok' if fitness >= .5 else 'poor_alignment',
                      voxel_points=len(shared['query']), selected_feature_points=len(features))
        if len(features) < 64:
            result.update(score_status='insufficient_score_support', link_score=np.nan,
                          point_scores=np.full(len(shared['raw']), np.nan, np.float32))
        else:
            scores, point_scores = self.scorer.predict_score(
                [features], [source[torch.as_tensor(keep, device=source.device)].cpu()],
                [torch.from_numpy(shared['raw'])])
            result.update(score_status='ok', link_score=float(scores[0]),
                          point_scores=np.asarray(point_scores, np.float32))
            if not np.isfinite(result['point_scores']).all() or not np.isfinite(result['link_score']):
                raise InvalidCloud('nonfinite_anomaly_scores')
            assert len(result['point_scores']) == len(shared['raw'])
        return result
