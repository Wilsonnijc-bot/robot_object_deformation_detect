"""Pixel-aligned metric backprojection; no query-to-CAD scale fitting."""
import cv2
import numpy as np


def partial_cloud(depth, mask, intrinsics, confidence=None):
    """Return points in camera metres, matching depth-grid xy and quality metadata.

    The caller supplies a *resize-only* depth grid, with K for that grid. Erode
    one depth pixel and remove sharp local depth jumps before backprojection.
    No global depth quantile clipping: that could remove an actual bulge.
    """
    depth = np.asarray(depth, np.float32)
    h, w = depth.shape
    k = np.asarray(intrinsics, np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("Invalid depth-grid intrinsics")
    if not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-6:
        raise ValueError("Expected standard pinhole intrinsics")
    mask = cv2.resize(np.asarray(mask, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    border_fraction = float(np.count_nonzero(mask[[0, -1], :]) +
                            np.count_nonzero(mask[:, [0, -1]])) / max(int(mask.sum()), 1)
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    valid = eroded & np.isfinite(depth) & (depth > 0)
    clean = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
    local_min = cv2.erode(clean, np.ones((3, 3), np.uint8))
    local_max = cv2.dilate(clean, np.ones((3, 3), np.uint8))
    valid &= (local_max - local_min) <= np.maximum(0.02, 0.05 * depth)
    if confidence is not None:
        confidence = np.asarray(confidence)
        if confidence.shape != depth.shape:
            raise ValueError("Confidence/depth pixel mismatch")
        valid &= np.isfinite(confidence) & (confidence > 0)
    # Depth-edge rejection can isolate small background islands. Retain every
    # substantial component, including separated visible portions of a link.
    count, components, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8), 8)
    if count > 1:
        sizes = stats[1:, cv2.CC_STAT_AREA]
        retained = np.flatnonzero(sizes >= max(4, 0.02 * sizes.max())) + 1
        valid &= np.isin(components, retained)
    y, x = np.where(valid)
    z = depth[y, x]
    points = np.column_stack(((x - k[0, 2]) * z / k[0, 0],
                              (y - k[1, 2]) * z / k[1, 1], z)).astype(np.float32)
    metadata = dict(mask_pixels=int(mask.sum()), valid_points=len(points),
                    touches_image_border=bool(border_fraction > 0),
                    removed_fraction=1 - len(points) / max(int(mask.sum()), 1))
    return points, np.column_stack((x, y)).astype(np.int32), metadata


def camera_to_cad_transform(reg_transform, radius_m, query_center_normalized,
                            cad_center_normalized):
    """Convert a rigid normalized-space transform back to camera→CAD metres."""
    t = np.asarray(reg_transform, np.float64).copy()
    t[:3, 3] = radius_m * (t[:3, 3] + cad_center_normalized -
                           t[:3, :3] @ query_center_normalized)
    return t
