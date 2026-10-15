"""Link2-only rejection of detached background islands after the shared depth filter.

Link2 and Link5 observations already share the Link5 ray-normalized density
filter. It removes sparse samples but keeps locally dense regions, so a Link2 mask
gap that sees background survives as a small detached island of retained pixels.
Across all 4,230 saved Link2 observations of this cohort, detached islands lie
either within 0.17 m of the main island's median depth or at least 0.334 m behind
it. The 0.25 m threshold sits in that empty interval. Islands at link depth, such
as a link split by an occluder, are kept. Retained XYZ is never moved.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy import ndimage

LINKS = ('link2',)


@dataclass(frozen=True)
class Config:
    depth_gap_m: float = .25


def settings(config=Config()):
    return dict(method='link2_detached_depth_island_v1', links=list(LINKS), connectivity=8,
                main_island='largest 8-connected island of retained depth-grid pixels',
                rule='remove a detached island when |median z(island) - median z(main)| > depth_gap_m',
                **asdict(config))


def filter_islands(points_m, pixels_yx, link, config=Config()):
    keep = np.ones(len(points_m), bool)
    info = dict(settings(config), applied=link in LINKS, islands=0, removed_islands=0,
                removed_points=0, retained_points=len(points_m))
    if link not in LINKS or len(points_m) == 0:
        return keep, info
    grid = np.zeros(pixels_yx.max(0) + 1, bool)
    grid[pixels_yx[:, 0], pixels_yx[:, 1]] = True
    labels, count = ndimage.label(grid, structure=np.ones((3, 3), bool))
    component = labels[pixels_yx[:, 0], pixels_yx[:, 1]]
    main = int(np.argmax(np.bincount(component)[1:]) + 1)
    depth = float(np.median(points_m[component == main, 2]))
    for label in range(1, count + 1):
        selected = component == label
        if label != main and abs(float(np.median(points_m[selected, 2])) - depth) > config.depth_gap_m:
            keep[selected] = False
            info['removed_islands'] += 1
    info.update(islands=int(count), main_island_points=int((component == main).sum()),
                main_island_median_depth_m=depth, removed_points=int((~keep).sum()),
                retained_points=int(keep.sum()))
    return keep, info
