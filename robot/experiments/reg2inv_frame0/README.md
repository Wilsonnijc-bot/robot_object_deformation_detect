# Reg2Inv with a per-video frame-0 normal reference

All 45 selected videos (15 COSMOS2.5, 15 COSMOS3, 15 LVP_ROBOWM), every frame,
Link2/upperarm and Link5/forearm, using Reg2Inv registration and scoring only.
Expected outputs: 90 video/link scores and 9,930 frame/link results. The October 13
authorization covered the first 30 videos; on October 14 the user extended it to
all 45 and asked for the CAD-reference results to be deleted.

Each video/link has its own frozen refined frame-0 point cloud, its own normal
feature bank and reference-derived normalization. No later-frame points enter
that bank. No CAD geometry, CAD banks, FoundationPose or mesh reconstruction is
used. The pretrained Reg2Inv checkpoint, feature extraction, 1-NN discrepancy,
64-neighbor pooling, maximum patch scoring and validated float64 GPU RANSAC
backend come from the CAD-reference Reg2Inv experiment, which was deleted on 2026-10-15
(locally and on ERIS). The scoring code it shared is kept byte-identical in `reg2inv_core/`;
the checkpoint is on ERIS under `/scratch/z/zy992/models/reg2inv/`. Its environment was deleted, so
rerunning needs a rebuilt environment (`reg2inv_core/README.md`). Raw scores remain raw;
frame 0 is scored normally rather than manually forced to zero.

Observations come from `observations/` on ERIS (`extract_observations.py`): the
depth-filtered clouds, pixels, intrinsics and depth-filter receipts, copied out of the
deleted CAD run's archives with every array compared exactly. Link2 then applies the
detached-island filter described below. Every observation, frame-0 input, core
source file and checkpoint is hash-verified. `preflight.py` requires a real Slurm H200 allocation
and native scorer equivalence before processing the experiment. The inherited
adapter calls its generic reference variables `cad_*`; those variables contain
only this run's frame-0 cloud and coordinates.

One H200 is allocated independently of video-worker concurrency. Matched
12-video, eight-frame trials compare 5/8/12 workers; additional workers are
retained only with >2% measured throughput improvement and memory headroom.
Every worker builds/loads one video's normal banks, processes all its frames,
then takes the next video. Each stage and allocation has durable telemetry.

Remote owner:
`/PHShome/zy992/Wilson/deformationdetection/workspace-reg2inv-frame0-20261013/robot/experiments/reg2inv_frame0/`.
Results: `results/20261014-45/` (all 45 videos). Cleaned on 2026-10-15 to what the scores
and replay need: per-frame `cases/` (observed points, point scores, registration), frame-0
`inputs/`, `scores.csv`, `frame_scores.csv`, `metadata/` and the locally built `replay/`.
The normal feature banks, the capacity-trial `pilot/`, compile caches and the superseded
roots `results/20261014/` (first 30 videos, job 5882225) and `results/20261013/` (job 5881590,
preflight failure) were deleted. Kept for the Link2 depth-filter follow-up: ERIS
`observations/` (the only copy of the depth-filtered clouds) and `results/link2_island_analysis/`.

`wait_and_submit.py` waits for Slurm recovery for at most approximately 16 hours,
submits once, and observes only its recorded allocation. It performs no GPU work
on the login node. The watcher runs detached (`setsid nohup`) and records itself as `frame0-submit-20261014`;
the allocation uses `reg2inv-frame0-JOBID` / `frame0-experiment-JOBID`. Logs and
explicit exit files were under `results/20261014/metadata/` (deleted).

Stop by writing `metadata/stop.request`; the watcher prevents submission or
cancels only its recorded job. There are no dependent allocations. Verify
cancellation before marking the run stopped. Never signal another user's job
or an unrelated allocation.

This experiment measures deviations from frame 0. Frame 0 is an assumed normal
reference, not verified healthy geometry; newly visible surfaces may have no
reference coverage. Saved registration fitness and distances are fit diagnostics,
not pose-ground-truth accuracy. No artifacts from the earlier run are deleted.

## Link2 detached-island filter (October 14, 2026)

Job 5881590 failed its preflight. COSMOS3_0001 Link2 scored
`insufficient_score_support` (48 selected feature points; 64 required). Link2 and
Link5 already share the same Link5 density filter
(`robot/preprocessing/depth/link5_depth_filter.py`). That filter keeps locally dense
regions, so small Link2 mask islands that see background through a gap in the link
survive it. These islands lie about 0.33–0.74 m behind the link. The official
reference radius is the farthest-point distance, so the islands inflated all 15
COSMOS3 Link2 frame-0 radii from about 0.15–0.21 m to 0.61–0.79 m. That shrinks
the normalized query below scoreable support.

At the user's direction, `link2_islands.py` now runs after the shared filter for
Link2 only, on frame 0 and every test frame. It finds the 8-connected islands of
retained depth-grid pixels, then removes a detached island when its median depth
differs from the largest island's median depth by more than 0.25 m. Across all
4,230 saved Link2 observations, detached islands lie within 0.17 m or at least
0.334 m from that depth; none fall in between. The filter changes 3,077 Link2
observations, removing at most 4.4% of any one cloud's points. Islands at link
depth are kept. XYZ is never moved. Each archive records its filter settings,
counts and removed pixels.

Link5 is unchanged. It has comparable islands in 257 of 4,230 observations. The
15 COSMOS3 Link2 references changed; the COSMOS2.5 Link2 frame-0 references had
no qualifying islands. This is an explicit numerical change relative to the
October 13 design, so do not compare Link2 scores across them without noting it.

## Labels and replay

`label_evaluation.py` compares the video mean score with the selected-45 labels in
`selected_45_matched_videos_styledv2.xlsx` (Link2 column AD, Link5 column AB): AUROC
for severe versus none (moderate excluded, ties half) with stratified bootstrap
intervals, within-generator AUROC and Spearman ρ. Outputs:
`results/20261014-45/label_evaluation.md` and `metadata/label_evaluation*.{json,csv}`.

`build_replay.py` writes `results/20261014-45/replay/index.html`, which opens from
`file://`. It reuses the MoSE3 error-signal replay's viewer (`common3d.js`,
`site.css`, vendored three.js, packer and quantizer) and its display videos. Each
frame's observed Link2/Link5 points are aligned onto the frame-0 normal reference
with Reg2Inv's registration and coloured by their saved point scores, white for
normal and red for error. A shared scale tops out at the 95th percentile of all
scores after frame 0. Run it from the repository root with OpenCV available, for
example `uv run --no-project --with numpy --with opencv-python-headless python -m
robot.experiments.reg2inv_frame0.build_replay`.
