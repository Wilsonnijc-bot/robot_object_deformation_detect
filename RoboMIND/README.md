# RoboMIND small sample

Browse the episodes in the [dataset viewer](https://wilsonnijc-bot.github.io/robot_object_deformation_detect/robotvideo/RoboMIND/).

Source: [x-humanoid-robomind/RoboMIND](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND),
revision `e7ffe31d1fe983a42c3d7b79d192f554fc05b86e`. License: **Apache-2.0**.
The dataset is gated; the download used authorized Hugging Face access and no
credentials are stored in this folder.

This is a convenience sample of 18 real-robot episodes: eight Franka and ten Tien Kung
(nine tasks, two episodes per task), not a representative sample of RoboMIND's 107k trajectories. Episodes `0923_164719` and `1211_104950` were the original
sample; the others were added later. Six AgileX dual-arm episodes were removed on 2026-10-15;
their provenance remains in `metamanifest/metadata/`. Six more episodes (Franka bread-on-plate,
Tien Kung upright-cup and banana-on-plate; the first two complete episodes of each task's first
archive part) were added on 2026-10-15 with `fetch_extra.py` (`metamanifest/metadata/download_extra.json`).

| Embodiment | Task | Episode | Frames | Generation reference | Generation camera |
| --- | --- | --- | ---: | ---: | --- |
| h5_franka_3rgb | pick apple into drawer | `0923_164719` | 600 | 192 | `camera_right` |
| h5_franka_3rgb | pick apple into drawer | `0923_164743` | 600 | 210 | `camera_right` |
| h5_franka_3rgb | pick up strawberry in bowl | `1017_102324` | 134 | 134 (complete) | `camera_right` |
| h5_franka_3rgb | pick up strawberry in bowl | `1017_102820` | 133 | 133 (complete) | `camera_right` |
| h5_franka_3rgb | slide close cabinet | `0923_111806` | 200 | 200 (complete) | `camera_right` |
| h5_franka_3rgb | slide close cabinet | `0923_111834` | 200 | 200 (complete) | `camera_right` |
| h5_tienkung_gello_1rgb | close the drawer under the combination cabinet | `1211_104950` | 759 | 759 (complete) | `camera_top` |
| h5_tienkung_gello_1rgb | close the drawer under the combination cabinet | `1211_111642` | 584 | 584 (complete) | `camera_top` |
| h5_tienkung_gello_1rgb | pick up tennis ball and place it in box | `1212_105714` | 719 | 719 (complete) | `camera_top` |
| h5_tienkung_gello_1rgb | pick up tennis ball and place it in box | `1212_105755` | 543 | 543 (complete) | `camera_top` |
| h5_tienkung_gello_1rgb | press down toaster | `1211_172414` | 455 | 455 (complete) | `camera_top` |
| h5_tienkung_gello_1rgb | press down toaster | `1211_172928` | 522 | 522 (complete) | `camera_top` |
| h5_franka_3rgb | place in bread in plate | `1014_101050` | 135 | 135 (complete) | `camera_right` |
| h5_franka_3rgb | place in bread in plate | `1014_101240` | 160 | 160 (complete) | `camera_right` |
| h5_tienkung_gello_1rgb | upright white cup | `1211_141210` | 700 | 700 (complete) | `camera_top` |
| h5_tienkung_gello_1rgb | upright white cup | `1211_141244` | 731 | 731 (complete) | `camera_top` |
| h5_tienkung_gello_1rgb | place yellow banana on plastic plate | `1212_141349` | 674 | 674 (complete) | `camera_top` |
| h5_tienkung_gello_1rgb | place yellow banana on plastic plate | `1212_141428` | 636 | 636 (complete) | `camera_top` |

All frames are 640 × 480.

**Franka episodes keep only the right RGB camera** (`camera_right`, the generation camera).
On 2026-10-15 `strip_franka_cameras.py` rewrote each Franka `trajectory.hdf5` without the left and
top RGB cameras and without all depth images, and deleted their decoded previews. Robot state,
language fields and the `camera_right` JPEG bytes are unchanged (verified). The original source hash
is kept as `hdf5.source_sha256` in each `original/metadata/metadata.json`.

## Read this first

* **No frame rate is stored.** The HDF5 files hold JPEG frames with no timestamps.
  Videos use a nominal 30 FPS (`fps_source: nominal_assumed_not_recorded_in_source`),
  so durations are not real time.
* **Sharpness gate** ([../data_quality_gate.md](../data_quality_gate.md)): the ten Tien Kung
  episodes (first-person head camera) pass at 89–251. The eight Franka `camera_right` episodes
  (third-person) score 15–27 against a threshold of 120 and are accepted as documented
  exceptions (`gate_result = exception`). Scores are in each `generation input/metadata/metadata.json` and the manifest.
* **RoboMIND ships no MP4s.** Every video is rendered from the stored JPEGs.

## Layout

```text
<embodiment>__<task>/<episode_id>/
├── original/
│   ├── data/trajectory.hdf5            # source member (Franka: right RGB camera only)
│   ├── task_prompt.txt                 # publisher instruction, where one exists
│   ├── decoded_videos/<camera>.mp4     # lossy previews of the retained RGB cameras (CRF 14)
│   └── metadata/metadata.json
└── generation input/
    ├── video.mp4                       # lossless prefix of the generation camera
    ├── first_frame.png
    ├── actiongeneration_prompt.txt
    ├── full_sequence_actiongeneration_prompt.txt   # only when the reference is shortened
    └── metadata/
        ├── metadata.json
        └── action_review.json
```

`decoded_videos/` are derived renderings kept for preview, not source bytes; the HDF5
is the source. The Franka episode has no publisher instruction (the file stores none
and the task is absent from `RoboMIND_v1_2_instr.csv`), so it has no `task_prompt.txt`.

## Provenance and validation

* Franka `pick apple into drawer`: both episodes extracted unchanged from the official 180 MB
  example zip; its SHA-256 matches the Hugging Face LFS hash.
* Every other episode comes from the first part of an official split tarball (10 GB+ per part).
  A 250–700 MB prefix was range-downloaded and complete HDF5 members were extracted in order.
  The whole-part hash was not verified; prefix and member hashes are in
  `metamanifest/metadata/download.json` (first three) and `download_more.json` (the rest).
  The extra Franka tasks come from `benchmark1_0_compressed/h5_franka_3rgb`; the extra Tien Kung
  and AgileX tasks from `benchmark1_0_compressed` too. A first choice of Franka task
  (`2024_09_20_open_the_box`) was replaced because its episodes are 440–550 MB each.
* Every `video.mp4` is a lossless H.264 encoding (`yuvj420p`, full range) of the first
  N stored JPEGs, with no crop, resize or FPS conversion. Each frame's YUV checksum
  equals the checksum of the corresponding JPEG decoded directly, each file decodes
  completely, and the PNG equals decoded frame zero in RGB. This does not establish
  camera calibration or semantic correctness of the robot telemetry.
* **Channel order.** RoboMIND's loader decodes JPEGs with OpenCV (BGR order) and converts to RGB only
  for `h5_franka_3rgb`, `h5_franka_1rgb`, `h5_ur_1rgb` and `h5_franka_fr3_dual`. For every other
  embodiment the stored JPEGs therefore hold red and blue swapped relative to a standard decoder.
  Franka decodes to natural colours as stored. **Tien Kung derivatives swap red and blue back**
  (`color_args` in `robomind_common.py`): the yellow banana, yellow-green tennis ball and brown bread
  crust confirm the correction. An earlier version of this sample decoded Tien Kung without the swap;
  on 2026-10-15 `fix_channel_order.py` and `finalize_sample.py` rebuilt every Tien Kung preview, first
  frame and generation video, four prompts had colour words corrected, and each Tien Kung review
  carries a `channel_order_note` (its colour terms were observed on swapped frames). The HDF5 files
  are unchanged. Tien Kung generation videos are lossless encodes of the corrected 4:2:0 decode, and
  the conditioning PNG equals their frame zero.
* Only real-robot data (`sim: False`) is included. The `failure_data/` episodes
  inspected were simulation and were excluded.

## Prompts and reviews

One dedicated reviewer inspected each complete generation-camera video at every second
decoded frame, plus native-resolution boundaries and the initial PNG. Franka follows
`concise_action_v1`. Tien Kung follows `humanoid_action_v1` by analogy, because it
is a dexterous-hand humanoid outside Humanoid Everyday and GR1. Its robot-relative side could
not be established from the overhead view in any episode, so those prompts describe the arm by
the image edge it enters from; palm and thumb orientation were left out where the review could
not resolve them.

* Franka: the bread-on-plate episodes are single placements, kept whole. The two apple-into-drawer episodes keep only the first manipulation (the drawer
  opening, 192 and 210 frames); the strawberry and cabinet episodes are single short tasks.
* Tien Kung: all ten are single continuous tasks, kept whole.

## Reproduce

```sh
HF_TOKEN=... ../.venv/bin/python fetch_source.py /path/to/new_work_directory
../.venv/bin/python prepare_sample.py /path/to/new_work_directory      # first sample episodes
HF_TOKEN=... ../.venv/bin/python fetch_more.py /path/to/new_work_directory
../.venv/bin/python prepare_more.py /path/to/new_work_directory        # other episodes
# author prompts and action_review.json, then:
../.venv/bin/python finalize_sample.py
../.venv/bin/python strip_franka_cameras.py
../.venv/bin/python ../quality_gate.py
../.venv/bin/python build_manifest.py
```

Requires the project `.venv` (Pillow, h5py, NumPy) and FFmpeg. The fetch and prepare scripts
still list the removed AgileX episodes; delete them again after a rebuild. Manifest:
[metamanifest/world_model_prompts.csv](metamanifest/world_model_prompts.csv). Publisher
README, instruction CSV and HDF5 format notes are under `metamanifest/`.
