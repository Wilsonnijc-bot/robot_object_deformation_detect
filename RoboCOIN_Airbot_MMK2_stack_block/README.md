# RoboCOIN Airbot MMK2 stack-block small sample

Browse the episodes in the [dataset viewer](https://wilsonnijc-bot.github.io/robot_object_deformation_detect/robotvideo/RoboCOIN_Airbot_MMK2_stack_block/).

Source: [RoboCOIN/Airbot_MMK2_stack_block](https://huggingface.co/datasets/RoboCOIN/Airbot_MMK2_stack_block),
revision `ee1901cfd4a45185b71b571d9fdd8608ab43c934` (LeRobot v2.1). License: **Apache-2.0** (dataset
card). The repository was readable without authentication; no credentials are stored here.

This is a convenience sample of the first three of 236 episodes, not a representative sample.

| Episode | Frames | Duration | Generation camera |
| --- | ---: | ---: | --- |
| `episode_000000` | 176 | 5.87 s | `cam_head_rgb` |
| `episode_000001` | 188 | 6.27 s | `cam_head_rgb` |
| `episode_000002` | 283 | 9.43 s | `cam_head_rgb` |

All videos are 640 × 480 at a recorded 30 FPS. The robot is a dual-arm Airbot MMK2 with
five-finger hands. `cam_head_rgb` was chosen because it shows both hands and the blocks; the
other three views (front, left wrist, right wrist) are kept as originals.

## Read this first

* **Visually reviewed.** One dedicated reviewer per video inspected the complete head-camera clip,
  the initial PNG, the wrist and front cameras and the Parquet telemetry, and wrote
  `actiongeneration_prompt.txt` and `metadata/action_review.json` following `humanoid_action_v1`
  (by analogy: dual-arm, five-finger hands). Complete clips are kept. In every episode the image-left
  hand is the robot's left hand (telemetry and wrist cameras agree), the left hand places first, the
  right hand second, and the final stack is visibly complete.
* **Episode `episode_000001` is not the published cube-and-cylinder task.** The left hand lays a
  glasses case across a red and a green block, and the right hand stands a blue prism on the case.
  The publisher instruction (shared by all episodes) does not describe it; the review records this.
  In the other two episodes the objects are a green cube and a striped green cylinder.
* **Quality-gate exception.** The upscaled head-camera frames score 12.3–19.8 on the sharpness gate
  (first-person threshold 60), mostly because of the large untextured table. They are admitted as a
  documented visual-review exception (`gate_result = exception`); see [../data_quality_gate.md](../data_quality_gate.md).
* The generation `video.mp4` is lossless H.264, not the source bytes, because the sources are AV1.

## Layout

```text
stack_block/<episode_id>/
├── original/
│   ├── task_prompt.txt                                   # publisher instruction
│   ├── data/chunk-000/<episode_id>.parquet               # byte-identical to the source
│   ├── videos/chunk-000/observation.images.<camera>/<episode_id>.mp4   # four AV1 cameras, byte-identical
│   └── metadata/metadata.json                            # provenance, scene text, subtask timeline
└── generation input/
    ├── video.mp4                                         # complete head-camera video, lossless H.264
    ├── first_frame.png
    ├── actiongeneration_prompt.txt
    └── metadata/{metadata.json, action_review.json}
```

## Provenance and validation

* All 22 downloaded files match the Hub's `X-Linked-Etag` (LFS SHA-256, or git blob SHA-1 for small
  files); see `metamanifest/metadata/download.json`.
* Parquet row counts equal every camera's frame count and the episode length, and `frame_index` runs
  contiguously from zero.
* `video.mp4` is a lossless (`-crf 0`, `yuv420p`) H.264 encoding of the AV1 head video with no crop,
  resize or FPS change. Every frame's YUV checksum equals the checksum of the AV1 decode, the file
  decodes completely, and `first_frame.png` is decoded frame zero. This does not establish camera
  calibration or that the telemetry is semantically correct.
* The per-frame `subtask_annotation` in the Parquet is recorded as a run-length timeline in
  `original/metadata/metadata.json`. It is a publisher label, not visual evidence.

## Reproduce

```sh
../.venv/bin/python fetch_source.py /path/to/new_empty_directory
../.venv/bin/python prepare_sample.py /path/to/new_empty_directory
../.venv/bin/python ../quality_gate.py
../.venv/bin/python build_manifest.py
```

Requires the project `.venv` (Pillow, PyArrow) and FFmpeg with an AV1 decoder (libdav1d).
Manifest: [metamanifest/world_model_prompts.csv](metamanifest/world_model_prompts.csv). The publisher
card, `info.yaml`, `info.json` and subtask list are under `metamanifest/`.
