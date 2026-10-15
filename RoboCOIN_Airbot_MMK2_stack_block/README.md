# RoboCOIN Airbot MMK2 stack-block small sample

Browse the episodes in the [dataset viewer](https://wilsonnijc-bot.github.io/robot_object_deformation_detect/robotvideo/RoboCOIN_Airbot_MMK2_stack_block/).

Source: [RoboCOIN/Airbot_MMK2_stack_block](https://huggingface.co/datasets/RoboCOIN/Airbot_MMK2_stack_block),
revision `ee1901cfd4a45185b71b571d9fdd8608ab43c934` (LeRobot v2.1). License: **Apache-2.0** (dataset
card). The repository was readable without authentication; no credentials are stored here.

This is a convenience sample of the first ten of 236 episodes, not a representative sample.
All ten are visually reviewed with an action generation prompt; `episode_000003`–`episode_000009`
were added later in a second batch.

| Episode | Frames | Duration | Generation camera | Review status |
| --- | ---: | ---: | --- | --- |
| `episode_000000` | 176 | 5.87 s | `cam_head_rgb` | visually reviewed |
| `episode_000001` | 188 | 6.27 s | `cam_head_rgb` | visually reviewed |
| `episode_000002` | 283 | 9.43 s | `cam_head_rgb` | visually reviewed |
| `episode_000003` | 256 | 8.53 s | `cam_head_rgb` | visually reviewed |
| `episode_000004` | 214 | 7.13 s | `cam_head_rgb` | visually reviewed |
| `episode_000005` | 392 | 13.07 s | `cam_head_rgb` | visually reviewed |
| `episode_000006` | 280 | 9.33 s | `cam_head_rgb` | visually reviewed |
| `episode_000007` | 299 | 9.97 s | `cam_head_rgb` | visually reviewed |
| `episode_000008` | 192 | 6.40 s | `cam_head_rgb` | visually reviewed |
| `episode_000009` | 191 | 6.37 s | `cam_head_rgb` | visually reviewed |

All videos are 640 × 480 at a recorded 30 FPS. The robot is a dual-arm Airbot MMK2 with
five-finger hands. `cam_head_rgb` was chosen because it shows both hands and the blocks; the
other three views (front, left wrist, right wrist) are kept as originals.

## Read this first

* **All ten episodes are visually reviewed.** One dedicated reviewer per video inspected the complete
  head-camera clip, the initial PNG, the wrist and front cameras and the Parquet telemetry, and wrote
  `actiongeneration_prompt.txt` and `metadata/action_review.json` following `humanoid_action_v1`
  (by analogy: dual-arm, five-finger hands). Complete clips are kept. In every episode the image-left
  hand is the robot's left hand (telemetry and wrist cameras agree), the left hand places first, the
  right hand second, and the final stack is visibly complete.
* **The objects differ between episodes, and the shared publisher instruction ("square blocks in the
  centre … cylindrical blocks on top") describes only some of them.** Each review records the mismatch.

  | Episodes | Left hand | Right hand |
  |---|---|---|
  | 0, 2, 9 | green cube to the table centre | striped green cylinder onto the cube |
  | 1, 5 | glasses case across a red and a green block | blue prism stood on the case |
  | 3, 7 | blue "K" cube onto a white mini table | red cube onto the blue cube |
  | 4, 6 | blue "K" cube onto a yellow cube | green cube onto the blue cube (three-cube tower) |
  | 8 | light-blue cube onto a pink cube | purple cube onto a red cube (two separate stacks) |
* **Quality-gate exception.** The upscaled head-camera frames score 12.3–35.9 on the sharpness gate
  (first-person threshold 60); in the reviewed episodes this is mostly because of the large untextured
  table. They are admitted as a documented dataset-wide exception (`gate_result = exception`), granted
  after visual review of the first episodes; see [../data_quality_gate.md](../data_quality_gate.md).
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

* All downloaded files match the Hub's `X-Linked-Etag` (LFS SHA-256, or git blob SHA-1 for small
  files): 22 for episodes 0–2 (`metamanifest/metadata/download.json`) and 42 for episodes 3–9
  (`metamanifest/metadata/download_more.json`). The seven static files in both downloads are identical.
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
# episodes 0–2 (first batch)
../.venv/bin/python scripts/fetch_source.py /path/to/new_empty_directory 0 1 2
../.venv/bin/python scripts/prepare_sample.py /path/to/new_empty_directory
# episodes 3–9 (second batch)
../.venv/bin/python scripts/fetch_source.py /path/to/another_new_empty_directory 3 4 5 6 7 8 9
../.venv/bin/python scripts/prepare_sample.py /path/to/another_new_empty_directory --record download_more.json
../.venv/bin/python scripts/build_manifest.py
../.venv/bin/python ../scripts/quality_gate.py
```

`prepare_sample.py` builds only episodes whose folder does not exist yet, so reviewed episodes are never
rewritten, and it refuses to overwrite an existing provenance record. Without episode indices,
`fetch_source.py` downloads all ten episodes.

Requires the project `.venv` (Pillow, PyArrow) and FFmpeg with an AV1 decoder (libdav1d).
Manifest: [metamanifest/world_model_prompts.csv](metamanifest/world_model_prompts.csv). The publisher
card, `info.yaml`, `info.json` and subtask list are under `metamanifest/`.
