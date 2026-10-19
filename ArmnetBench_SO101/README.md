# ArmnetBench v0.1 SO-101 small sample

Source: [armnet/armnetbench_v01_lerobot_so101](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_so101),
revision `1db69c82b589c7172ee31a6b8f7ef58d1fc2c663` (LeRobot v3.0). License: **Apache-2.0** (dataset card). The
repository is public; no credentials are stored here.

The source has 2,499 episodes of a single-arm SO-101 on 8 tasks: 50 human teleoperated demonstrations per task plus
rollouts of 7 policies. This sample is **10 teleoperated episodes** (publisher label `successful`): the first
demonstration of each of the eight tasks plus the second `ring_insert` and `block_stack` demonstrations. It is a
convenience selection, not a representative one. All videos are 20 FPS; the publisher's success label is not a visual
verification.

| Task | Episode | Frames | Duration | Sharpness | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| `block_stack` | `episode_000560` | 228 | 11.40 s | 18.9 | fail |
| `block_stack` | `episode_000561` | 245 | 12.25 s | 23.8 | fail |
| `cable_clip` | `episode_002159` | 632 | 31.60 s | 61.8 | fail |
| `cable_unclip` | `episode_002449` | 241 | 12.05 s | 69.5 | fail |
| `eye_drops_to_basket` | `episode_001839` | 242 | 12.10 s | 53.9 | fail |
| `eye_drops_to_shelf` | `episode_001549` | 246 | 12.30 s | 44.3 | fail |
| `ring_insert` | `episode_000255` | 231 | 11.55 s | 23.1 | fail |
| `ring_insert` | `episode_000256` | 216 | 10.80 s | 25.3 | fail |
| `tool_insert` | `episode_000910` | 608 | 30.40 s | 30.2 | fail |
| `tool_removal` | `episode_001200` | 389 | 19.45 s | 24.8 | fail |

## Read this first

* **Review is pending.** No `actiongeneration_prompt.txt` or `metadata/action_review.json` exists yet; each
  `generation input/metadata/metadata.json` has `review_status: pending_visual_review`. They need one dedicated
  visual review per video, as for the other datasets.
* **Only the `front` camera (1024 × 576) is retained**, and it is the generation view, scored as third-person. The
  source's overhead `top` and `wrist` cameras were deliberately dropped from this sample.
* **All ten first frames fail the third-person sharpness gate (120)** at 18.9–69.5; the low `front` camera sees a
  mostly dark mat. They are kept and flagged (`gate_result = fail`); no exception has been granted.
* **The originals are derivatives, not source bytes.** LeRobot v3 packs many episodes into one AV1 shard per camera,
  so no per-episode source file exists. The front video is the episode's frames re-encoded as lossless H.264, and
  every frame's YUV checksum equals the matching frame of the full shard decode. The Parquet holds the episode's
  rows unchanged. Metadata records the source shard, its SHA-256 and the frame range.
* Because the generation reference is the complete, unchanged `front` cut, `generation input/` holds no video:
  `metadata/metadata.json` points to it in `original/` (`generation_video_is_source: true`).

## Layout

```text
<task>/<episode_id>/
├── original/
│   ├── task_prompt.txt                                   # publisher instruction
│   ├── data/chunk-000/<episode_id>.parquet               # episode rows, values unchanged
│   ├── videos/chunk-000/observation.images.<camera>/<episode_id>.mp4   # front camera only; lossless H.264
│   └── metadata/metadata.json                            # provenance, outcome label, shard checksums
└── generation input/
    ├── first_frame.png                                   # front camera, frame 0
    └── metadata/metadata.json
scripts/        fetch_source.py, prepare_sample.py, ab_common.py
metamanifest/   source README/info/tasks, download record (SHA-256 of every downloaded shard; it still lists the top and wrist shards fetched before they were dropped)
```

Rebuild: `fetch_source.py <new empty dir under /tmp>`, then `prepare_sample.py <that dir>`; delete the directory afterwards.
