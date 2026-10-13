# Humanoid_Everyday

This dataset is a first-level project folder. Its manifests, audits and provenance records are under `metamanifest/`.
Audit: 2026-10-13T17:22:26.948316+00:00. Complete downloaded dataset audit.

## Counts and layout

Downloaded and validated MP4s: 98. Tasks with both views: 49.
Download or validation failures: 0.
Original MP4 storage: 252,604,626 bytes.
Dataset and metadata logical storage at audit: 514,915,328 bytes.

`<official-task-id>__<sanitized-task-name>/` contains the task prompt and task metadata, plus `third_person/` and `first_person/`. Reviewed third-person clips store unchanged source artifacts under `third_person/original/` and full generation references plus visually authored action prompts under `third_person/generation prompt/`. First-person source artifacts remain intact in `first_person/`. The downloader supports both direct and original/ source layouts.

`metamanifest/humanoid_everyday_manifest.csv` has one row per validated downloaded video. Each relative_file_path resolves relative to `metamanifest/` and points to that video's unchanged original MP4. `quality_audit.csv`, `download_failures.csv` and `annotation_discrepancies.csv` contain only this dataset's rows. `final_report.json` contains this dataset's counts, storage, quality distribution and limitations. Source snapshots, inventories, assessments and execution evidence are under `_sources/`.

## Original instructions and provenance

Prompts use the exact text_description field. Both viewpoints are associated in the same original taskData entry. Each view has its own unchanged copy of the instruction. Original `task_prompt.txt` files contain only the original field encoded in UTF-8, without an added newline. Original author spelling, punctuation and wording are retained.

Official source: https://humanoideveryday.github.io/data_viewer/data_viewer_page.html
Pinned revision: `85c82f5bfc85a669e039b43fefc2d5cd21232250`
Original metadata source: https://raw.githubusercontent.com/HumanoidEveryday/humanoideveryday.github.io/85c82f5bfc85a669e039b43fefc2d5cd21232250/data_viewer/data_viewer_page.html

The Humanoid HF card says Apache-2.0 and the official code README says MIT; the gallery repository has no explicit video asset license. Both claims are retained without inferring that either applies to the gallery MP4s.

## Humanoid world-model action prompts

Every reviewed third-person Humanoid clip and every available GR1 clip has a dedicated `generation prompt/` folder. `actiongeneration_prompt.txt` is the visually authored `humanoid_action_v1` instruction to pair with `first_frame.png`; it is the sole generation prompt file. `video.mp4` retains the complete original bytes. `action_review.json` records observed hand roles, coordination, terminal state, inspection coverage and uncertainties. Original task annotations remain under `original/`.

`world_model_prompts.csv` indexes completed prompts, and `world_model_prompt_report.json` records completed and pending reviews. `bimanual_second_review_report.json` records the focused second pass on coordinated or potentially coordinated hand actions, including prompt corrections and detailed contact evidence. The project `AGENTS.md` and this folder's `REVIEW_WORKFLOW.md` define the protocol. No cropping or shortening is used for these single-task references.

## Quality and verification

Official GitHub alternatives and publisher-linked Dropbox MP4s were inspected with native ffprobe properties, source hashes and decoded pixel comparisons. Identical files or decoded frames provide no pixel-quality upgrade. Different files are not substituted on filename or task similarity. Published raw lossless 640x480 RGB PNG sequences and LeRobot episode metadata were investigated, but no exact gallery-to-episode mapping was verified. The enormous full corpus was not downloaded. The book/plates annotation conflict and duplicate paper-towel third-person video are preserved and flagged in annotation_discrepancies.csv.

Published MP4 bytes remain untouched. No recompression, upscaling, interpolation, cropping, trimming or normalization is performed. Higher bitrate alone is not used to rank pixel quality. No higher-quality exact recording variant was verified, so no quality upgrade is claimed. Each successful MP4 passed source checksum/size checks, ffprobe and full FFmpeg decoding. The lossless PNG is decoded frame zero with displayed orientation retained, native dimensions checked and pixels verified against an independent frame-zero decode. Semantic instruction correctness and cross-view synchronization are not assumed.

```json
{
  "native_resolutions": {
    "448x960": 4,
    "640x480": 49,
    "480x360": 1,
    "568x320": 11,
    "720x1280": 18,
    "592x1280": 8,
    "1440x1920": 5,
    "1920x1440": 2
  },
  "displayed_resolutions": {
    "448x960": 4,
    "640x480": 49,
    "360x480": 1,
    "320x568": 11,
    "720x1280": 18,
    "592x1280": 8,
    "1440x1920": 5,
    "1920x1440": 2
  },
  "frame_rates": {
    "30.001023925731253": 1,
    "30.0": 33,
    "29.995989840930356": 1,
    "29.91643454038997": 1,
    "30.000659355150663": 1,
    "30.000621130872275": 1,
    "30.000622419552272": 2,
    "29.930715935334874": 1,
    "29.71719287745025": 1,
    "29.94392523364486": 1,
    "25.0": 35,
    "29.935897435897434": 1,
    "30.00064656565874": 1,
    "30.00082419846699": 2,
    "30.00107917550991": 1,
    "29.782833505687694": 1,
    "30.008778402585175": 1,
    "29.97569866342649": 1,
    "29.924050632911392": 1,
    "29.933035714285715": 1,
    "29.892857142857142": 1,
    "29.908536585365855": 1,
    "29.952076677316295": 1,
    "29.942528735632184": 1,
    "30.000977230528683": 1,
    "30.000990131687516": 1,
    "29.96385542168675": 1,
    "29.925": 1,
    "30.000869590422912": 1,
    "29.780309194467048": 1
  },
  "codecs": {
    "h264": 98
  }
}
```

## Reproduce or resume

From the project root, install FFmpeg/ffprobe (`brew install ffmpeg` on macOS), create `.venv` with `python3 -m venv .venv`, and install `requirements-robot-datasets.txt`. The shared downloader manages both datasets and stores each one's outputs under its own metamanifest:

```
.venv/bin/python -m pip install -r requirements-robot-datasets.txt
.venv/bin/python download_robot_datasets.py --phase discover
.venv/bin/python download_robot_datasets.py --phase research --workers 3
.venv/bin/python download_robot_datasets.py --phase pilot --workers 3
.venv/bin/python download_robot_datasets.py --phase full --workers 3
.venv/bin/python download_robot_datasets.py --phase audit
```

Use `--output <project-directory>` for another project root. Discovery pins revisions for safe resumes; use a new output directory for independent upstream versions. Downloads stream to temporary .part files with range/entity validation, exponential retry and atomic completion. Validated originals are skipped after checksum verification; corrupt originals are quarantined. Each dataset has its own `_sources/pilot_run.json` and `_sources/full_run.json`. Historical shared logs are retained as execution evidence.
