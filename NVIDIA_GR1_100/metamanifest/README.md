# NVIDIA_GR1_100

This dataset is a first-level project folder. Its manifests, audits and provenance records are under `metamanifest/`.
Audit: 2026-10-13T17:22:26.948316+00:00. Complete downloaded dataset audit.

## Counts and layout

Downloaded and validated MP4s: 92. Metadata rows: 100. Metadata without repository video: 11.mp4, 22.mp4, 28.mp4, 44.mp4, 5.mp4, 7.mp4, 8.mp4, 87.mp4. Videos without instructions: none.
Download or validation failures: 0.
Original MP4 storage: 142,323,121 bytes.
Dataset and metadata logical storage at audit: 511,650,966 bytes.

`<original-video-id>/original/` contains original.mp4, task_prompt.txt and metadata/metadata.json. The sibling `generation input/` contains the complete unchanged generation video, first frame, visually authored actiongeneration_prompt.txt, metadata/action_review.json and metadata/metadata.json. IDs remain unchanged, including numbering gaps. The downloader supports both direct and original/ source layouts.

`metamanifest/nvidia_gr1_100_manifest.csv` has one row per validated downloaded video. Each relative_file_path resolves relative to `metamanifest/` and points to that video's unchanged original MP4. `quality_audit.csv`, `download_failures.csv` and `annotation_discrepancies.csv` contain only this dataset's rows. `final_report.json` contains this dataset's counts, storage, quality distribution and limitations. Source snapshots, inventories, assessments and execution evidence are under `_sources/`.

## Original instructions and provenance

Prompts use the exact text field whose file_name equals the original repository MP4 basename. CSV row order is never used for matching. Original `task_prompt.txt` files contain only the original field encoded in UTF-8, without an added newline. Original author spelling, punctuation and wording are retained.

Official source: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1
Pinned revision: `b114c39c0d93393d86bbe875be8bf17b75b85082`
Original metadata source: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1/resolve/b114c39c0d93393d86bbe875be8bf17b75b85082/metadata.csv

The official NVIDIA dataset license is CC-BY-4.0.

## Humanoid world-model action prompts

Every reviewed third-person Humanoid clip and every available GR1 clip has a dedicated `generation input/` folder. `actiongeneration_prompt.txt` is the visually authored `humanoid_action_v1` instruction to pair with `first_frame.png`; it is the sole generation prompt file. The generation video is the unchanged `original/original.mp4`; `metadata/metadata.json` points to it with `relative_video_path` and no copy is kept. `metadata/action_review.json` records observed hand roles, coordination, terminal state, inspection coverage and uncertainties. Original task annotations remain under `original/`.

`world_model_prompts.csv` indexes completed prompts, and `world_model_prompt_report.json` records completed and pending reviews. `bimanual_second_review_report.json` records the focused second pass on coordinated or potentially coordinated hand actions, including prompt corrections and detailed contact evidence. The project `AGENTS.md` and this folder's `REVIEW_WORKFLOW.md` define the protocol. No cropping or shortening is used for these single-task references.

## Quality and verification

All selected files are original MP4 assets in the official gr1/ repository at the pinned revision. Hugging Face viewer previews were excluded. File sizes and publisher LFS SHA-256 hashes are verified. The published camera viewpoint is attributed to the dataset card; physical camera placement has not been independently calibrated.

Published MP4 bytes remain untouched. No recompression, upscaling, interpolation, cropping, trimming or normalization is performed. Higher bitrate alone is not used to rank pixel quality. No higher-quality exact recording variant was verified, so no quality upgrade is claimed. Each successful MP4 passed source checksum/size checks, ffprobe and full FFmpeg decoding. The lossless PNG is decoded frame zero with displayed orientation retained, native dimensions checked and pixels verified against an independent frame-zero decode. Semantic instruction correctness and cross-view synchronization are not assumed.

```json
{
  "native_resolutions": {
    "1280x800": 92
  },
  "displayed_resolutions": {
    "1280x800": 92
  },
  "frame_rates": {
    "20.0": 92
  },
  "codecs": {
    "h264": 92
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

The conditioning `first_frame.png` lives only in `generation input/`; its original-byte checksum remains recorded in source metadata. Humanoid first-person media retain their source layout; their JSON metadata lives in `metadata/`.
