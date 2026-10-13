# TAVP real-world robot video dataset

This local dataset contains exactly **23 real-world demonstration MP4s** linked in the official TAVP project page's Franka evaluation, Dobot evaluation, Dobot failure, and Dobot robustness galleries: **8 Franka Research 3** and **15 Dobot Nova 2**. Both successful and failed executions are included. RLBench simulation videos and other unselected repository assets are excluded.

Official sources:

- Project: https://hcplab-sysu.github.io/TAVP/
- Repository: https://github.com/HCPLab-SYSU/TAVP
- Pinned commit: `fcb05cd3430f238cfe1efd26b1662c309f2d79b8`

The upstream page currently calls the method TVVE; the requested local dataset name remains TAVP. Numeric IDs follow the pinned page's gallery/source order. CSV information is matched by the full decoded repository-relative MP4 path, never by row order or filename alone.

## Contents

Each of the 23 numerically prefixed video directories contains two folders: `original/` for source data and `generation input/` for generation references. The files under `original/` are:

- `video.mp4`: exact original repository bytes, verified against repository file size and Git blob SHA-1, with a separately calculated SHA-256. No re-encoding, resizing, cropping, or recompression occurred.
- `task_prompt.txt`: one action instruction, with no metadata or commentary.
- `metadata/metadata.json`: source identity, instruction provenance, original caption, execution status, scenario/robustness type, original CSV records, video technical metadata, checksums, paths, and validation results. Unknown fields are JSON `null`.

First-frame PNGs live under `generation input/`, beside the generation prompt
(or inside `first_substep/`). The source manifest's `relative_first_frame_path`
selects the unchanged full source-view frame: for Franka this is the conditioning
PNG; for Dobot it is `generation input/source_first_frame.png`, distinct from the
cropped middle-view conditioning PNG. No first-frame PNG remains under `original/`.

Dataset-wide files:

- `manifest.csv`: 23 rows with relative artifact paths, annotation provenance, source URLs, technical fields, SHA-256, and download/validation status. CSV unknowns are empty cells; booleans are `true`/`false`.
- `download_dataset.py`: discovery, resumable downloads, frame extraction, metadata/manifest generation, and complete validation.
- `validation_report.json`: summary, tool versions, uncertain annotations, and explicit failures from the most recent run.
- `sources/`: pinned repository tree and HTML, live-page snapshot, repository/branch snapshots, original supplied CSV, and generated `source_catalog.json`.
- `tests/test_download_dataset.py`: offline HTTP recovery and stable-identity regression checks.

## Verified dataset summary

All 23 folders passed full video decode, source integrity, PNG decode/dimension/pixel checks, prompt checks, and metadata checks.

| Property | Count/value |
|---|---:|
| Original MP4 bytes | 78,365,401 bytes (78.365 MB; 74.735 MiB) |
| Franka Research 3 | 8 |
| Dobot Nova 2 | 15 |
| Successful executions | 19 |
| Failed executions | 4 |
| H.264 videos | 23 |
| 1280 × 720, 30 fps | 8 |
| 1920 × 480, 20 fps | 14 |
| 2880 × 720, 30 fps | 1 |
| Individually published instructions | 13 |
| Transferred task-level instructions | 5 |
| Inferred generic instructions | 5 |
| Annotation review flags | 10 |

The wide Dobot frames are the authors' original multi-view video format and are preserved intact. The upstream Dobot filenames contain `_converted`; these are the official publicly linked MP4s, with no corresponding unconverted/higher-quality originals in the pinned repository tree. Additional unlinked Dobot assets are not substitutions for these 23 identities.

## Annotation provenance

Only 13 videos have independently published language instructions. The clean prompt files use those verbatim, removing surrounding webpage markup/quotation delimiters.

- `published_verbatim`: IDs `001`–`013`. `original_task_instruction` is the individually published text.
- `transferred_from_matching_task`: IDs `016`, `017`, `018`, `022`, `023`. The instruction is transferred from standard Dobot Grape Picking video `011`. Each clip's `original_task_instruction` is `null`; `matching_task_instruction_video_id` records `011`. The supplied CSV's original label `transferred_from_matching_success_task` is preserved inside `source_csv_records` and normalized in the manifest/metadata.
- `inferred_from_task_category`: IDs `014`, `015`, `019`, `020`, `021`. Generic instructions come from the supplied updated CSV. Exact intended objects/targets are not individually specified by the authors, and `original_task_instruction` is `null`.

All ten non-individual instructions retain `manual_verification_needed: true`. Technical validation passing does not verify those annotations. Robustness captions describe test conditions and are preserved separately; they are never used as action instructions. Three object-generalization videos share a caption, but have distinct source paths, IDs, and directories.

The available input was `/Users/nijiachen/Downloads/tavp_real_23_with_usable_task_prompts.csv`, copied intact into `sources/`. No second TAVP CSV was found in the workspace, Downloads, Desktop, Documents, or attachments. All useful input columns survive under `source_csv_records[].columns`; the pinned official HTML independently verifies published instructions, captions, robot/task identity, and outcomes.

## Rebuild or validate

Requires Python 3.9+ and FFmpeg/FFprobe on `PATH`; no third-party Python packages are required.

```sh
# Download missing/corrupt originals, repair frames, regenerate prompts/metadata,
# and validate every video and artifact. Verified originals are not downloaded again.
python3 TAVP/download_dataset.py

# Validate existing content without repairing videos, frames, prompts, metadata,
# catalog, or manifest. Refreshes validation_report.json with current results.
python3 TAVP/download_dataset.py --verify-only

# Rebuild to another directory with the bundled pinned sources and CSV annotations.
python3 TAVP/download_dataset.py --output /path/to/new/TAVP

# Merge additional CSVs by video URL/path. Repeat --csv for multiple inputs.
python3 TAVP/download_dataset.py --csv /path/to/original.csv --csv /path/to/updated.csv

# Optional controls; timeout is network socket timeout, decode-timeout is per command.
python3 TAVP/download_dataset.py --workers 4 --retries 4 --timeout 60 --decode-timeout 300

# Run offline recovery/discovery regression checks.
python3 -m unittest discover -s TAVP/tests -v
```

IDs and annotation sources are deterministic for the pinned HTML and input snapshots. Copy the script together with `sources/` to retain the same CSV-derived annotations during a fresh rebuild. If no CSV/catalog is available, the script retrieves the pinned official HTML/tree and uses clearly marked task-level transfers or generic inferred instructions for unpublished clips. New CSV files must contain a recognized video URL column; unmatched identities and conflicting published annotations fail explicitly, while all original columns are retained.

Downloads use `.mp4.part` files, HTTP Range resumption, retry/backoff, and a socket timeout. Servers ignoring Range trigger a safe restart. Completed partials and existing MP4s are trusted only after exact size and pinned Git blob verification. A failed download or validation is printed, included in the manifest/report, and causes a nonzero exit code. Rerunning repairs missing/corrupt artifacts. The MP4s are decoded from beginning to end using strict FFmpeg error handling, and decoded frame count is checked against the MP4-declared count when present.

SHA-256 represents the original MP4 bytes; `first_frame_sha256` represents PNG bytes, and `first_frame_rgb24_sha256` represents decoded RGB pixel bytes. Frame extraction disables automatic rotation and applies no scale/crop filters. Runtime timestamps/download status can change between runs; pinned source identities, MP4 bytes, prompts, and annotation provenance remain stable.

## Programmatic loading

```python
import csv
import json
from pathlib import Path

root = Path("TAVP")
with (root / "manifest.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

successful = [r for r in rows if r["execution_status"] == "success"
              and r["validation_status"] == "passed"]
published = [r for r in successful if r["prompt_provenance"] == "published_verbatim"]

for row in published:
    video = root / row["relative_video_path"]
    first_frame = root / row["relative_first_frame_path"]
    prompt = row["task_instruction"]
    metadata = json.loads((root / row["relative_metadata_path"]).read_text())
```

## World-model generation references

The first-frame-conditioned generation dataset is described in [WORLD_MODEL_README.md](WORLD_MODEL_README.md). It adds visually reviewed concise prompts, middle-view-only Dobot references, and first-subtask clips where useful. Use `world_model_manifest.csv` for the preferred generation inputs. The prompting rule is maintained in the project [AGENTS.md](../AGENTS.md), with examples in [WORLD_MODEL_PROMPT_PROTOCOL.md](WORLD_MODEL_PROMPT_PROTOCOL.md). Original source videos and annotations remain preserved.
