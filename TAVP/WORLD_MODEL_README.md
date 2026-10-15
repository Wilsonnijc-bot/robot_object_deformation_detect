# First-frame-conditioned robot video generation references

Each original video has a dedicated visual-review subagent. Reviewers inspect the complete MP4 at every second decoded frame, inspect native-resolution action transitions, then inspect the native first frame to author concrete motion prompts. The action reviews record observed chronology, actual outcomes, source-frame boundaries, ambiguous details, and inspected frame indices. Published task text is context; a filmed failure is described as a filmed failure.

## Files and preferred generation reference

```text
TAVP/
  001_franka_collect_fruits_standard_success/
    original/
      video.mp4                       # original source, unchanged
      task_prompt.txt                 # original dataset task annotation, unchanged
      metadata/metadata.json          # source metadata with updated artifact paths
    generation input/
      prompt for original multi step action(reproducing the original video).txt
                                      # full original-sequence generation prompt
      metadata/action_review.json     # dedicated reviewer's detailed observations
      first_substep/                  # selected shorter complete first action
        video.mp4
        first_frame.png
        prompt.txt                    # generation prompt ends at the retained action
        metadata/metadata.json
  world_model_manifest.csv            # one preferred generation reference per source video
  world_model_validation_report.json
  build_world_model_dataset.py
```

Dobot folders also retain `generation input/source_first_frame.png`, the unchanged full-panel source image. Source-frame PNGs are stored here rather than under `original/`.

The layout above applies to the ten simplified cases. For the thirteen videos without a shorter first-substep branch, `generation input/` contains the single generation reference directly: `first_frame.png`, `prompt.txt`, `metadata/metadata.json`, and `metadata/action_review.json`. Where that reference is the unchanged source, the generation video is `original/video.mp4` itself (`relative_video_path` in the metadata and manifest); no copy is kept. Simplified cases retain no additional full-video MP4, duplicate initial PNG or full-reference metadata under `generation input/`.

Use `world_model_manifest.csv` for generation benchmarking. Its relative MP4, PNG and prompt paths select `first_substep/` when present; otherwise they select the complete single action in `generation input/`. For shortened references, `full_reference_prompt_path` points to `prompt for original multi step action(reproducing the original video).txt`, which describes the full original video. For complete single-task references, it points to the selected `prompt.txt`. The complete source video remains under `original/`. The selected MP4 and PNG are the reference motion and conditioning image; `generation_prompt` and the selected `prompt.txt` contain the same instruction.

A multi-object collection is reduced to the first object placement, a multi-button sequence to the first press, and a drawer-open-plus-placement task can be reduced to opening the drawer. Each cutoff follows the completed action or complete first failed attempt and precedes the next independent subtask. Approach, grasp, lift, transport and release of one object remain together. A single continuous fold or tool-mediated sweep stays together. Every reference starts at original frame 0, so its first PNG and prompt share the original initial state.

## Dobot view and image quality

All Dobot generation MP4s, PNGs, temporary inspection images and prompts use only the **middle third-person camera**. The original triptych remains under `original/`. For 1920×480 sources, the crop is `x=640, y=0, width=640, height=480`; for the 2880×720 source, it is `x=960, y=0, width=960, height=720`. No spatial resizing is applied to generation artifacts. The prompts do not refer to the removed camera panels.

Franka uses its original single view. Complete single-task Franka references are exact copies of the original MP4s. Cropped Dobot references and shortened clips use **lossless H.264, CRF 0**, preserving decoded source pixels; these larger derivative files are distinct from the original compressed repository MP4s. Frame count and source FPS are preserved, and cuts are frame-accurate. The first PNG and the first/last video-frame RGB pixels are checked against the original source in the selected camera view. All retained generation MP4s undergo full decoding. Audio is omitted from re-encoded motion references.

## Annotation interpretation

Generation prompts follow `concise_action_v1` in the project [AGENTS.md](../AGENTS.md): one essential-action/end-state sentence plus one shared robot/camera constraint, normally 30–55 words. The initial frame supplies the scene appearance. Detailed chronology and contact observations stay in the review JSON.

New prompts have provenance `visually_authored_from_video_and_initial_frame`. This is independent of the earlier author-published/transferred/inferred task-instruction labels, which remain in the source metadata. The visible geometry, color, relative position, gripper interaction, motion sequence and terminal state guide the new prompts. Visually ambiguous identities are described by appearance. A claim that cannot be confirmed because of occlusion is recorded under `uncertainties`; technical validation does not erase that limitation.

`metadata/action_review.json` contains full `observed_sequence` stages, `initial_scene`, inspected native frame indices, dense-sheet coverage, selected frame interval, `terminal_state`, both prompts, and uncertainty notes. Inspection images are temporary; no evidence-frame directory is retained or recreated under `generation input/`. The conditioning `first_frame.png` remains in each generation reference. The manifest distinguishes `selected_execution_status` from `original_execution_status`; a successful first action may come from an overall failed source clip. The source frame interval uses a half-open range: `[source_frame_start, source_frame_end_exclusive)`.

## Rebuild and validation

```sh
# Regenerate references from the saved visual action reviews.
# Existing derivatives are reused only if their source, cut/crop and SHA-256 agree.
python3 TAVP/build_world_model_dataset.py

# Validate existing derivative files and the consolidated generation manifest.
python3 TAVP/build_world_model_dataset.py --verify-only

# Build selected reviewed IDs while work is in progress.
python3 TAVP/build_world_model_dataset.py --ids 001 002 003
```

The builder requires Python 3.9+, FFmpeg and FFprobe; no third-party Python packages. Dedicated reviewer judgments are preserved in the saved JSON rather than silently regenerated by an automatic captioner. The protocol is archived at `sources/world_model_review_protocol.txt`. The optional dense-sheet preparation helper, `world_model_tools/prepare_reviews.py`, requires Pillow and stores transient sheets outside the dataset, under `/tmp/tavp-dense-reviews/` by default. It samples every second frame without altering generation artifacts.

## Programmatic loading

```python
import csv
from pathlib import Path

root = Path("TAVP")
with (root / "world_model_manifest.csv").open(encoding="utf-8", newline="") as f:
    references = list(csv.DictReader(f))
for ref in references:
    initial_image = root / ref["relative_first_frame_path"]
    reference_video = root / ref["relative_video_path"]
    generation_instruction = ref["generation_prompt"]
```

## Completed review and validation summary

All **23** dedicated video reviews and generation references passed validation. Reviewers inspected **8,466** stride-2 frames across the complete source videos and **546** native-resolution detail frames. Preferred prompts are **31–49 words** under `concise_action_v1`. There are **10 first-subtask clips** and **13 complete single-task references**. All **15 Dobot references** use only the middle view: 14 at 640×480/20 FPS, and one at 960×720/30 FPS. All eight Franka references use 1280×720/30 FPS.

The preferred references contain **22 successful actions and one failed placement**. The original sources retain **19 success / 4 failure** labels; three failed sources have a successful first subtask. Complete source videos, full-sequence prompts and action observations remain available. Every retained generation MP4 decoded, every cut had its exact intended frame count, and initial/final retained pixel checks matched the original selected view. Source MP4 SHA-256 checks also confirmed that the downloaded originals remain unchanged.

The exact artifact checks, recorded visual ambiguities and sizes are in `world_model_validation_report.json`. The one-agent-per-video registry is in `sources/world_model_review_registry.json`.
