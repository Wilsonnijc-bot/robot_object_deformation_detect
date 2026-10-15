# Robot video data branch

This branch contains the Robotvideo dataset project: first-frame-conditioned world-model
generation inputs (first frame, action prompt, reference clip) from six robot datasets, with
source annotations, visual reviews, quality-gate scores and preparation scripts.
It has an independent history on `data/robotvideo` in
`Wilsonnijc-bot/robot_object_deformation_detect`.

**Dataset viewer:** https://wilsonnijc-bot.github.io/robot_object_deformation_detect/robotvideo/
The viewer shows the first 15 NVIDIA GR1 videos (`SITE_LIMIT` in `build_site.py`); all 92 are in the repository.

| Dataset | Episodes | Robot · view | Manifest |
|---|---:|---|---|
| [NVIDIA_GR1_100](NVIDIA_GR1_100/) | 92 | Fourier GR1 humanoid · first-person | `metamanifest/world_model_prompts.csv` |
| [Humanoid_Everyday](Humanoid_Everyday/) | 26 | Humanoid · third-person | `metamanifest/world_model_prompts.csv` |
| [ActionNet](ActionNet/) | 15 | Fourier humanoid · first-person | `metamanifest/world_model_prompts.csv` |
| [RoboMIND](RoboMIND/) | 12 | Franka (right camera) · Tien Kung humanoid | `metamanifest/world_model_prompts.csv` |
| [TAVP](TAVP/) | 8 | Franka Research 3 · third-person | `world_model_manifest.csv` |
| [RoboCOIN_Airbot_MMK2_stack_block](RoboCOIN_Airbot_MMK2_stack_block/) | 3 | Dual-arm Airbot MMK2 · first-person | `metamanifest/world_model_prompts.csv` |

Each dataset README documents its source, license, selection and validation. Prompt
protocols are in [AGENTS.md](AGENTS.md).

## Quality gate

Every input is scored on the sharpness of its final upscaled conditioning frame:
third-person ≥ 120, first-person ≥ 60. Inputs below threshold are kept, not deleted: RoboCOIN,
RoboMIND Franka and two Humanoid Everyday inputs are documented exceptions;
filter on `gate_result` (`pass`, `exception`, `fail`) in the manifests or in
[quality_gate.csv](quality_gate.csv). See [data_quality_gate.md](data_quality_gate.md).

```sh
.venv/bin/python quality_gate.py    # rescore inputs, update metadata and manifests
.venv/bin/python build_site.py      # rebuild the viewer in site/
```

## Layout

Each video folder holds `original/` (unchanged source media, publisher annotation and
`metadata/metadata.json`) and `generation input/` (`first_frame.png`, the action prompt,
`metadata/{metadata,action_review}.json`, and `video.mp4` only when the reference is a derived
clip). When the generation video is the unchanged source, metadata and manifests point to the
file in `original/` instead of keeping a copy. Humanoid Everyday nests these under
`third_person/`. TAVP shortened references keep their selected prompt in
`generation input/first_substep/prompt.txt`, with the full-video prompt named
`prompt for original multi step action(reproducing the original video).txt` in the parent
folder. Paths inside per-video metadata resolve relative to the JSON's directory.

## Getting the data

Videos, PNGs, HDF5 and MKV files use Git LFS. The viewer's previews in `site/` are small
regular files so GitHub Pages can serve them.

```sh
git lfs install
git clone --branch data/robotvideo --single-branch https://github.com/Wilsonnijc-bot/robot_object_deformation_detect.git Robotvideo
cd Robotvideo
git lfs pull
```

The local Python environment, caches, incomplete downloads and credentials are excluded.
Licenses differ by dataset (CC BY 4.0, CC BY-NC-SA 4.0, Apache-2.0, or not stated); check each
dataset README before reuse.
