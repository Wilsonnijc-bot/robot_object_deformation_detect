# Data quality gate

An input (first frame + prompt) is eligible for world-model generation if the **final upscaled conditioning frame** is sharp enough. The gate measures the image the generator actually receives, not the source resolution.

| View | Pass if sharpness ≥ | Roughly |
|---|---:|---|
| Third-person | 120 | a sharp frame enlarged ~1.4× |
| First-person | 60 | a sharp frame enlarged ~2× |

Run `.venv/bin/python quality_gate.py` after adding or changing inputs. It writes `quality_gate.csv`, the `data_quality_gate` block in each input's `metadata/metadata.json`, and the `gate_view`, `gate_sharpness`, `gate_threshold` and `gate_result` columns in every generation manifest.

## How sharpness is measured

1. Take the conditioning PNG actually fed to the model: `generation input/first_frame.png`, or `generation input/first_substep/first_frame.png` where it exists.
2. Cover-resize it with Lanczos to the nearest-aspect 720p-class generation size (1280 × 704, 960 × 704, 832 × 832, 704 × 960 or 704 × 1280) and center-crop. This is the "model view"; a 640 × 480 frame is enlarged 1.5×.
3. Sharpness = variance of the 4-neighbour Laplacian of the model view's luma (0–255, reflect padding).

This is the same measurement as the `Sharpness (model view)` column of `deformationdetection/documentation/reports/robotvideo_input_quality.md`; `quality_gate.py` reproduces its values exactly. Models with a 480p native size (ABot-PhysWorld, LVP) enlarge less, so passing at 720p is the stricter case.

**View** is the camera's relation to the robot. Third-person: TAVP, Humanoid Everyday third-person, RoboMIND Franka `camera_right`. First-person (robot head or egocentric camera): NVIDIA GR1, ActionNet `top`, RoboCOIN `cam_head_rgb`, RoboMIND Tien Kung `camera_top`.

## Limits of the metric

Laplacian variance rewards high-contrast edges and texture as well as focus. A sharp frame of a plain wall or empty table scores low, and a busy scene scores high. Three EWMBench episodes from one camera at one resolution scored 64.1 (tiled wall, ice maker), 47.2 (plain plate and counter) and 19.0 (smooth marble wall, slightly soft). Treat a score near the threshold as a prompt to look at the frame at 1:1, not as a verdict.

The first-person threshold was set to 60 (from the proposed 70) on 2026-10-15 after reviewing those EWMBench frames.

## Exceptions

Exceptions are admitted by visual review and keep their measured score; `gate_result` is `exception`.

| Dataset | Inputs | Sharpness | Reason |
|---|---:|---|---|
| RoboCOIN_Airbot_MMK2_stack_block | 3 | 12.3–19.8 | Hand and block edges judged acceptable at the model view; the score is driven mostly by the large untextured table. |

## Failing inputs are kept and flagged

Failing inputs are **not** deleted. They remain in place with `gate_result = fail` so they can be reviewed or used deliberately. Filter on `gate_result` in the manifests or `quality_gate.csv` before generation.

## Current classification (156 inputs, 2026-10-15)

| Dataset | Source size | View | n | Model view | Upscale | Sharpness min / median / max | Result |
|---|---|---|---:|---|---:|---|---|
| NVIDIA_GR1_100 | 1280 × 800 | first | 92 | 1280 × 704 | 1.00 | 519 / 696 / 1144 | 92 pass |
| ActionNet | 1280 × 800 | first | 15 | 1280 × 704 | 1.00 | 364 / 511 / 889 | 15 pass |
| RoboMIND Tien Kung | 640 × 480 | first | 6 | 960 × 704 | 1.50 | 90 / 150 / 251 | 6 pass |
| RoboCOIN Airbot MMK2 | 640 × 480 | first | 3 | 960 × 704 | 1.50 | 12 / 17 / 20 | 3 exception |
| TAVP Franka | 1280 × 720 | third | 8 | 1280 × 704 | 1.00 | 132 / 160 / 182 | 8 pass |
| Humanoid Everyday | 720 × 1280 | third | 18 | 704 × 1280 | 1.00 | 99 / 183 / 523 | 16 pass, 2 fail |
| Humanoid Everyday | 592 × 1280 | third | 1 | 704 × 1280 | 1.19 | 128 | 1 pass (human-robot-7) |
| Humanoid Everyday | 1440 × 1920 / 1920 × 1440 | third | 7 | 704 × 960 / 960 × 704 | 0.50 | 363 / 417 / 1296 | 7 pass |
| RoboMIND Franka | 640 × 480 | third | 6 | 960 × 704 | 1.50 | 15 / 25 / 27 | 6 fail |

Summary: 145 pass, 3 exception, 8 fail.

Failing inputs: Humanoid_Everyday `human-robot-2__hand_over_toilet_paper` (99.1), `tool-use-6__wipe_table_with_cloth` (117.3); all six RoboMIND Franka episodes (14.7–27.3).

## History

Before 2026-10-15 the gate was a hard resolution rule: an input was removed if its first frame needed more than 1.2× upscaling for any model. That rule removed 15 TAVP Dobot folders and 24 Humanoid Everyday folders. Under the sharpness gate, `human-robot-7__shake_hand_with_person` (592 × 1280, 127.9) passes and was restored. The other 38 removed folders all score below 120 and stay removed; they remain recoverable from git history (commit `d2ec7dd`). Their scores are in `deformationdetection/documentation/reports/robotvideo_input_quality.md`.

The same day, DROID_raw, Galaxea_Open_World, EWMBench and the RoboMIND AgileX episodes were removed from the project, and RoboMIND Franka episodes were reduced to the right RGB camera.
