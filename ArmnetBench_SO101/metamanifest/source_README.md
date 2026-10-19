---
license: apache-2.0
pretty_name: ArmnetBench v0.1 (LeRobot, single-arm SO-101)
task_categories:
- robotics
tags:
- robotics
- manipulation
- so-101
- so101
- lerobot
- armnetbench
- evaluation
- teleoperation
- imitation-learning
- reinforcement-learning
size_categories:
- 1K<n<10K
---

![ArmnetBench](armnet_header.png)

<a class="flex" href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=armnet/armnetbench_v01_lerobot_so101">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>
</a>

# ArmnetBench v0.1 — LeRobot (single-arm SO-101)

ArmnetBench v0.1 contains **50 human-teleoperated reference trajectories per task**, plus
**evaluation trajectories from 7 policies** trained or fine-tuned on those reference datasets,
across **8 single-arm tasks** on the low-cost **SO-101** robot arm. Data was collected for the
**ArmnetBench v0.1** benchmark using the [Armnet arm farm](https://armnet.dev).

This repository is the **native [LeRobot](https://github.com/huggingface/lerobot) v3.0** release
(one multi-camera episode per row, state/action parquet + packed AV1 videos). A companion
**[RoboMeter](https://github.com/robometer/robometer)** export (one video per camera) lives at
[`armnet/armnetbench_v01_robometer`](https://huggingface.co/datasets/armnet/armnetbench_v01_robometer)
(subset `so101`). The bimanual embodiment is
[`armnet/armnetbench_v01_lerobot_bimanual_so101`](https://huggingface.co/datasets/armnet/armnetbench_v01_lerobot_bimanual_so101).

## At a glance

| | |
|---|---|
| Robot | SO-101 (single-arm) |
| Tasks | 8 |
| Policies | 7 (ACT, Diffusion, SmolVLA, π0, π0.5, GR00T N1.7, MolmoAct 2) + human teleoperation |
| Reference demos | 50 human-teleoperated trajectories per task |
| Cameras | `front` (576×1024), `top` (576×1024), `wrist` (720×1280) — AV1, 20 fps |
| Episodes | 2,499 |
| Frames | 1,127,881 (~15.7 h at 20 fps) |
| Labels | `successful` / `failure` / `suboptimal` per episode (915 / 1,532 / 52) |
| Format | LeRobot v3.0 + ArmnetBench eval extensions (see below) |

## Tasks

Each task has 50 human-teleoperated reference demonstrations (linked below), used to train or
fine-tune the evaluated policies. Language instructions in the dataset override the original
source task strings.

| Task ID | Instruction | Reference dataset (teleop) |
|---|---|---|
| `block_stack` | Stack the colouful blocks on top of each other | [villekuosmanen/armnetbench_block_stack](https://huggingface.co/datasets/villekuosmanen/armnetbench_block_stack) |
| `cable_clip` | Push the Displayport cable into the cable holder on the white block | [pravsels/cable_clip_remote_v2](https://huggingface.co/datasets/pravsels/cable_clip_remote_v2) |
| `cable_unclip` | Remove the power cable from cable holder | [pravsels/cable_unclip_remote](https://huggingface.co/datasets/pravsels/cable_unclip_remote) |
| `eye_drops_to_basket` | Put the eye drops into the basket | [pravsels/object_top_shelf_reset_remote](https://huggingface.co/datasets/pravsels/object_top_shelf_reset_remote) |
| `eye_drops_to_shelf` | Put the eye drops on the shelf | [pravsels/object_top_shelf_remote](https://huggingface.co/datasets/pravsels/object_top_shelf_remote) |
| `ring_insert` | Insert the colourful ring into the central wooden peg | [villekuosmanen/armnetbench_ring_insert](https://huggingface.co/datasets/villekuosmanen/armnetbench_ring_insert) |
| `tool_insert` | Insert the missing tool into the empty slot on the toolbox | [villekuosmanen/armnetbench_tool_insert](https://huggingface.co/datasets/villekuosmanen/armnetbench_tool_insert) |
| `tool_removal` | Remove the small middle tool (3/8 Extension Bar 75mm) from the toolbox | [villekuosmanen/armnetbench_tool_removal](https://huggingface.co/datasets/villekuosmanen/armnetbench_tool_removal) |

## Policies

Seven policies were trained or fine-tuned per task on the reference demonstrations, then rolled
out on the robot to produce the evaluation trajectories in this dataset.

| Policy (`policy_type`) | Family | Checkpoints by |
|---|---|---|
| `act` | Action Chunking Transformer (ACT) | [@pravsels](https://huggingface.co/pravsels) |
| `diffusion` | Diffusion Policy | [@villekuosmanen](https://huggingface.co/villekuosmanen) |
| `smolvla` | SmolVLA (vision-language-action) | [@pravsels](https://huggingface.co/pravsels) |
| `pi0` | π0 flow-matching VLA | [@lorenzouttini](https://huggingface.co/lorenzouttini) |
| `pi0.5` | π0.5 VLA | [@lorenzouttini](https://huggingface.co/lorenzouttini) |
| `grootn1.7` | NVIDIA GR00T N1.7 VLA | [@pravsels](https://huggingface.co/pravsels) |
| `molmoact2` | MolmoAct 2 VLA | [@pravsels](https://huggingface.co/pravsels) |

| Task | ACT | Diffusion | SmolVLA | π0 | π0.5 | GR00T N1.7 | MolmoAct 2 |
|---|---|---|---|---|---|---|---|
| `block_stack` | [ckpt](https://huggingface.co/pravsels/act_block_stack_20k) | [ckpt](https://huggingface.co/villekuosmanen/armnetbench_block_stack_diffusion) | [ckpt](https://huggingface.co/pravsels/smolvla_block_stack) | [ckpt](https://huggingface.co/lorenzouttini/pi0-so101-armnetbench-block-stack-isambard-v50) | [ckpt](https://huggingface.co/lorenzouttini/pi05-so101-armnetbench-block-stack-isambard-v50) | [ckpt](https://huggingface.co/pravsels/groot1.7_block_stack_20k) | [ckpt](https://huggingface.co/pravsels/molmoact2_block_stack_quantile_norm_fix_25k) |
| `cable_clip` | [ckpt](https://huggingface.co/pravsels/act_cable_clip_20k) | [ckpt](https://huggingface.co/villekuosmanen/cable_clip_remote_v2_diffusion) | [ckpt](https://huggingface.co/pravsels/smolvla_cable_clip) | [ckpt](https://huggingface.co/lorenzouttini/pi0-so101-cable-clip-isambard-v50) | [ckpt](https://huggingface.co/lorenzouttini/pi05-so101-cable-clip-isambard) | [ckpt](https://huggingface.co/pravsels/groot1.7_cable_clip_20k) | [ckpt](https://huggingface.co/pravsels/molmoact2_cable_clip_quantile_norm_fix_25k) |
| `cable_unclip` | [ckpt](https://huggingface.co/pravsels/act_cable_unclip_20k) | [ckpt](https://huggingface.co/villekuosmanen/cable_unclip_remote_diffusion) | [ckpt](https://huggingface.co/pravsels/smolvla_cable_unclip) | [ckpt](https://huggingface.co/lorenzouttini/pi0-so101-cable-unclip-isambard-v50) | [ckpt](https://huggingface.co/lorenzouttini/pi05-so101-cable-unclip-isambard) | [ckpt](https://huggingface.co/pravsels/groot1.7_cable_unclip_20k) | [ckpt](https://huggingface.co/pravsels/molmoact2_cable_unclip_quantile_norm_fix_25k) |
| `eye_drops_to_basket` | [ckpt](https://huggingface.co/pravsels/act_eyedrops_basket_20k) | [ckpt](https://huggingface.co/villekuosmanen/object_top_shelf_reset_remote_diffusion) | [ckpt](https://huggingface.co/pravsels/smolvla_eyedrops_basket) | [ckpt](https://huggingface.co/lorenzouttini/pi0-so101-object-top-shelf-reset-isambard-v50) | [ckpt](https://huggingface.co/lorenzouttini/pi05-so101-object-top-shelf-reset-isambard-v50) | [ckpt](https://huggingface.co/pravsels/groot1.7_eyedrops_basket_20k) | [ckpt](https://huggingface.co/pravsels/molmoact2_eyedrops_basket_quantile_norm_fix_25k) |
| `eye_drops_to_shelf` | [ckpt](https://huggingface.co/pravsels/act_eyedrops_shelf_20k) | [ckpt](https://huggingface.co/villekuosmanen/object_top_shelf_remote_diffusion) | [ckpt](https://huggingface.co/pravsels/smolvla_eyedrops_shelf) | [ckpt](https://huggingface.co/lorenzouttini/pi0-so101-object-top-shelf-isambard-v50) | [ckpt](https://huggingface.co/lorenzouttini/pi05-so101-object-top-shelf-isambard) | [ckpt](https://huggingface.co/pravsels/groot1.7_eyedrops_shelf_20k) | [ckpt](https://huggingface.co/pravsels/molmoact2_eyedrops_shelf_quantile_norm_fix_25k) |
| `ring_insert` | [ckpt](https://huggingface.co/pravsels/act_ring_insert_20k) | [ckpt](https://huggingface.co/villekuosmanen/armnetbench_ring_insert_diffusion) | [ckpt](https://huggingface.co/pravsels/smolvla_ring_insert) | [ckpt](https://huggingface.co/lorenzouttini/pi0-so101-armnetbench-ring-insert-isambard-v50) | [ckpt](https://huggingface.co/lorenzouttini/pi05-so101-armnetbench-ring-insert-isambard-v50) | [ckpt](https://huggingface.co/pravsels/groot1.7_ring_insert_20k) | [ckpt](https://huggingface.co/pravsels/molmoact2_ring_insert_quantile_norm_fix_25k) |
| `tool_insert` | [ckpt](https://huggingface.co/pravsels/act_tool_insert_20k) | [ckpt](https://huggingface.co/villekuosmanen/armnetbench_tool_insert_diffusion) | [ckpt](https://huggingface.co/pravsels/smolvla_tool_insert) | [ckpt](https://huggingface.co/lorenzouttini/pi0-so101-armnetbench-tool-insert-isambard-v50) | [ckpt](https://huggingface.co/lorenzouttini/pi05-so101-armnetbench-tool-insert-isambard-v50) | [ckpt](https://huggingface.co/pravsels/groot1.7_tool_insert_20k) | [ckpt](https://huggingface.co/pravsels/molmoact2_tool_insert_quantile_norm_fix_25k) |
| `tool_removal` | [ckpt](https://huggingface.co/pravsels/act_tool_removal_20k) | [ckpt](https://huggingface.co/villekuosmanen/armnetbench_tool_removal_diffusion) | [ckpt](https://huggingface.co/pravsels/smolvla_tool_removal) | [ckpt](https://huggingface.co/lorenzouttini/pi0-so101-armnetbench-tool-removal-isambard-v50) | [ckpt](https://huggingface.co/lorenzouttini/pi05-so101-armnetbench-tool-removal-isambard-v50) | [ckpt](https://huggingface.co/pravsels/groot1.7_tool_removal_20k) | [ckpt](https://huggingface.co/pravsels/molmoact2_tool_removal_quantile_norm_fix_25k) |

## Data format

This dataset follows **LeRobot v3.0** layout (`meta/`, `data/*.parquet`, `videos/`). One
**episode** = one robot rollout with all cameras, proprioception, and actions synchronised at
20 fps.

### Standard LeRobot features

| Feature | Type | Description |
|---|---|---|
| `observation.images.front` | video 576×1024 | Front camera (AV1) |
| `observation.images.top` | video 576×1024 | Overhead camera (AV1) |
| `observation.images.wrist` | video 720×1280 | Wrist camera (AV1; cell-3 1080p sources downscaled) |
| `observation.state` | float32[6] | Joint positions (shoulder_pan … gripper) |
| `action` | float32[6] | Commanded joint positions |
| `task` | string | Language instruction (from `meta/tasks`, overridden per ArmnetBench task) |

### ArmnetBench extensions (beyond stock LeRobot)

We add **per-frame** and **per-episode** fields so eval rollouts carry success/failure labels
usable for reward modelling, filtering, or offline RL — without leaving the LeRobot loader.

**Per-frame** (in `data/*.parquet`):

| Feature | Type | Description |
|---|---|---|
| `next.reward` | float32[1] | Sparse terminal reward: `1.0` on the **final kept frame** iff the episode is a strict success (`success_class == successful`), else `0.0` |
| `next.done` | bool[1] | `True` on the final kept frame of each episode |

**Per-episode** (in `meta/episodes/*.parquet`):

| Column | Type | Description |
|---|---|---|
| `success` | int (0/1) | Binary success flag. `1` only for strict successes; **suboptimal completions are `0`** |
| `success_class` | string | Three-way label: `successful`, `failure`, or `suboptimal` |
| `policy_repo_id` | string | Hugging Face model repo for the evaluated policy (empty for `teleoperated`) |
| `policy_type` | string | Policy family: `act`, `diffusion`, `smolvla`, `pi0`, `pi0.5`, `grootn1.7`, `molmoact2`, or `teleoperated` |

**Processing notes:**

- **Teleoperation** episodes are always `success_class=successful`, `success=1`.
- **Policy rollouts** are labelled from human evaluation (`successful` / `failure` / `suboptimal`).
- Episodes with a labelled **`success_cutoff_time`** (successful or suboptimal) are **end-trimmed**
  so trailing idle frames after task completion are removed.
- Source episodes that were **corrupted or unlabelled** in a multi-episode recording are dropped
  before merge; only labelled good episodes are kept.
- Task strings are taken from the ArmnetBench canonical instruction set, not the original source
  dataset `tasks.jsonl`.

## Versions / tags

| Tag | Contents |
|---|---|
| `sample` | Small sample (subset of tasks/policies) for quick inspection |
| `preview` | Early preview snapshot (superseded) |
| `v1.0` / `main` | Full finalised release — **current release** |

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("armnet/armnetbench_v01_lerobot_so101", revision="v1.0")
print(ds.meta.total_episodes, ds.meta.total_frames)

# episode-level labels
import pandas as pd
from pathlib import Path
ep = pd.concat(pd.read_parquet(p) for p in Path(ds.root).glob("meta/episodes/*/*.parquet"))
print(ep[["episode_index", "success", "success_class", "policy_type"]].head())
```

## Citation / links

- Benchmark: **ArmnetBench v0.1** · Arm farm: [armnet.dev](https://armnet.dev)
- LeRobot tooling: [huggingface/lerobot](https://github.com/huggingface/lerobot)
- RoboMeter export (per-camera videos): [armnetbench_v01_robometer](https://huggingface.co/datasets/armnet/armnetbench_v01_robometer)
