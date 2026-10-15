---
task_categories:
  - robotics

language:
  - en


extra_gated_prompt: 'By accessing this dataset, you agree to cite the associated paper in your research/publications—see the "Citation" section for details. You agree to not use the dataset to conduct experiments that cause harm to human subjects.'



extra_gated_fields:

  Company/Organization:
    type: 'text'
    description: 'e.g., "ETH Zurich", "Boston Dynamics", "Independent Researcher"'

  Country:
    type: 'country'
    description: 'e.g., "Germany", "China", "United States"'



tags:
  - RoboCOIN
  - LeRobot

license: apache-2.0

configs:
- config_name: default
  data_files: data/chunk-{id}/episode_{id}.parquet
---

# Airbot_MMK2_stack_block

## Dataset Description

This dataset uses an extended format based on LeRobot and is fully compatible with LeRobot.

## Task Preview

<video src="videos/chunk-000/observation.images.cam_front_rgb/episode_000000.mp4" controls width="640"></video>

[View Video Directly](videos/chunk-000/observation.images.cam_front_rgb/episode_000000.mp4)

### Overview

- **Total Episodes:** 236
- **Total Frames:** 60925
- **FPS:** 30
- **Dataset Size:** 2.00 GB
- **Robot Name:** `Airbot_MMK2`
- **End-Effector Type:** `five_finger_gripper`
- **Teleoperation Type:** `Due to some reasons, this dataset temporarily cannot provide the teleoperation type information.`
- **Sensors:** `cam_head_rgb`, 
`cam_left_wrist_rgb`, 
`cam_right_wrist_rgb`, 
`cam_front_rgb`

- **Camera Information:** cam_head_rgb; 
cam_left_wrist_rgb; 
cam_right_wrist_rgb; 
cam_front_rgb

- **Scene:** `household->bedroom`
- **Objects:** `square_building_blocks(unknown)`, 
`block_pillar(unknown)`

- **Task Description:** place the square building blocks in the center with left hand and put the cylindrical building blocks on top with right hand.


### Primary Task Instruction
> place the square building blocks in the center with left hand and put the cylindrical building blocks on top with right hand.

### Robot Configuration

- **Robot Name:** `Airbot_MMK2`
- **Codebase Version:** `v2.1`
- **End-Effector Type:** `five_finger_gripper`
- **Teleoperation Type:** `Due to some reasons, this dataset temporarily cannot provide the teleoperation type information.`

## Scene and Objects

### Scene Type
`household->bedroom`

### Objects
- `square_building_blocks(unknown)`
- `block_pillar(unknown)`


## Task Descriptions

- **Standardized Task Description:** `place the square building blocks in the center with left hand and put the cylindrical building blocks on top with right hand.`
- **Operation Type:** `Due to some reasons, this dataset temporarily cannot provide the operation type information.`

- **Environment Type:** `Due to some reasons, this dataset temporarily cannot provide the environment type information.`

### Sub-Tasks
This dataset includes 20 distinct subtasks:

1. **Grasp the blue build blocks with the left gripper** (Index: 0)
2. **Grasp the green build blocks with the right gripper** (Index: 1)
3. **Place the blue build blocks on the Mini table with the left gripper** (Index: 2)
4. **Place the green build blocks on the blue build block with the right gripper** (Index: 3)
5. **Grasp the blue diamond shaped build blocks with the right gripper** (Index: 4)
6. **Grasp the red build blocks with the right gripper** (Index: 5)
7. **Place the blue diamond shaped build blocks on the glasses case with the right gripper** (Index: 6)
8. **Place the purple build blocks on the red build block with the right gripper** (Index: 7)
9. **Place the green build blocks on the center of the table with the left gripper** (Index: 8)
10. **Grasp the green build blocks with the left gripper** (Index: 9)
11. **Place the glasses case on the red and green build blocks with the left gripper** (Index: 10)
12. **Grasp the purple build blocks with the right gripper** (Index: 11)
13. **Place the blue build blocks on the yellow build block with the left gripper** (Index: 12)
14. **Grasp the glasses case with the left gripper** (Index: 13)
15. **Place the green cylindrical build blocks on the green build blocks with the right gripper** (Index: 14)
16. **End** (Index: 15)
17. **Place the blue build blocks on the pink build block with the left gripper** (Index: 16)
18. **Place the red build blocks on the blue build block with the right gripper** (Index: 17)
19. **Grasp the green cylindrical build blocks with the right gripper** (Index: 18)
20. **null** (Index: 19)


### Atomic Actions
- `grasp`
- `pick`
- `place`


## Hardware and Sensors


### Sensors

- `cam_head_rgb`

- `cam_left_wrist_rgb`

- `cam_right_wrist_rgb`

- `cam_front_rgb`




### Camera Information


- `cam_head_rgb`: dtype=video, shape=480x640x3, resolution=640x480, codec=av1, pix_fmt=yuv420p

- `cam_left_wrist_rgb`: dtype=video, shape=480x640x3, resolution=640x480, codec=av1, pix_fmt=yuv420p

- `cam_right_wrist_rgb`: dtype=video, shape=480x640x3, resolution=640x480, codec=av1, pix_fmt=yuv420p

- `cam_front_rgb`: dtype=video, shape=480x640x3, resolution=640x480, codec=av1, pix_fmt=yuv420p




### Coordinate System
- **Definition:** `right-hand-frame`


### Dimensions & Units
- **Joint Rotation:** `radian`
- **End-Effector Rotation:** `end_rotation_dim`
- **End-Effector Translation:** `end_translation_dim`




## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Episodes** | 236 |
| **Total Frames** | 60925 |
| **Total Tasks** | 20 |
| **Total Videos** | 944 |
| **Total Chunks** | 1 |
| **Chunk Size** | 1000 |
| **FPS** | 30 |
| **State Dimensions** | 36 |
| **Action Dimensions** | 36 |
| **Camera Views** | 4 |
| **Dataset Size** | 2.00 GB |


## Data Splits

The dataset is organized into the following splits:

- **Training**: Episodes 0:235


## Dataset Structure

This dataset follows the LeRobot format and contains the following components:

### Data Files
- **Videos**: Compressed video files containing RGB camera observations
- **State Data**: Robot joint positions, velocities, and other state information
- **Action Data**: Robot action commands and trajectories
- **Metadata**: Episode metadata, timestamps, and annotations

### File Organization
- **Data Path Pattern**: `data/chunk-{id}/episode_{id}.parquet`
- **Video Path Pattern**: `videos/chunk-{id}/observation.images.cam_left_wrist_rgb/episode_{id}.mp{id}`
- **Chunking**: Data is organized into 1 chunk(s)
of size 1000

### Data Structure (Tree)

```
Airbot_MMK2_stack_block_qced_hardlink/
|-- annotations
|   |-- eef_acc_mag_annotation.jsonl
|   |-- eef_direction_annotation.jsonl
|   |-- eef_velocity_annotation.jsonl
|   |-- gripper_activity_annotation.jsonl
|   |-- gripper_mode_annotation.jsonl
|   |-- scene_annotations.jsonl
|   `-- subtask_annotations.jsonl
|-- data
|   `-- chunk-000
|       |-- episode_000000.parquet
|       |-- episode_000001.parquet
|       |-- episode_000002.parquet
|       |-- episode_000003.parquet
|       |-- episode_000004.parquet
|       |-- episode_000005.parquet
|       |-- episode_000006.parquet
|       |-- episode_000007.parquet
|       |-- episode_000008.parquet
|       |-- episode_000009.parquet
|       |-- episode_000010.parquet
|       `-- episode_000011.parquet
|       `-- ... (224 more entries)
|-- meta
|   |-- episodes.jsonl
|   |-- episodes_stats.jsonl
|   |-- info.json
|   `-- tasks.jsonl
`-- videos
    `-- chunk-000
        |-- observation.images.cam_front_rgb
        |-- observation.images.cam_head_rgb
        |-- observation.images.cam_left_wrist_rgb
        `-- observation.images.cam_right_wrist_rgb
```

## Camera Views






This dataset includes 4 camera views: `cam_head_rgb`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb`, `cam_front_rgb`.


## Features (Full YAML)

```yaml
observation.images.cam_head_rgb:
  dtype: video
  shape:
  - 480
  - 640
  - 3
  names:
  - height
  - width
  - channels
  info:
    video.height: 480
    video.width: 640
    video.codec: av1
    video.pix_fmt: yuv420p
    video.is_depth_map: false
    video.fps: 30
    video.channels: 3
    has_audio: false
observation.images.cam_left_wrist_rgb:
  dtype: video
  shape:
  - 480
  - 640
  - 3
  names:
  - height
  - width
  - channels
  info:
    video.height: 480
    video.width: 640
    video.codec: av1
    video.pix_fmt: yuv420p
    video.is_depth_map: false
    video.fps: 30
    video.channels: 3
    has_audio: false
observation.images.cam_right_wrist_rgb:
  dtype: video
  shape:
  - 480
  - 640
  - 3
  names:
  - height
  - width
  - channels
  info:
    video.height: 480
    video.width: 640
    video.codec: av1
    video.pix_fmt: yuv420p
    video.is_depth_map: false
    video.fps: 30
    video.channels: 3
    has_audio: false
observation.images.cam_front_rgb:
  dtype: video
  shape:
  - 480
  - 640
  - 3
  names:
  - height
  - width
  - channels
  info:
    video.height: 480
    video.width: 640
    video.codec: av1
    video.pix_fmt: yuv420p
    video.is_depth_map: false
    video.fps: 30
    video.channels: 3
    has_audio: false
observation.state:
  dtype: float32
  shape:
  - 36
  names:
  - left_arm_joint_1_rad
  - left_arm_joint_2_rad
  - left_arm_joint_3_rad
  - left_arm_joint_4_rad
  - left_arm_joint_5_rad
  - left_arm_joint_6_rad
  - right_arm_joint_1_rad
  - right_arm_joint_2_rad
  - right_arm_joint_3_rad
  - right_arm_joint_4_rad
  - right_arm_joint_5_rad
  - right_arm_joint_6_rad
  - left_hand_joint_1_rad
  - left_hand_joint_2_rad
  - left_hand_joint_3_rad
  - left_hand_joint_4_rad
  - left_hand_joint_5_rad
  - left_hand_joint_6_rad
  - left_hand_joint_7_rad
  - left_hand_joint_8_rad
  - left_hand_joint_9_rad
  - left_hand_joint_10_rad
  - left_hand_joint_11_rad
  - left_hand_joint_12_rad
  - right_hand_joint_1_rad
  - right_hand_joint_2_rad
  - right_hand_joint_3_rad
  - right_hand_joint_4_rad
  - right_hand_joint_5_rad
  - right_hand_joint_6_rad
  - right_hand_joint_7_rad
  - right_hand_joint_8_rad
  - right_hand_joint_9_rad
  - right_hand_joint_10_rad
  - right_hand_joint_11_rad
  - right_hand_joint_12_rad
action:
  dtype: float32
  shape:
  - 36
  names:
  - left_arm_joint_1_rad
  - left_arm_joint_2_rad
  - left_arm_joint_3_rad
  - left_arm_joint_4_rad
  - left_arm_joint_5_rad
  - left_arm_joint_6_rad
  - right_arm_joint_1_rad
  - right_arm_joint_2_rad
  - right_arm_joint_3_rad
  - right_arm_joint_4_rad
  - right_arm_joint_5_rad
  - right_arm_joint_6_rad
  - left_hand_joint_1_rad
  - left_hand_joint_2_rad
  - left_hand_joint_3_rad
  - left_hand_joint_4_rad
  - left_hand_joint_5_rad
  - left_hand_joint_6_rad
  - left_hand_joint_7_rad
  - left_hand_joint_8_rad
  - left_hand_joint_9_rad
  - left_hand_joint_10_rad
  - left_hand_joint_11_rad
  - left_hand_joint_12_rad
  - right_hand_joint_1_rad
  - right_hand_joint_2_rad
  - right_hand_joint_3_rad
  - right_hand_joint_4_rad
  - right_hand_joint_5_rad
  - right_hand_joint_6_rad
  - right_hand_joint_7_rad
  - right_hand_joint_8_rad
  - right_hand_joint_9_rad
  - right_hand_joint_10_rad
  - right_hand_joint_11_rad
  - right_hand_joint_12_rad
timestamp:
  dtype: float32
  shape:
  - 1
  names: null
frame_index:
  dtype: int64
  shape:
  - 1
  names: null
episode_index:
  dtype: int64
  shape:
  - 1
  names: null
index:
  dtype: int64
  shape:
  - 1
  names: null
task_index:
  dtype: int64
  shape:
  - 1
  names: null
subtask_annotation:
  names: null
  shape:
  - 5
  dtype: int32
scene_annotation:
  names: null
  shape:
  - 1
  dtype: int32
eef_sim_pose_state:
  names:
  - left_eef_pos_x
  - left_eef_pos_y
  - left_eef_pos_z
  - left_eef_rot_x
  - left_eef_rot_y
  - left_eef_rot_z
  - right_eef_pos_x
  - right_eef_pos_y
  - right_eef_pos_z
  - right_eef_rot_x
  - right_eef_rot_y
  - right_eef_rot_z
  shape:
  - 12
  dtype: float32
eef_sim_pose_action:
  names:
  - left_eef_pos_x
  - left_eef_pos_y
  - left_eef_pos_z
  - left_eef_rot_x
  - left_eef_rot_y
  - left_eef_rot_z
  - right_eef_pos_x
  - right_eef_pos_y
  - right_eef_pos_z
  - right_eef_rot_x
  - right_eef_rot_y
  - right_eef_rot_z
  shape:
  - 12
  dtype: float32
eef_direction_state:
  names:
  - left_eef_direction
  - right_eef_direction
  shape:
  - 2
  dtype: int32
eef_direction_action:
  names:
  - left_eef_direction
  - right_eef_direction
  shape:
  - 2
  dtype: int32
eef_velocity_state:
  names:
  - left_eef_velocity
  - right_eef_velocity
  shape:
  - 2
  dtype: int32
eef_velocity_action:
  names:
  - left_eef_velocity
  - right_eef_velocity
  shape:
  - 2
  dtype: int32
eef_acc_mag_state:
  names:
  - left_eef_acc_mag
  - right_eef_acc_mag
  shape:
  - 2
  dtype: int32
eef_acc_mag_action:
  names:
  - left_eef_acc_mag
  - right_eef_acc_mag
  shape:
  - 2
  dtype: int32

```

## Available Annotations

This dataset includes rich annotations to support diverse learning approaches:

- `eef_acc_mag_annotation.jsonl`
- `eef_direction_annotation.jsonl`
- `eef_velocity_annotation.jsonl`
- `gripper_activity_annotation.jsonl`
- `gripper_mode_annotation.jsonl`
- `scene_annotations.jsonl`
- `subtask_annotations.jsonl`


## Dataset Tags

- `RoboCOIN`
- `LeRobot`


## Authors

### Contributors
This dataset is contributed by:-RoboCOIN Team at Beijing Academy of Artificial Intelligence (BAAI)

### Annotators
No annotator information available.

## Links

- **Homepage:** [https://flagopen.github.io/RoboCOIN/](https://flagopen.github.io/RoboCOIN/)
- **Paper:** [https://arxiv.org/abs/2511.17441](https://arxiv.org/abs/2511.17441)
- **Repository:** [https://github.com/FlagOpen/RoboCOIN](https://github.com/FlagOpen/RoboCOIN)
## Contact and Support

For questions, issues, or feedback regarding this dataset, please contact us.
### Support
For technical support, please open an issue on our GitHub repository.

## License

apache-2.0

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{robocoin,
  title={RoboCOIN: An Open-Sourced Bimanual Robotic Data Collection for Integrated Manipulation},
  author={Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, Zhaoye Long, Yue Wang, Chong Liu, Dihan Wang, Ziqiang Ni, Xiang Yang, You Liu, Ruoxuan Feng, Runtian Xu, Lei Zhang, Denghang Huang, Chenghao Jin, Anlan Yin, Xinlong Wang, Zhenguo Sun, Junkai Zhao, Mengfei Du, Mingyu Cao, Xiansheng Chen, Hongyang Cheng, Xiaojie Zhang, Yankai Fu, Ning Chen, Cheng Chi, Sixiang Chen, Huaihai Lyu, Xiaoshuai Hao, Yequan Wang, Bo Lei, Dong Liu, Xi Yang, Yance Jiao, Tengfei Pan, Yunyan Zhang, Songjing Wang, Ziqian Zhang, Xu Liu, Ji Zhang, Caowei Meng, Zhizheng Zhang, Jiyang Gao, Song Wang, Xiaokun Leng, Zhiqiang Xie, Zhenzhen Zhou, Peng Huang, Wu Yang, Yandong Guo, Yichao Zhu, Suibing Zheng, Hao Cheng, Xinmin Ding, Yang Yue, Huanqian Wang, Chi Chen, Jingrui Pang, YuXi Qian, Haoran Geng, Lianli Gao, Haiyuan Li, Bin Fang, Gao Huang, Yaodong Yang, Hao Dong, He Wang, Hang Zhao, Yadong Mu, Di Hu, Hao Zhao, Tiejun Huang, Shanghang Zhang, Yonghua Lin, Zhongyuan Wang and Guocai Yao},
  journal={arXiv preprint arXiv:2511.17441},
  url = {https://arxiv.org/abs/2511.17441},
  year={2025},
  }

```

### Additional References

If you use this dataset, please also consider citing:
LeRobot Framework: https://github.com/huggingface/lerobot


## Version Information

Initial Release
