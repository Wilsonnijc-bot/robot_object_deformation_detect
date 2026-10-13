---
license: apache-2.0
task_categories:
- robotics
language:
- en
size_categories:
- 1K<n<10K
---

# Humanoid Everyday
A Comprehensive Robotic Dataset for Open-World Humanoid Manipulation

---

## Overview
**Humanoid Everyday** is a large-scale, diverse humanoid manipulation dataset designed for open-world robotic learning and embodied intelligence.

It contains over 260 tasks across 7 major categories, covering dexterous manipulation, human–humanoid interaction, and locomotion-integrated activities.  
All data were collected through a human-supervised teleoperation pipeline, recording multimodal sensory streams at 30 Hz (RGB, depth, LiDAR, tactile, IMU).

---
## Dataset Access
### LeRobot v2.0
This dataset is compliant with the LeRobot v2.0 format. You can load it as follows:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

ds_meta = LeRobotDatasetMetadata("USC-GVL/humanoid-everyday")

ds = LeRobotDataset(
    "USC-GVL/humanoid-everyday",
    tolerance_s=1e-2,
)
```

### Legacy
You can also explore our dataset through the public spreadsheet below:

**[Humanoid Everyday Dataset Spreadsheet](https://docs.google.com/spreadsheets/d/158Wzf8Xywky3aHJSCfp3OZxf4bkhzAJdcG94eHf8gVc/edit?gid=1307250382#gid=1307250382)**

The sheet includes:
- A full list of 260 humanoid tasks with labeled categorys and robot types. 
- Download URLs for each task  

A manual dataloader is provided at [github.com/ausbxuse/Humanoid-Everyday](https://github.com/ausbxuse/Humanoid-Everyday)

## Citation
```
@misc{zhao2025humanoideverydaycomprehensiverobotic,
      title={Humanoid Everyday: A Comprehensive Robotic Dataset for Open-World Humanoid Manipulation}, 
      author={Zhenyu Zhao and Hongyi Jing and Xiawei Liu and Jiageng Mao and Abha Jha and Hanwen Yang and Rong Xue and Sergey Zakharov and Vitor Guizilini and Yue Wang},
      year={2025},
      eprint={2510.08807},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.08807}, 
}
```
