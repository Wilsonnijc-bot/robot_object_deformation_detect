# Robotvideo world-model inputs: inventory and initial-image pixel quality

Source: `~/Downloads/Robotvideo` on the Mac, byte-identical to `/PHShome/zy992/Wilson/generation input/` on ERIS (all 353 generation-input PNG/TXT files match by MD5). One input = one video folder's generation first frame + generation prompt.

## 1. How many inputs are left

| Dataset | Inputs available | Generated (first 15) | Not generated |
|---|---:|---:|---:|
| TAVP | 23 | 15 | 8 |
| Humanoid_Everyday | 49 | 15 | 34 |
| NVIDIA_GR1_100 | 92 | 15 | 77 |
| **Total** | **164** | **45** | **119** |

Each generated input was run through 3 generators (Cosmos 2.5, Cosmos 3, ABot-PhysWorld), so 45 inputs = 135 videos. Running the remaining 119 inputs through the same 3 generators would add 357 videos.

Not counted as inputs:

- TAVP: 10 videos also have `prompt for original multi step action(reproducing the original video).txt`, a full multi-step prompt for the same first frame; only the `first_substep/` pair was used.
- TAVP Dobot: `source_first_frame.png` is the full 1920x480 three-camera frame; the input is the cropped middle view (640x480).
- Humanoid_Everyday: `first_person/first_frame.png` exists but has no generation prompt.

## 2. Method

Every metric is computed on luminance (0-255). Script: `robot/experiments/world_model_generation/input_quality.py`; per-image values in `robotvideo_input_quality.csv` next to this file.

| Metric | Meaning |
|---|---|
| Upscale | Cover-resize factor from the source frame to the generation size (1280x704, 960x704, 704x1280 or 704x960, nearest aspect). >1 means the model saw an enlarged image. |
| Sharpness (model view) | Variance of the Laplacian of the exact image the generators received. **Grades are based on this.** |
| Sharpness (native) | Same at the source's own size. Not comparable across sizes. |
| Detail bandwidth | Fraction of the frequency limit (Nyquist) holding 99% of spectral energy. ~0.5 = full detail (sharp GR1 frames); ~0.25 = the frame resolves only about half its nominal linear resolution, i.e. it is soft even at its own size (often an upsampled or low-bitrate video frame). |
| Noise σ | Immerkaer fast noise estimate. |
| Blockiness | Mean step across 8-pixel grid lines / elsewhere. ~1.0 none; >1.35 visible compression blocks. |
| Exposure | Mean luma; % of pixels clipped dark (≤5) / bright (≥250). |
| Contrast, colorfulness | RMS contrast; Hasler-Suesstrunk colorfulness. |

No input has black letterbox/pillarbox borders.

## 3. Grades

Thresholds on model-view sharpness, set by inspecting 1:1 crops of every source-size group:

| Grade | Sharpness (model view) | Label | What it looks like |
|---|---|---|---|
| **A** | ≥ 400 | Good | crisp edges and fine texture at model resolution |
| **B** | 120–400 | Fair | usable; mild softness or compression, small detail smeared |
| **C** | 40–120 | Poor | visibly soft and/or blocky; upscaled or low-bitrate source |
| **D** | < 40 | Very poor | heavily blurred after upscaling; fine structure (fingers, edges) lost |

| Dataset | A | B | C | D | Generated: A/B/C/D | Remaining: A/B/C/D |
|---|---:|---:|---:|---:|---|---|
| TAVP | 0 | 8 | 15 | 0 | 0/8/7/0 | 0/0/8/0 |
| Humanoid_Everyday | 7 | 17 | 14 | 11 | 2/5/5/3 | 5/12/9/8 |
| NVIDIA_GR1_100 | 92 | 0 | 0 | 0 | 15/0/0/0 | 77/0/0/0 |
| **All** | **99** | **25** | **29** | **11** | | |

## 4. Quality by source group

| Dataset | Source size | n | Upscale | Sharpness model view | Detail bandwidth | Noise σ | Blockiness | Dark clip % | Grades | Assessment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| NVIDIA_GR1_100 | 1280x800 | 92 | 1.00 | 694 | 0.48 | 1.89 | 1.18 | 8.1 | A×92 | Best set. Native resolution, full detail bandwidth, mild sensor noise. Dark clipping comes from black robot/background regions, not underexposure. |
| Humanoid_Everyday | 1440x1920 | 5 | 0.50 | 562 | 0.29 | 0.48 | 1.17 | 0.7 | A×3 B×2 | High-resolution portrait; soft at native size but downscaled 2x for the model, so the model view is crisp. |
| Humanoid_Everyday | 1920x1440 | 2 | 0.50 | 415 | 0.26 | 0.39 | 1.20 | 0.1 | A×2 | As above, landscape. |
| Humanoid_Everyday | 720x1280 | 18 | 1.00 | 181 | 0.25 | 0.51 | 1.23 | 1.4 | A×2 B×14 C×2 | Nominally native, but bandwidth ~0.25: soft content (likely a resized or low-bitrate source). Mostly grade B. |
| TAVP | 1280x720 | 8 | 1.00 | 159 | 0.29 | 0.69 | 1.19 | 0.0 | B×8 | Franka. Native size, but detail bandwidth ~0.29: the source videos are ~0.16 Mbps, so edges are smoothed and fine texture is gone. Blocking is mild. |
| Humanoid_Everyday | 592x1280 | 8 | 1.19 | 78 | 0.24 | 0.39 | 1.33 | 0.1 | B×1 C×7 | Slight 1.19x upscale on top of soft content; some frames blocky. |
| TAVP | 640x480 | 14 | 1.50 | 71 | 0.37 | 0.86 | 1.43 | 1.1 | C×14 | Dobot middle view cut from the 1920x480 three-camera video, upscaled 1.5x. Strong 8-px compression blocking, low light (mean luma ~78). |
| Humanoid_Everyday | 360x480 | 1 | 2.00 | 59 | 0.48 | 1.61 | 1.09 | 0.1 | C×1 | Upscaled 2x. |
| TAVP | 960x720 | 1 | 1.00 | 54 | 0.25 | 0.38 | 1.27 | 13.9 | C×1 | Dobot illumination-variation clip: dim (mean luma ~56, 14% crushed blacks) and soft. |
| Humanoid_Everyday | 448x960 | 4 | 1.57 | 49 | 0.24 | 0.47 | 1.35 | 0.1 | C×4 | Upscaled 1.57x from an already soft frame; blocky. |
| Humanoid_Everyday | 320x568 | 11 | 2.25 | 18 | 0.41 | 0.91 | 1.15 | 1.4 | D×11 | Upscaled 2.25x. Very blurry at model resolution; hands, edges and small objects lose definition. Worst group. |

Values are medians within the group.

## 5. Key findings

1. **NVIDIA_GR1_100 is uniformly good**: all 92 inputs are grade A, so the 77 remaining GR1 inputs can be generated without quality concerns.
2. **All remaining TAVP inputs are grade C**: the 8 not yet generated (016-023) are Dobot clips (640x480 middle-view crops, plus one dim 960x720 clip). Generating them adds only low-quality, blocky conditioning frames.
3. **Humanoid_Everyday is mixed and source size is misleading**: 720x1280 frames have detail bandwidth ~0.25 versus ~0.5 for sharp frames, i.e. they resolve roughly half their nominal linear resolution (about what a sharp 360x640 frame holds), while the 1440x1920 frames are the sharpest Humanoid inputs once downscaled. Of 34 remaining Humanoid inputs, 8 are grade D (320x568, upscaled 2.25x).
4. **Compression blocking** above the visible threshold appears in all 14 TAVP 640x480 Dobot frames and in 4 Humanoid frames (592x1280 and 448x960).
5. **Exposure** is acceptable everywhere except TAVP 023 (illumination-variation clip, deliberately dark). Highlight clipping is ≤2% for every image.
6. The `low_res_input/` folders in the generated outputs (TAVP 09-15, Humanoid 06/11/14/15) are exactly the generated grade C/D items with ≥1.5x upscaling; grade C also includes some native-size but soft or blocky frames that are not in those folders.

## 6. Per-image table

### TAVP

| # | Folder | Generated | Size | Upscale | Sharpness model | Bandwidth | Noise σ | Blockiness | Mean luma | Dark clip % | Grade |
|---:|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | `001_franka_collect_fruits_standard_success` | yes | 1280x720 | 1.00 | 149 | 0.28 | 0.68 | 1.21 | 118 | 0.0 | B |
| 2 | `002_franka_fold_towel_standard_success` | yes | 1280x720 | 1.00 | 132 | 0.27 | 0.59 | 1.23 | 115 | 0.0 | B |
| 3 | `003_franka_pick_and_place_standard_success` | yes | 1280x720 | 1.00 | 181 | 0.32 | 0.76 | 1.18 | 119 | 0.0 | B |
| 4 | `004_franka_press_buttons_standard_success` | yes | 1280x720 | 1.00 | 182 | 0.31 | 0.76 | 1.19 | 117 | 0.0 | B |
| 5 | `005_franka_reach_and_drag_standard_success` | yes | 1280x720 | 1.00 | 140 | 0.28 | 0.61 | 1.24 | 116 | 0.0 | B |
| 6 | `006_franka_stack_bowls_standard_success` | yes | 1280x720 | 1.00 | 160 | 0.29 | 0.70 | 1.19 | 117 | 0.0 | B |
| 7 | `007_franka_collect_fruits_failure_case_failure` | yes | 1280x720 | 1.00 | 169 | 0.30 | 0.72 | 1.19 | 117 | 0.0 | B |
| 8 | `008_franka_pick_and_place_failure_case_failure` | yes | 1280x720 | 1.00 | 158 | 0.29 | 0.69 | 1.19 | 115 | 0.0 | B |
| 9 | `009_dobot_collect_fruits_standard_success` | yes | 640x480 | 1.50 | 74 | 0.37 | 0.90 | **1.41** | 77 | 1.1 | C |
| 10 | `010_dobot_stack_bowls_standard_success` | yes | 640x480 | 1.50 | 73 | 0.38 | 0.88 | **1.40** | 76 | 1.2 | C |
| 11 | `011_dobot_grape_picking_standard_success` | yes | 640x480 | 1.50 | 72 | 0.38 | 0.85 | **1.44** | 77 | 1.1 | C |
| 12 | `012_dobot_press_buttons_standard_success` | yes | 640x480 | 1.50 | 79 | 0.38 | 0.93 | **1.40** | 79 | 0.5 | C |
| 13 | `013_dobot_put_item_in_drawer_standard_success` | yes | 640x480 | 1.50 | 80 | 0.39 | 0.94 | **1.38** | 80 | 0.5 | C |
| 14 | `014_dobot_collect_fruits_failure_case_failure` | yes | 640x480 | 1.50 | 72 | 0.36 | 0.87 | **1.47** | 76 | 1.9 | C |
| 15 | `015_dobot_put_item_in_drawer_failure_case_failure` | yes | 640x480 | 1.50 | 81 | 0.38 | 0.95 | **1.38** | 81 | 0.8 | C |
| 16 | `016_dobot_grape_picking_location_generalization_success` |  | 640x480 | 1.50 | 71 | 0.37 | 0.83 | **1.44** | 79 | 0.9 | C |
| 17 | `017_dobot_grape_picking_background_generalization_success` |  | 640x480 | 1.50 | 69 | 0.47 | 0.95 | **1.42** | 114 | 0.3 | C |
| 18 | `018_dobot_grape_picking_instance_generalization_success` |  | 640x480 | 1.50 | 67 | 0.37 | 0.81 | **1.45** | 77 | 1.9 | C |
| 19 | `019_dobot_grape_picking_object_generalization_success` |  | 640x480 | 1.50 | 67 | 0.36 | 0.81 | **1.47** | 77 | 2.2 | C |
| 20 | `020_dobot_grape_picking_object_generalization_success` |  | 640x480 | 1.50 | 67 | 0.36 | 0.80 | **1.44** | 77 | 2.1 | C |
| 21 | `021_dobot_grape_picking_object_generalization_success` |  | 640x480 | 1.50 | 69 | 0.37 | 0.82 | **1.45** | 78 | 1.6 | C |
| 22 | `022_dobot_grape_picking_heavy_occlusion_success` |  | 640x480 | 1.50 | 69 | 0.36 | 0.86 | **1.43** | 80 | 2.1 | C |
| 23 | `023_dobot_grape_picking_illumination_variation_success` |  | 960x720 | 1.00 | 54 | 0.25 | 0.38 | 1.27 | 56 | 13.9 | C |

### Humanoid_Everyday

| # | Folder | Generated | Size | Upscale | Sharpness model | Bandwidth | Noise σ | Blockiness | Mean luma | Dark clip % | Grade |
|---:|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | `articulated-1__rotate_chair` | yes | 720x1280 | 1.00 | 239 | 0.43 | 0.72 | 1.22 | 99 | 1.4 | B |
| 2 | `articulated-2__compress_liquid_hand_soap` | yes | 720x1280 | 1.00 | 139 | 0.22 | 0.43 | 1.23 | 102 | 1.7 | B |
| 3 | `articulated-3__unfold_multimeter_stand` | yes | 720x1280 | 1.00 | 148 | 0.23 | 0.40 | 1.25 | 99 | 2.5 | B |
| 4 | `articulated-4__compress_air_pumper` | yes | 592x1280 | 1.19 | 74 | 0.19 | 0.32 | **1.44** | 88 | 0.1 | C |
| 5 | `articulated-5__turn_on_faucet_handle` | yes | 592x1280 | 1.19 | 98 | 0.25 | 0.43 | 1.34 | 91 | 0.0 | C |
| 6 | `articulated-6__press_light_switch` | yes | 448x960 | 1.57 | 50 | 0.26 | 0.51 | 1.30 | 88 | 0.0 | C |
| 7 | `articulated-7__close_cabinet_door` | yes | 720x1280 | 1.00 | 205 | 0.38 | 0.48 | 1.19 | 142 | 0.9 | B |
| 8 | `deformable-1__adjust_loosely_closed_bag_opening` | yes | 592x1280 | 1.19 | 65 | 0.23 | 0.35 | 1.35 | 85 | 0.1 | C |
| 9 | `deformable-2__fold_towel` | yes | 720x1280 | 1.00 | 462 | 0.42 | 0.91 | 1.16 | 108 | 1.9 | A |
| 10 | `deformable-3__fold_jacket_on_desk` | yes | 720x1280 | 1.00 | 154 | 0.26 | 0.51 | 1.24 | 98 | 1.5 | B |
| 11 | `deformable-4__fold_toolkit` | yes | 320x568 | 2.25 | 14 | 0.30 | 0.75 | 1.15 | 102 | 2.2 | D |
| 12 | `deformable-5__compress_soft_travel_bag` | yes | 592x1280 | 1.19 | 88 | 0.26 | 0.45 | 1.31 | 93 | 0.1 | C |
| 13 | `deformable-6__compress_plastic_bag` | yes | 720x1280 | 1.00 | 523 | 0.52 | 1.09 | 1.15 | 111 | 5.0 | A |
| 14 | `high-precision-1__open_a_bottle` | yes | 320x568 | 2.25 | 20 | 0.42 | 0.97 | 1.15 | 123 | 2.0 | D |
| 15 | `high-precision-2__place_soldering_gun_in_base` | yes | 320x568 | 2.25 | 18 | 0.36 | 0.78 | 1.16 | 104 | 1.4 | D |
| 16 | `high-precision-3__align_stack_of_books` |  | 320x568 | 2.25 | 13 | 0.32 | 0.63 | 1.17 | 112 | 1.5 | D |
| 17 | `high-precision-4__place_kettle_on_base` |  | 720x1280 | 1.00 | 292 | 0.33 | 0.68 | 1.28 | 72 | 0.5 | B |
| 18 | `high-precision-5__insert_flower_into_vase` |  | 720x1280 | 1.00 | 197 | 0.24 | 0.52 | 1.18 | 110 | 1.8 | B |
| 19 | `high-precision-6__press_start_boiling_button` |  | 592x1280 | 1.19 | 83 | 0.31 | 0.45 | 1.29 | 62 | 0.1 | C |
| 20 | `high-precision-7__unplug_a_charger` |  | 320x568 | 2.25 | 29 | 0.46 | 1.30 | 1.14 | 127 | 0.0 | D |
| 21 | `high-precision-8__stack_paper_towel_rolls` |  | 448x960 | 1.57 | 48 | 0.22 | 0.44 | **1.39** | 91 | 0.1 | C |
| 22 | `human-robot-1__hand_over_emotional_support_dumpling` |  | 720x1280 | 1.00 | 137 | 0.21 | 0.39 | 1.28 | 103 | 1.5 | B |
| 23 | `human-robot-2__hand_over_toilet_paper` |  | 720x1280 | 1.00 | 99 | 0.19 | 0.31 | 1.34 | 106 | 0.8 | C |
| 24 | `human-robot-3__hand_over_grey_duster` |  | 720x1280 | 1.00 | 149 | 0.23 | 0.41 | 1.33 | 99 | 1.1 | B |
| 25 | `human-robot-4__hand_over_book_from_magazine_stand` |  | 720x1280 | 1.00 | 150 | 0.23 | 0.44 | 1.28 | 102 | 1.8 | B |
| 26 | `human-robot-5__handover_dumpling` |  | 720x1280 | 1.00 | 183 | 0.23 | 0.51 | 1.24 | 104 | 1.7 | B |
| 27 | `human-robot-6__hand_over_flower` |  | 720x1280 | 1.00 | 240 | 0.27 | 0.54 | 1.24 | 105 | 0.1 | B |
| 28 | `human-robot-7__shake_hand_with_person` |  | 592x1280 | 1.19 | 128 | 0.41 | 0.63 | 1.30 | 73 | 0.1 | B |
| 29 | `loco-manip-1__clean_the_window` |  | 1440x1920 | 0.50 | 719 | 0.33 | 0.70 | 1.17 | 122 | 0.7 | A |
| 30 | `loco-manip-2__pick_up_kettle_place_on_base_and_turn_on` |  | 1440x1920 | 0.50 | 562 | 0.29 | 0.48 | 1.17 | 133 | 1.7 | A |
| 31 | `loco-manip-3__throw_paper_to_trashcan` |  | 1440x1920 | 0.50 | 1296 | 0.41 | 0.86 | 1.13 | 122 | 0.7 | A |
| 32 | `loco-manip-4__walk_to_gate_button_and_open_gate` |  | 1920x1440 | 0.50 | 413 | 0.28 | 0.32 | 1.21 | 133 | 0.1 | A |
| 33 | `loco-manip-5__pick_up_container_walk_and_stick_to_whiteboard` |  | 1440x1920 | 0.50 | 363 | 0.21 | 0.30 | 1.19 | 141 | 0.4 | B |
| 34 | `loco-manip-6__hold_whiteboard_and_walk` |  | 1920x1440 | 0.50 | 417 | 0.24 | 0.45 | 1.19 | 133 | 0.0 | A |
| 35 | `loco-manip-7__walk_to_fire_extinguisher_and_open_door` |  | 1440x1920 | 0.50 | 377 | 0.25 | 0.37 | 1.20 | 130 | 0.4 | B |
| 36 | `pick-place-1__lift_a_kettle_from_its_base` |  | 448x960 | 1.57 | 68 | 0.31 | 0.60 | 1.27 | 87 | 0.1 | C |
| 37 | `pick-place-2__place_a_cube_in_a_bag` |  | 360x480 | 2.00 | 59 | 0.48 | 1.61 | 1.09 | 121 | 0.1 | C |
| 38 | `pick-place-3__pick_up_a_tripod_and_stand_it_on_the_desk` |  | 320x568 | 2.25 | 27 | 0.50 | 1.36 | 1.15 | 122 | 1.1 | D |
| 39 | `pick-place-4__pick_up_an_air_purify_spray_and_put_it_inside_a_container` |  | 720x1280 | 1.00 | 194 | 0.28 | 0.56 | 1.22 | 107 | 1.3 | B |
| 40 | `pick-place-5__pick_a_multi_meter_and_place_it_in_a_container` |  | 592x1280 | 1.19 | 48 | 0.17 | 0.29 | 1.32 | 95 | 0.0 | C |
| 41 | `pick-place-6__pick_a_3d_printing_filament_and_put_it_inside_the_container` |  | 592x1280 | 1.19 | 64 | 0.20 | 0.35 | **1.38** | 92 | 0.8 | C |
| 42 | `pick-place-7__stack_a_rolls_of_paper_towel_on_top_of_the_other` |  | 448x960 | 1.57 | 48 | 0.22 | 0.44 | **1.39** | 91 | 0.1 | C |
| 43 | `tool-use-1__pour_water_from_a_kettle_into_a_cup` |  | 320x568 | 2.25 | 18 | 0.42 | 1.06 | 1.16 | 123 | 0.0 | D |
| 44 | `tool-use-2__wipe_a_table_with_an_eraser` |  | 320x568 | 2.25 | 9 | 0.25 | 0.53 | 1.15 | 97 | 1.7 | D |
| 45 | `tool-use-3__wipe_a_monitor_with_a_cloth` |  | 320x568 | 2.25 | 10 | 0.34 | 0.68 | 1.16 | 94 | 0.6 | D |
| 46 | `tool-use-4__use_a_garden_tool` |  | 320x568 | 2.25 | 19 | 0.41 | 0.91 | 1.15 | 126 | 3.9 | D |
| 47 | `tool-use-5__use_a_duster_to_dust_the_desk` |  | 720x1280 | 1.00 | 179 | 0.27 | 0.54 | 1.20 | 106 | 1.3 | B |
| 48 | `tool-use-6__wipe_table_with_cloth` |  | 720x1280 | 1.00 | 117 | 0.19 | 0.33 | 1.23 | 105 | 0.5 | C |
| 49 | `tool-use-7__click_a_mouse` |  | 320x568 | 2.25 | 22 | 0.43 | 1.07 | 1.13 | 124 | 0.3 | D |

### NVIDIA_GR1_100

| # | Folder | Generated | Size | Upscale | Sharpness model | Bandwidth | Noise σ | Blockiness | Mean luma | Dark clip % | Grade |
|---:|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | `1` | yes | 1280x800 | 1.00 | 815 | 0.51 | 2.18 | 1.17 | 108 | 7.8 | A |
| 2 | `2` | yes | 1280x800 | 1.00 | 703 | 0.47 | 1.84 | 1.18 | 108 | 7.2 | A |
| 3 | `3` | yes | 1280x800 | 1.00 | 869 | 0.52 | 2.08 | 1.18 | 114 | 6.8 | A |
| 4 | `4` | yes | 1280x800 | 1.00 | 646 | 0.47 | 1.84 | 1.18 | 112 | 7.8 | A |
| 5 | `6` | yes | 1280x800 | 1.00 | 621 | 0.44 | 1.59 | 1.19 | 109 | 14.1 | A |
| 6 | `9` | yes | 1280x800 | 1.00 | 519 | 0.47 | 1.67 | 1.20 | 117 | 5.4 | A |
| 7 | `10` | yes | 1280x800 | 1.00 | 910 | 0.52 | 2.31 | 1.15 | 113 | 9.5 | A |
| 8 | `12` | yes | 1280x800 | 1.00 | 652 | 0.48 | 1.91 | 1.18 | 112 | 7.7 | A |
| 9 | `13` | yes | 1280x800 | 1.00 | 843 | 0.48 | 2.08 | 1.17 | 108 | 10.6 | A |
| 10 | `14` | yes | 1280x800 | 1.00 | 816 | 0.48 | 2.06 | 1.16 | 106 | 11.5 | A |
| 11 | `15` | yes | 1280x800 | 1.00 | 790 | 0.47 | 1.98 | 1.17 | 109 | 8.1 | A |
| 12 | `16` | yes | 1280x800 | 1.00 | 804 | 0.48 | 2.12 | 1.16 | 111 | 7.3 | A |
| 13 | `17` | yes | 1280x800 | 1.00 | 972 | 0.52 | 2.26 | 1.18 | 106 | 10.5 | A |
| 14 | `18` | yes | 1280x800 | 1.00 | 928 | 0.51 | 2.20 | 1.20 | 108 | 10.8 | A |
| 15 | `19` | yes | 1280x800 | 1.00 | 756 | 0.48 | 1.96 | 1.19 | 111 | 7.2 | A |
| 16 | `20` |  | 1280x800 | 1.00 | 709 | 0.47 | 1.89 | 1.19 | 106 | 10.3 | A |
| 17 | `21` |  | 1280x800 | 1.00 | 696 | 0.46 | 1.86 | 1.19 | 104 | 10.7 | A |
| 18 | `23` |  | 1280x800 | 1.00 | 585 | 0.48 | 1.78 | 1.18 | 114 | 5.8 | A |
| 19 | `24` |  | 1280x800 | 1.00 | 631 | 0.48 | 1.82 | 1.21 | 112 | 8.0 | A |
| 20 | `25` |  | 1280x800 | 1.00 | 639 | 0.45 | 1.73 | 1.18 | 107 | 12.9 | A |
| 21 | `26` |  | 1280x800 | 1.00 | 675 | 0.50 | 1.52 | 1.29 | 120 | 4.9 | A |
| 22 | `27` |  | 1280x800 | 1.00 | 936 | 0.51 | 2.31 | 1.17 | 108 | 9.7 | A |
| 23 | `29` |  | 1280x800 | 1.00 | 658 | 0.43 | 1.77 | 1.19 | 108 | 14.4 | A |
| 24 | `30` |  | 1280x800 | 1.00 | 641 | 0.44 | 1.85 | 1.19 | 106 | 8.2 | A |
| 25 | `31` |  | 1280x800 | 1.00 | 584 | 0.48 | 1.77 | 1.20 | 115 | 5.0 | A |
| 26 | `32` |  | 1280x800 | 1.00 | 671 | 0.46 | 1.82 | 1.18 | 107 | 9.0 | A |
| 27 | `33` |  | 1280x800 | 1.00 | 712 | 0.52 | 1.58 | 1.28 | 116 | 4.8 | A |
| 28 | `34` |  | 1280x800 | 1.00 | 978 | 0.52 | 2.32 | 1.18 | 110 | 9.5 | A |
| 29 | `35` |  | 1280x800 | 1.00 | 723 | 0.48 | 1.78 | 1.20 | 115 | 10.0 | A |
| 30 | `36` |  | 1280x800 | 1.00 | 899 | 0.50 | 2.09 | 1.17 | 108 | 10.1 | A |
| 31 | `37` |  | 1280x800 | 1.00 | 589 | 0.50 | 1.37 | 1.29 | 122 | 2.1 | A |
| 32 | `38` |  | 1280x800 | 1.00 | 542 | 0.46 | 1.67 | 1.20 | 112 | 6.8 | A |
| 33 | `39` |  | 1280x800 | 1.00 | 1121 | 0.55 | 2.42 | 1.16 | 114 | 8.2 | A |
| 34 | `40` |  | 1280x800 | 1.00 | 644 | 0.50 | 1.87 | 1.16 | 115 | 5.3 | A |
| 35 | `41` |  | 1280x800 | 1.00 | 869 | 0.49 | 2.16 | 1.19 | 106 | 8.2 | A |
| 36 | `42` |  | 1280x800 | 1.00 | 573 | 0.48 | 1.74 | 1.18 | 114 | 5.7 | A |
| 37 | `43` |  | 1280x800 | 1.00 | 641 | 0.49 | 1.89 | 1.18 | 118 | 4.5 | A |
| 38 | `45` |  | 1280x800 | 1.00 | 716 | 0.47 | 1.91 | 1.18 | 112 | 7.8 | A |
| 39 | `46` |  | 1280x800 | 1.00 | 592 | 0.47 | 1.67 | 1.18 | 110 | 7.3 | A |
| 40 | `47` |  | 1280x800 | 1.00 | 577 | 0.51 | 1.35 | 1.29 | 120 | 1.8 | A |
| 41 | `48` |  | 1280x800 | 1.00 | 595 | 0.47 | 1.78 | 1.21 | 114 | 8.0 | A |
| 42 | `49` |  | 1280x800 | 1.00 | 763 | 0.52 | 1.65 | 1.26 | 116 | 5.5 | A |
| 43 | `50` |  | 1280x800 | 1.00 | 782 | 0.49 | 1.99 | 1.18 | 109 | 7.8 | A |
| 44 | `51` |  | 1280x800 | 1.00 | 620 | 0.50 | 1.82 | 1.18 | 121 | 3.9 | A |
| 45 | `52` |  | 1280x800 | 1.00 | 602 | 0.44 | 1.76 | 1.18 | 108 | 12.0 | A |
| 46 | `53` |  | 1280x800 | 1.00 | 567 | 0.50 | 1.31 | 1.29 | 117 | 2.4 | A |
| 47 | `54` |  | 1280x800 | 1.00 | 667 | 0.46 | 1.87 | 1.19 | 109 | 7.1 | A |
| 48 | `55` |  | 1280x800 | 1.00 | 565 | 0.45 | 1.65 | 1.19 | 114 | 10.7 | A |
| 49 | `56` |  | 1280x800 | 1.00 | 793 | 0.51 | 2.08 | 1.18 | 108 | 9.1 | A |
| 50 | `57` |  | 1280x800 | 1.00 | 880 | 0.55 | 2.31 | 1.16 | 114 | 4.1 | A |
| 51 | `58` |  | 1280x800 | 1.00 | 692 | 0.51 | 1.95 | 1.18 | 116 | 4.6 | A |
| 52 | `59` |  | 1280x800 | 1.00 | 894 | 0.49 | 2.24 | 1.16 | 110 | 12.3 | A |
| 53 | `60` |  | 1280x800 | 1.00 | 614 | 0.44 | 1.83 | 1.19 | 103 | 9.5 | A |
| 54 | `61` |  | 1280x800 | 1.00 | 573 | 0.50 | 1.35 | 1.28 | 121 | 2.1 | A |
| 55 | `62` |  | 1280x800 | 1.00 | 702 | 0.47 | 1.92 | 1.17 | 112 | 10.2 | A |
| 56 | `63` |  | 1280x800 | 1.00 | 568 | 0.47 | 1.74 | 1.19 | 116 | 5.0 | A |
| 57 | `64` |  | 1280x800 | 1.00 | 726 | 0.49 | 2.04 | 1.17 | 113 | 8.2 | A |
| 58 | `65` |  | 1280x800 | 1.00 | 604 | 0.48 | 1.82 | 1.18 | 118 | 4.8 | A |
| 59 | `66` |  | 1280x800 | 1.00 | 680 | 0.46 | 1.89 | 1.22 | 110 | 7.1 | A |
| 60 | `67` |  | 1280x800 | 1.00 | 748 | 0.47 | 1.92 | 1.20 | 107 | 8.5 | A |
| 61 | `68` |  | 1280x800 | 1.00 | 971 | 0.52 | 2.13 | 1.19 | 105 | 11.0 | A |
| 62 | `69` |  | 1280x800 | 1.00 | 582 | 0.49 | 1.35 | 1.26 | 118 | 3.1 | A |
| 63 | `70` |  | 1280x800 | 1.00 | 674 | 0.48 | 1.85 | 1.18 | 109 | 9.3 | A |
| 64 | `71` |  | 1280x800 | 1.00 | 647 | 0.46 | 1.75 | 1.18 | 109 | 12.8 | A |
| 65 | `72` |  | 1280x800 | 1.00 | 871 | 0.49 | 2.18 | 1.17 | 103 | 15.1 | A |
| 66 | `73` |  | 1280x800 | 1.00 | 811 | 0.46 | 2.12 | 1.19 | 102 | 11.9 | A |
| 67 | `74` |  | 1280x800 | 1.00 | 765 | 0.48 | 1.92 | 1.19 | 111 | 8.2 | A |
| 68 | `75` |  | 1280x800 | 1.00 | 651 | 0.49 | 1.86 | 1.19 | 115 | 5.2 | A |
| 69 | `76` |  | 1280x800 | 1.00 | 688 | 0.47 | 1.96 | 1.20 | 100 | 10.3 | A |
| 70 | `77` |  | 1280x800 | 1.00 | 572 | 0.47 | 1.73 | 1.19 | 118 | 5.0 | A |
| 71 | `78` |  | 1280x800 | 1.00 | 947 | 0.50 | 2.18 | 1.19 | 104 | 10.9 | A |
| 72 | `79` |  | 1280x800 | 1.00 | 996 | 0.49 | 2.23 | 1.19 | 103 | 15.7 | A |
| 73 | `80` |  | 1280x800 | 1.00 | 683 | 0.48 | 1.94 | 1.20 | 109 | 6.4 | A |
| 74 | `81` |  | 1280x800 | 1.00 | 688 | 0.43 | 1.68 | 1.18 | 99 | 18.4 | A |
| 75 | `82` |  | 1280x800 | 1.00 | 801 | 0.49 | 2.03 | 1.18 | 112 | 7.2 | A |
| 76 | `83` |  | 1280x800 | 1.00 | 615 | 0.49 | 1.84 | 1.19 | 117 | 5.1 | A |
| 77 | `84` |  | 1280x800 | 1.00 | 652 | 0.46 | 1.75 | 1.19 | 115 | 10.9 | A |
| 78 | `85` |  | 1280x800 | 1.00 | 1144 | 0.57 | 2.54 | 1.14 | 114 | 7.0 | A |
| 79 | `86` |  | 1280x800 | 1.00 | 853 | 0.49 | 2.24 | 1.16 | 114 | 8.3 | A |
| 80 | `88` |  | 1280x800 | 1.00 | 849 | 0.49 | 1.91 | 1.18 | 115 | 11.5 | A |
| 81 | `89` |  | 1280x800 | 1.00 | 656 | 0.43 | 1.84 | 1.20 | 113 | 8.1 | A |
| 82 | `90` |  | 1280x800 | 1.00 | 918 | 0.51 | 2.21 | 1.15 | 112 | 8.3 | A |
| 83 | `91` |  | 1280x800 | 1.00 | 800 | 0.48 | 2.16 | 1.19 | 106 | 11.2 | A |
| 84 | `92` |  | 1280x800 | 1.00 | 600 | 0.45 | 1.78 | 1.18 | 111 | 7.0 | A |
| 85 | `93` |  | 1280x800 | 1.00 | 916 | 0.50 | 2.29 | 1.17 | 110 | 11.5 | A |
| 86 | `94` |  | 1280x800 | 1.00 | 701 | 0.46 | 1.72 | 1.19 | 112 | 12.9 | A |
| 87 | `95` |  | 1280x800 | 1.00 | 650 | 0.46 | 1.79 | 1.20 | 112 | 9.8 | A |
| 88 | `96` |  | 1280x800 | 1.00 | 730 | 0.49 | 2.06 | 1.17 | 115 | 6.2 | A |
| 89 | `97` |  | 1280x800 | 1.00 | 814 | 0.52 | 2.14 | 1.17 | 108 | 8.6 | A |
| 90 | `98` |  | 1280x800 | 1.00 | 1096 | 0.57 | 2.46 | 1.17 | 116 | 7.3 | A |
| 91 | `99` |  | 1280x800 | 1.00 | 606 | 0.49 | 1.83 | 1.17 | 117 | 4.7 | A |
| 92 | `100` |  | 1280x800 | 1.00 | 680 | 0.47 | 1.90 | 1.20 | 112 | 6.0 | A |

