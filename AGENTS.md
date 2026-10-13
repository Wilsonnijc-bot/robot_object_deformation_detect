# Project instructions

## How to craft world-model action-generation prompts

Follow protocol `concise_action_v1` for first-frame-conditioned robot video generation. The reference is [RoboWM-Bench's close-drawer input](https://github.com/fffstrong/RoboWM-Bench/tree/main/wm_inputs/close_drawer): a concise action sentence plus shared physical/camera constraints. The first frame supplies the scene appearance.

### Humanoid Everyday and NVIDIA GR1 prompt extension

For every Humanoid Everyday **third-person** video and every NVIDIA GR1 video, use `humanoid_action_v1`, an extension of `concise_action_v1`. Each dedicated review agent owns one complete video and writes a prompt grounded in the observed motion, with more manipulation detail than the single-arm Franka examples.

- State the active **robot-relative left or right arm/hand**. Do not confuse image left/right with the robot's anatomy. Use the initial frame and visible body/arm attachment to establish the convention; if anatomy cannot be determined, use an unambiguous visible-arm description and record the uncertainty rather than inventing a side.
- If both hands participate, state each hand's distinct role and how they coordinate: holding versus manipulating, stabilizing versus pulling, supporting versus transferring, or grasping separate edges together. Describe hand-to-hand transfers and interaction with a person explicitly when present. Mention the other hand only when its movement or support matters.
- For coordinated manipulation, trace **each hand separately through every contact change** at native resolution. Establish which part it grasps (device body, handle, cap, flap, rim or mechanism), which hand bears the object, which part moves relative to the other, and the order of engagement, repositioning and release. Distinguish whole-object reorientation from motion of a hinge, dial, cap or plunger; compare object-relative landmarks across adjacent frames before naming a mechanism or motion direction. A hand beside an object does not establish stabilizing contact. Preserve essential support and regrasp transitions in the prompt; put detailed contact evidence and occlusions in the review. Use neutral, precise visible manipulation wording if the mechanism itself cannot be established.
- Retain essential grasp, lift, transport, placement/release, pressing, pulling, folding or tool-use details and the **actual visible end state**. For locomotion, include essential body movement when it is part of the manipulation. Do not infer an unseen successful release, contact, target or outcome.
- Use the NVIDIA `task_prompt.txt` as an annotation starting point, then verify and improve it from the full video and initial frame. Preserve that original file verbatim. Source task names and captions do not replace visual evidence.
- Prefer **two or three action sentences plus the shared constraint sentence**, usually **55–100 words total**; fewer words are fine for a simple action, and essential coordination may justify more. Keep descriptions direct and specific, without a scene inventory, frame numbers or uncertainty commentary in the usable prompt.
- Append the existing shared constraint sentence exactly: “Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.” Treat humanoid finger articulation as part of physically valid hand motion; do not describe a dexterous hand as a parallel-jaw gripper in the action sentences.
- These references retain the **complete original single-task clip**. Do not crop, shorten, resize, re-encode, split into first_substep, or add generated frames. Ordinary phases of the same task remain together.
- Organize each reviewed Humanoid Everyday third-person and NVIDIA GR1 video folder with `original/` for the unchanged source MP4, original task prompt and metadata, and `generation input/` containing `video.mp4` (an unchanged source copy), `first_frame.png`, `actiongeneration_prompt.txt`, `action_review.json` and `metadata.json`. Use only `actiongeneration_prompt.txt`; do not recreate the redundant `prompt.txt` alias. All three datasets use this generation-folder naming. TAVP retains its `prompt.txt` filenames and `first_substep/` branches. Humanoid first-person source files are outside this review scope and stay intact.
- In `action_review.json`, record full-sequence chronology with frame indices, anatomical-side evidence, each arm's role, coordination, outcome, uncertainties, source annotation discrepancies and inspection coverage. Save `prompt_protocol: "humanoid_action_v1"` and `prompt_provenance: "visually_authored_from_video_and_initial_frame"`.
- Temporary contact sheets may cover every second decoded frame; inspect transitions and the initial PNG at native resolution. Do not save evidence-frame directories under `generation input/`. Each source video must receive its own dedicated agent; concurrency is bounded by the session's available agent slots.

### Inspect before writing

- Assign one dedicated subagent to each source video when reviewing the dataset. Each agent owns one video; do not distribute a single video's review among several agents.
- Inspect the entire MP4 closely. Sampling every second decoded frame is acceptable; inspect action transitions and proposed clip boundaries at native resolution.
- After understanding the whole sequence, inspect the initial PNG in the generation camera view. Do not substitute the author caption for visual evidence.
- For Dobot, inspect and use only the **middle third-person panel**. Generation MP4s, initial PNGs, temporary inspection images and prompts must all refer to that view.
- Preserve detailed chronology, frame indices, ambiguity and observed outcomes in `action_review.json`. Keep that detail out of the generation prompt.

### Write a concise prompt

Use **two sentences**, normally **30–55 words total**:

1. State the **essential action and actual end state**: “The robotic arm [essential manipulation] [identifiable object] [target/outcome].” Use one or two connected clauses. Include causal order or tool use when necessary.
2. Append this shared constraint sentence exactly:

   > Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

Let the initial image provide the robot's appearance, camera angle, lighting, background and scene layout. Name an object's color or relative position only when needed to distinguish the target. Use an appearance-based description if its exact identity is uncertain.

Do not narrate routine approach, lowering, alignment, pauses, wrist rotations, finger contact points, transport trajectories or withdrawal when “picks up and places” already captures the action. Do not add a scene inventory, timestamps, generated duration, robustness-test captions, annotation commentary or a long negative-prompt list. Keep a motion distinction if it changes the task, such as the direction of a towel fold or using a strip to sweep an apple.

The word range is guidance; retain essential action distinctions. A full multi-action reference may need a longer action sentence, while the preferred simplified prompt should remain short.

### Simplify multi-step references

- When the video contains independent subtasks, create a shorter reference for the **first complete meaningful manipulation** or first complete failed attempt.
- Examples: first fruit placed, first bowl placed, first button pressed, or drawer opened before subsequent item placement.
- Keep approach, grasp, lift, transfer and release of one object together. Do not cut mid-action just to shorten the video.
- Keep a brief corrective nudge or settling adjustment of the same placement with that single task. Do not manufacture multiple subtasks from ordinary phases of one action.
- Start at original frame 0 so the initial PNG remains valid. End after the retained action visibly completes and before the next independent subtask begins.
- The simplified prompt must stop at that same action. Keep a separate concise full-sequence prompt for the original video.
- Preserve the actual filmed outcome. A missed grasp, dropped object or placement beside a target must not be rewritten as success. Record the selected action's outcome separately from the source video's overall success/failure label.

### Examples

**Fruit placement**

> The robotic arm picks up the green cucumber-shaped object and places it inside the blue bowl. Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

**First button only**

> The robotic arm presses the yellow button with its closed gripper, then lifts clear. Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

**Towel fold**

> The robotic arm grasps the towel's left edge and folds the left half over the right half. Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

**Observed failed placement**

> The robotic arm picks up the carrot, carries it toward the purple plate, and releases it onto the tabletop beside the plate. Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

### Final checks and saved artifacts

Before saving, confirm that the target is identifiable from the initial frame, the action matches the retained frames, and the described end state matches the filmed outcome. Keep uncertainties in the review metadata.

Save `prompt_protocol: "concise_action_v1"` and `prompt_provenance: "visually_authored_from_video_and_initial_frame"` in the action review. Store the usable prompt alone in `prompt.txt`. Keep downloaded source artifacts under each video's `original/` directory, preserving the media bytes; save cropped and shortened generation references under the sibling `generation input/` directory. Use frame-accurate, lossless derivatives and validate their decoding, dimensions, FPS and source-frame correspondence.

Keep each video folder organized as `original/` and `generation input/` only. Do not export or recreate evidence-frame directories under `generation input/`; use temporary inspection images and retain frame indices/observations in `action_review.json`.

When a TAVP first-substep reference exists, retain `first_substep/`, the full-sequence `prompt for original multi step action(reproducing the original video).txt`, and `action_review.json` under `generation input/`, plus the distinct Dobot source frame where applicable. Do not retain or recreate an additional full-video MP4, initial PNG or derivative metadata there. For single-task videos without a shortened branch, retain the one complete generation reference directly under `generation input/`.

Additional examples and rationale: [TAVP/WORLD_MODEL_PROMPT_PROTOCOL.md](TAVP/WORLD_MODEL_PROMPT_PROTOCOL.md).

### First-frame storage

Store each reviewed video's conditioning PNG beside its action prompt under
`generation input/` (inside `first_substep/` for a shortened TAVP reference).
Do not retain another identical PNG under `original/`. For Dobot, keep the
distinct full-panel source frame as `generation input/source_first_frame.png`;
`first_frame.png` remains the middle-view conditioning image. The distinct source
PNG is permitted alongside a `first_substep/` branch. Keep first-frame checksums
and paths in metadata and manifests current. Humanoid first-person files retain
their existing source layout.
