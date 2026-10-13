# Concise first-frame action prompt protocol

Protocol version: `concise_action_v1`.

The reference [RoboWM-Bench close-drawer prompt](https://github.com/fffstrong/RoboWM-Bench/blob/main/wm_inputs/close_drawer/episode_000000.txt) has a short action sentence and shared robot/camera constraints. Its [alternative prompts](https://github.com/fffstrong/RoboWM-Bench/blob/main/wm_inputs/close_drawer/new.txt) vary the necessary interaction while retaining those constraints. The accompanying first frame already shows the robot, open drawer and scene. This protocol adapts that division of labor to TAVP.

## Prompt format

Use **two sentences**, normally **30–55 words total**:

1. **Action and visible end state.** Name the target and the essential manipulation, in the correct order. Use one or two connected clauses. Add color or a relative position only when it distinguishes the target from another visible object. Include an essential tool, release, or failure outcome when it changes what must be generated.
2. **Shared physical/camera constraint.** Use the same short sentence across the dataset:

> Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

Template:

> The robotic arm [essential manipulation] [identifiable object] [target/end state]. Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

The word range is a guide, not a reason to remove an essential distinction. A complete multi-action reference can need a longer first sentence; the preferred simplified reference usually does not.

## What belongs in the action sentence

- The interaction that changes the scene: pick and place, press, pull open, fold, sweep, or attempt and fail.
- An object description sufficient to identify it in the first frame.
- The target or final relationship: inside a bowl, on a plate, drawer open, towel folded, apple between buttons.
- Causal order when needed: grasp the tool before sweeping; lift the bowl before setting it down.
- The actual observed result. A misplaced object remains misplaced in the prompt; an unsuccessful grasp is not rewritten as a successful placement.

## What stays out of the generation prompt

The detailed review still records these, but the prompt omits them unless essential to identify the action:

- A scene inventory, camera elevation, lighting description, robot color, or background furniture already supplied by the first frame.
- Repeated approach/lower/align/close/pause/lift/carry/lower/open/withdraw narration when “picks up and places” expresses the intended manipulation.
- Exact finger contact points, wrist rotations, path directions, incidental pauses, speed, or source-video timestamps.
- Minor settling, incidental object shifts, and ambiguous material/fruit classifications.
- A fixed generated duration, author task captions, robustness labels, uncertainty commentary, or benchmark metadata.

Keep a necessary motion distinction: “folds the left half over the right half” distinguishes a fold, and “uses the wooden strip to sweep” distinguishes tool use. A normal release or small withdrawal need not be listed if the main verb and visible end state already imply it. If the selected clip ends at a press release, a brief “then lifts clear” is acceptable; do not describe the next button.

## Review and simplification workflow

1. A dedicated agent inspects the complete video, sampling every second decoded frame and examining native-resolution transition frames.
2. The agent inspects the initial PNG in the generation camera view. For Dobot, this is only the middle third-person panel.
3. The agent records the full observed chronology and any ambiguity in `metadata/action_review.json`.
4. If there are independent subtasks, retain the first complete manipulation or first complete failed attempt, starting at original frame 0 and ending before the next subtask. Keep the action sentence within that endpoint.
   A brief corrective nudge or settling adjustment of the same placement remains part of that single task; ordinary action phases do not justify a split.
5. Write the two-sentence preferred prompt. Write a separate concise full-sequence prompt for the original video. If a first-substep branch is retained, keep that branch plus the full-sequence prompt and action review; omit duplicated full-video media and derivative metadata.
6. Check that the target is identifiable from the first frame, the action matches the selected frames, and the stated end state is visible or honestly reflects the filmed failure.

## TAVP examples

**First fruit placement:**

> The robotic arm picks up the green cucumber-shaped object and places it inside the blue bowl. Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

**First button only:**

> The robotic arm presses the yellow button with its closed gripper, then lifts clear. Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

**Towel fold:**

> The robotic arm grasps the towel's left edge and folds the left half over the right half. Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

**Tool-mediated sweep:**

> The robotic arm grasps the wooden strip and uses it to sweep the green apple into the gap between the green and red buttons. Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

**Observed misplaced carrot:**

> The robotic arm picks up the carrot, carries it toward the purple plate, and releases it onto the tabletop beside the plate. Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.

Each example is grounded in a reviewed TAVP sequence. Descriptions of uncertainty and exact contact timing remain in the review JSON rather than being added to these prompts.
