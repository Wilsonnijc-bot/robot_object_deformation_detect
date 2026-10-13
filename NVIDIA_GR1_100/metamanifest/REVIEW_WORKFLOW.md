# Dedicated humanoid-video review workflow

The parent assigns exactly one `key` from `humanoid_world_model_review_jobs.json`. Work in `/Users/nijiachen/Downloads/Robotvideo`. Read the current project `AGENTS.md`, especially `humanoid_action_v1`, and the `finalize()` function in `review_humanoid_world_models.py` for the required review fields. Review only your assigned video; do not edit shared files or another video's outputs.

1. Run `.venv/bin/python review_humanoid_world_models.py prepare --key KEY`. This returns the source video/initial PNG, target generation input directory, original task instruction, full frame count and temporary contact-sheet paths; use these returned paths without reading the whole shared job index. **Visually view every returned contact sheet** using `tools.view_image({path, detail: "original"})` and emit its image through the image helper. Reading filenames or metadata is not visual review. These sheets cover every second decoded frame across the complete original MP4.
2. Establish the full action and its actual ending, then inspect meaningful transitions and the final frame at **native resolution**. Run `.venv/bin/python review_humanoid_world_models.py native --key KEY --frames N N ...` and visually view those returned PNGs. Also view the native source initial PNG. Use enough native details to establish the grasp, object identity, hand roles, release or absence of release, and final state. Select frames to resolve these essential transitions, without a redundant frame-count quota; record genuinely hidden details as uncertainties. Batch independent image reads when useful.
3. Write the usable prompt alone to your job's `generation input/actiongeneration_prompt.txt`. Usually use 55–100 words, two or three clear action sentences plus the **exact** shared constraint from AGENTS.md. More detail than Franka is intentional: identify the anatomical active left/right arm or hand and explain distinct two-hand roles/coordination or interaction with a person when visible. Ground anatomy from visible attachment/body pose; image left/right alone is insufficient. If the side cannot be established, use an unambiguous visible-arm description and record uncertainty instead of guessing.
4. The original GR1 instruction is a starting point, but observed frames determine the object, destination and outcome. Some source clips can end before placement/release; never invent success or extra movements. Preserve original annotation files. Do not turn the prompt into a frame-by-frame transcript, include timestamps, describe irrelevant scene appearance, or add uncertainty commentary. Do not describe the clip or its review; phrase incomplete endings as visible action or held state. Keep essential contact, grasp, transport, release, folding, pressing, tool use, coordinated support and locomotion when they distinguish the task.
5. Write `generation input/metadata/action_review.json` containing:
   - `prompt_protocol`: `humanoid_action_v1`
   - `prompt_provenance`: `visually_authored_from_video_and_initial_frame`
   - `reviewed_contact_sheets`: filenames of **all sheets actually viewed**
   - `inspected_frame_stride`: 2
   - `inspected_sampled_frame_count`: exact sample count in temporary inspection.json
   - `observed_sequence`: chronological actions with inclusive source frame intervals and confidence
   - `arm_hand_roles`: explicit roles of the relevant left/right hands (include idle/supporting hand as evidence when useful)
   - `anatomical_side_evidence`: body/arm evidence for anatomical naming, including uncertainty
   - `terminal_state`: the actual visible ending
   - `uncertainties`: list, empty when none
   - `annotation_discrepancies`: observed contradictions with the original instruction, empty when none
   - `initial_scene`, `evidence_frame_indices`, and `selected_execution_status` (success, failure, incomplete, or uncertain, grounded in the actual clip)
6. Run `.venv/bin/python review_humanoid_world_models.py finalize --key KEY`. It validates complete inspection coverage, original-byte reference and PNG checksums, prompt protocol and evidence fields; creates generation metadata using the sole `actiongeneration_prompt.txt` file. Fix real validation failures and rerun. Do not fabricate inspection coverage to pass validation.

The complete original video and first-frame PNG are already staged in generation input/. **No cropping, trimming, spatial resizing, re-encoding, first_substep branches or new media are requested.** Temporary inspection thumbnails are not deliverables; do not save an evidence-frame directory under generation input/.

Return the key, final prompt, visual coverage and material uncertainties. You are the dedicated reviewer for this one video.
