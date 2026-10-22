"""Build the evaluation review site for the world-model evaluation sets (PhysicsLENS, MiraBench, RoboJudge-Bench)
into site/eval/, published with the dataset viewer at /robotvideo/eval/.

Usage: .venv/bin/python scripts/build_eval_site.py [--jobs N] [--out DIR]

The site is public: it names the ERIS location by path only (ERIS_PATH), never the host or user, and links to the
unchanged files on Hugging Face rather than to the local dataset folders (which are not in the repository). It reuses
the dataset viewer's helpers, stylesheet and player script (copied into <out>/assets/). The home page states the project's own metrics and the
proposed mapping from each dataset's human annotations to them, and where the full filtered sets are on ERIS
(metamanifest/existing_annotations_eris.json and existing_annotations_manifest.csv, written after
scripts/stage_existing_annotations.py was uploaded). Each dataset page is its signal page: only the error groups of
<dataset>/metamanifest/<group CSV>, whose videos are the only ones kept locally. Each video page shows the full human
annotation.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from build_site import PLAYER, ROOT, esc, fmt_num, probe, stale, transcode

ERIS_PATH = "~/scratch/existing_annotations"  # on the ERIS cluster; host and user are deliberately not published
ERIS_JSON = ROOT / "metamanifest" / "existing_annotations_eris.json"
ERIS_MANIFEST = ROOT / "metamanifest" / "existing_annotations_manifest.csv"
EVAL_DATASETS = {
    "PhysicsLENS": dict(
        title="PhysicsLENS", kind="Physical plausibility of image-to-video generation",
        source="https://huggingface.co/datasets/swiftrando/PhysicsLENS", upstream_total=440, groups="label_groups.csv",
        license="Prompts and annotations CC BY-NC-SA 4.0; frames keep their source licenses",
        order=["Wan2.2 TI2V-5B", "Cosmos3 Nano", "HunyuanVideo 1.5", "MAGI-1 4.5B distill"],
        signal="Contact & Collision · Momentum · Deformation",
        eris_filter="Every generated video that has a human annotation (observable and hidden-property prompts); the real "
                    "source videos, the upscaled_720p copies and unannotated videos are left out.",
        selection="PhysicsLENS/scripts/fetch_label_groups.py"),
    "MiraBench": dict(
        title="MiraBench", kind="Action-conditioned humanoid world models",
        source="https://huggingface.co/datasets/Anonymous-nips-submissions/Anonymous-nips-submissions",
        upstream_total=906, groups="label_groups.csv",
        license="Annotations CC BY 4.0; videos are third-party model outputs",
        order=["DreamDojo-14B", "DreamDojo-2B", "Happy Horse (i2v)", "Wan2.1-14B (i2v)"],
        signal="Interaction D · Interpenetration D · Appearance D",
        eris_filter="Physical-consistency videos only (not action-following), duration ≤ 12.1 s: the 12.1 s (12.125 s) "
                    "DreamDojo-14B pour20 videos are kept; the 24.1 s DreamDojo-14B/2B occ68 videos are not.",
        selection="MiraBench/scripts/fetch_label_groups.py"),
    "RoboJudge_Bench": dict(
        title="RoboJudge-Bench", kind="Human-rated benchmark for embodied-video judges",
        source="https://huggingface.co/datasets/HuggingFriends/RoboJudge-Bench", upstream_total=800,
        groups="agent_integrity_groups.csv",
        license="CC BY-NC 4.0 (dataset card); videos keep their source and generator terms",
        order=["ABot-PhysWorld", "CogVideoX I2V", "Cosmos 3", "GigaWorld GR1 2B", "Happy Horse", "Kling", "Veo 3.1 Lite",
               "Wan2.2 TI2V-5B", "Wan 2.6 Flash", "WoW Wan 14B"],
        signal="Agent integrity 0 · Agent integrity 1",
        eris_filter="Robot first frames only: AgiBot World, RoboTwin, GR1 in-lab, Open X-Embodiment and DROID. The "
                    "human-hand corpora (EgoDex-human, EgoScaler, EPIC-KITCHENS) are left out.",
        selection="RoboJudge_Bench/scripts/fetch_agent_integrity_groups.py"),
}
# The project's own metrics, and the proposed comparison with each dataset's human annotations.
METRICS = [
    ("solidity", "Solidity", "Object interpenetration / collision violations."),
    ("contact", "Contact-mediated interaction", "Physically correct contact and resulting motion. Specifically for picking "
                                                 "behaviour; not applicable to cloth folding."),
    ("persistence", "Object persistence", "Object identity, continuity, and disappearance."),
    ("robot_deformation", "Robot deformation", "Unnatural deformation of rigid robot links."),
    ("object_deformation", "Object deformation", "Unrealistic deformation of manipulated objects."),
]
# (our metric keys, dataset, error groups, annotation items [(name, zh, definition or anchor)], scale, note)
MAPPING = [
    (["solidity"], "MiraBench", ("interpenetration_d",),
     [("Interpenetration (objects passing through each other)", "穿模检测",
       "Example answer: Slight — surface overlap for 1–2 frames (微穿模：1-2 帧极短暂的表面级重叠，可能是插值问题) → B · 0.67")],
     "Graded A–D: A 1 · B 0.67 · C 0.33 · D 0. Higher score, lower error.",
     "General, not a clean solidity label: it covers any two solid objects, so robot–object (R-O) and "
     "object–environment (O-E) penetration are mixed together."),
    (["contact"], "PhysicsLENS", ("momentum",),
     [("Momentum", "", "An object accelerates or changes motion without a supporting interaction, including apparent "
                       "creation or loss of energy.")],
     "Binary per video: 1 = label selected (a violation).",
     "Primary comparison for the contact score: it asks whether the object's motion is supported by an interaction."),
    (["contact"], "PhysicsLENS", ("contact_collision",),
     [("Collision", "", "Objects pass through one another or fail to exhibit an expected response after impact."),
      ("Contact", "", "An object floats without support, rests at an incorrect contact point, or fails to maintain "
                      "visible contact.")],
     "Binary per video: 1 = label selected (a violation). An unselected label is not a confirmation of correctness.",
     "The paper merges Contact into Collision in all analyses. The local group keeps only rigid manipulated objects "
     "(no cloth), matching the picking scope of the contact metric."),
    (["contact"], "MiraBench", ("interaction_any_d",),
     [("Action has a visible effect", "动作效果存在性",
       "Example answer: Yes — clear physical response matching the action (是：被操作物有清晰可见的物理响应，符合动作力度) → A · 1"),
      ("Contact precedes motion (causality)", "接触时序因果性",
       "Example answer: No violation — object moves only after contact (是（无违规）：接触后才开始运动，因果时序正确) → A · 1"),
      ("Response direction matches applied force", "响应方向合理性",
       "Example answer: Clear — 60–90° off (明显违规：方向偏差 60-90°，方向明显不对) → C · 0.33")],
     "Graded A–D: A 1 · B 0.67 · C 0.33 · D 0. Higher score, lower error.",
     "Compare the three items with the contact score together (for example their mean) or each separately."),
    (["object_deformation"], "PhysicsLENS", ("deformation",),
     [("Deformation", "", "A rigid object warps or melts, or a soft object fails to deform under an interaction that "
                          "should alter its shape.")],
     "Binary per video: 1 = label selected (a violation).",
     "The label does not say which object deformed; check that it is the manipulated object, not the robot."),
    (["object_deformation", "persistence"], "MiraBench", ("appearance_d",),
     [("Colour stability", "颜色稳定性", "Severe: colour changes completely (e.g. red to blue) or hue shift >45° throughout "
                                       "(严重违规：颜色完全改变（如红变蓝），或全段持续色偏幅度 >45°) → D · 0"),
      ("Shape plausibility for the material", "形状/材质合理性",
       "Severe: deformation violates the material — metal squashed like rubber, cloth moves rigidly "
       "(严重违规：形变严重违反材质性质（如金属被压扁橡皮化、布料像刚体一样运动）) → D · 0"),
      ("Size stability", "尺寸稳定性", "Severe: area change >60%, or the object shrinks and grows suddenly "
                                     "(严重违规：面积变化 >60%，或物体突然从大变小再变大) → D · 0"),
      ("Material behaviour consistency", "材质行为一致性",
       "Severe: behaviour violates the material — solid liquefies, cloth does not deform on contact "
       "(严重违规：行为完全违反材质性质（如固体液化、布料碰撞后完全无形变）) → D · 0")],
     "Graded A–D: A 1 · B 0.67 · C 0.33 · D 0. Higher score, lower error.",
     "These items rate the manipulated (primary) object, not the robot (MiraBench Protocol A)."),
    (["robot_deformation"], "RoboJudge_Bench", ("agent_integrity_0", "agent_integrity_1"),
     [("Agent integrity (agent_consistency)", "主体一致性",
       "The arm/gripper/hand stays structurally complete and consistent (no melting, fused/extra fingers, warping).")],
     "0–2 per video: 2 fully met, 1 partly met / minor problem, 0 clear violation. Higher score, lower error.",
     "Both label versions are kept in the annotation data: v1 = the authors' review-repo file "
     "test_final_800_20260916.jsonl, v2 = the released data/test.jsonl (2026-09-29). They agree on agent integrity "
     "for 443 of 800 videos."),
]
METRIC_HUE = {"solidity": 205, "contact": 158, "persistence": 182, "robot_deformation": 235, "object_deformation": 128}  # muted, around the site accent
ITEM_SHORT = {"Interpenetration (objects passing through each other)": "Interpenetration",
              "Action has a visible effect": "Action effect", "Contact precedes motion (causality)": "Contact before motion",
              "Response direction matches applied force": "Response direction", "Colour stability": "Colour",
              "Shape plausibility for the material": "Shape for material", "Size stability": "Size",
              "Material behaviour consistency": "Material behaviour", "Agent integrity (agent_consistency)": "Agent integrity"}
SCALE_SHORT = {"PhysicsLENS": "0/1 flag · 1 = violation", "MiraBench": "A–D → 1 · 0.67 · 0.33 · 0",
               "RoboJudge_Bench": "0–2 · higher is better"}
PRIMARY = {"momentum": "Primary comparison for our contact score"}
# Rows of the home-page mapping table: one row per metric, holding every dataset compared with it.
METRIC_ROWS = [("solidity",), ("contact",), ("object_deformation", "persistence"), ("robot_deformation",)]
# Short button labels for the error groups.
GROUP_SHORT = {"contact_collision": "Contact & Collision", "momentum": "Momentum", "deformation": "Deformation",
               "interaction_any_d": "Interaction", "interpenetration_d": "Interpenetration", "appearance_d": "Appearance",
               "agent_integrity_0": "Agent integrity 0", "agent_integrity_1": "Agent integrity 1"}
SEVERITY = {"A": "none", "B": "minor", "C": "clear", "D": "severe"}
RJB_SUBS = [("agent_consistency", "physical_adherence", "Agent integrity"),
            ("scene_consistency", "physical_adherence", "Scene & object consistency"),
            ("interaction_realism", "physical_adherence", "Interaction realism"),
            ("agent_match", "instruction_alignment", "Agent match"),
            ("object_correct", "instruction_alignment", "Object correctness"),
            ("goal_completed", "instruction_alignment", "Goal completion")]
RJB_SHORT = {"agent_consistency": "Agent int.", "scene_consistency": "Scene", "interaction_realism": "Interact.",
             "agent_match": "Agent match", "object_correct": "Object", "goal_completed": "Goal"}
RJB_CODE_NOTE = ("Codes (not published; implied by the authors' label-review files, which sum the sub-scores and lower them "
                 "with the main score): 0 = criterion not met / clear violation, 1 = partly met / minor problem, 2 = fully met. "
                 "v2 (released data/test.jsonl) differs widely from v1 (the authors' earlier final file, "
                 "test_final_800_20260916.jsonl) on the sub-scores, while PA and IA mostly agree.")
RJ_PROJECT = "https://siyuanmacs.github.io/RoboJudge/"
RJ_SCALES = {  # quoted from the RoboJudge judge prompts
    "physical_adherence": {"1": "gross violations throughout", "2": "major violations", "3": "noticeable local inconsistencies",
                           "4": "minor issues only", "5": "no visible violation"},
    "instruction_alignment": {"1": "unrelated or task not performed", "2": "major misalignment", "3": "partial completion",
                              "4": "minor shortfalls only", "5": "full, correct completion"},
}
RJ_CRITERIA = {  # quoted from the judge prompts
    "physical_adherence": [
        ("Agent integrity", "the arm/gripper/hand stays structurally complete and consistent (no melting, fused/extra fingers, warping)."),
        ("Scene & object consistency", "background and objects stay temporally stable (no flicker, teleport, morphing, appear/disappear)."),
        ("Interaction realism", "contacts obey physics (grasps close and bear weight, no interpenetration, motion respects gravity/inertia).")],
    "instruction_alignment": [
        ("Agent match", "the task is done by the SAME manipulator shown in the first frame (not a different/new agent)."),
        ("Object correctness", "the manipulated object is the instruction's target object."),
        ("Goal completion", "the instructed goal is actually achieved by the end (not merely approached).")],
}
PL_PAPER = "https://arxiv.org/abs/2610.01162"
CODEBOOK = json.loads((ROOT / "PhysicsLENS" / "metamanifest" / "annotation_codebook.json").read_text(encoding="utf-8"))
PL_SHORT = {"Causality": "Caus", "Collision": "Coll", "Contact": "Cont", "Deformation": "Def", "Fluid": "Fluid",
            "Friction": "Fric", "Gravity": "Grav", "Momentum": "Mom"}
PL_GROUP_LABELS = {"contact_collision": ["Contact", "Collision"], "momentum": ["Momentum"], "deformation": ["Deformation"]}
MIRA_PAPER = "https://arxiv.org/abs/2605.29360"
MIRA_GROUP_CODES = {"interaction_any_d": ["IC_0", "IC_1", "IC_2", "IC_3"], "interpenetration_d": ["IC_0", "IC_1", "IC_2", "IC_3"],
                    "appearance_d": ["SC_A1", "SC_A2", "SC_A3", "SC_A4"]}
# Scope of MiraBench's questionnaire families, from the paper's Protocol A (Appendix G.1) and the item wording.
FAMILY_NOTES = {
    "Appearance": "Scope: the manipulated (primary) object, not the robot. The colour item asks about “主要物体（被操作物）” "
                  "(the main, manipulated object); Protocol A asks whether “the primary object” keeps its colour, shape, size "
                  "and material behaviour, and the paper's automated version says “the item being grasped or moved by the robot "
                  "arm (not the arm itself)”. No item rates the robot's own appearance.",
    "Motion": "Scope: the manipulated object's speed and direction (“Is the object's velocity smooth…”), not the robot's motion.",
    "Occlusion": "Scope: the manipulated object's identity and path when hidden by the robot arm or another object.",
    "Interaction": "Robot–object interaction: whether the robot's action moves the target, contact precedes motion, response "
                   "direction, and interpenetration of any two solid objects.",
    "Environment": "Background, lighting and the non-manipulated objects in the scene.",
}
EVAL_CSS = """/* Evaluation review site: keep cards inside their grid column; questionnaire tables; metric mapping */
html { font-size: 112.5%; }
.card { min-width: 0; }
.card__title { overflow-wrap: anywhere; }
.anno--wide { grid-column: 1 / -1; }
.chrono.indicators tbody th { white-space: normal; width: 34%; }
.chrono .zh, .kv .zh { display: block; margin-top: 2px; font-size: 0.8125rem; color: var(--ink-3); font-weight: 400; }
.indicators td.grade { white-space: nowrap; width: 7rem; color: var(--ink-2); }
.tags { display: flex; flex-wrap: wrap; gap: 4px; }
.tag { font: 500 0.6875rem/1 var(--sans); padding: 0.3em 0.55em; border-radius: 3px; background: var(--fail-bg); color: var(--fail); }
.tag--muted { background: var(--paper-2); color: var(--ink-2); }
.scores { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 17rem), 1fr)); gap: var(--s5); }
.score h4 { display: flex; justify-content: space-between; gap: var(--s3); margin: 0 0 var(--s2); color: var(--ink); }
.score h4 b { font: 500 1.25rem/1 var(--serif); }
.scale { list-style: none; margin: 0; padding: 0; display: grid; gap: 3px; font-size: 0.8125rem; }
.scale li { padding: 0.35em 0.6em; border-radius: 3px; color: var(--ink-3); background: var(--paper-2); }
.scale li[aria-current] { color: var(--ink); background: var(--exception-bg); font-weight: 500; }
.scale li[aria-current].lvl-hi { background: var(--pass-bg); }
.scale li[aria-current].lvl-lo { background: var(--fail-bg); }
.norm { font: 400 0.75rem var(--mono); color: var(--ink-2); margin-top: var(--s2) !important; }
.flag { display: inline-block; min-width: 1.6em; text-align: center; font: 600 0.75rem/1.6 var(--mono); border-radius: 3px; background: var(--paper-2); color: var(--ink-3); }
.flag--on { background: var(--fail); color: var(--paper); }
.chrono.labels tbody th { width: 9rem; }
.chrono.labels tr.on th, .chrono.labels tr.on td { color: var(--ink); }
.chrono.labels tr:not(.on) td:last-child { color: var(--ink-3); }
.chrono.labels tr.merged th { font-style: italic; }
.matrix-wrap { overflow-x: auto; }
.chrono.matrix { font-size: 0.8125rem; }
.chrono.matrix th, .chrono.matrix td { padding-right: var(--s2); white-space: nowrap; }
.chrono.matrix tbody th { width: auto; }
.chrono.matrix td.c { text-align: center; }
.codebook dl.kv { margin-top: var(--s3); font-size: 0.9375rem; }
.codebook h3 { font: 600 0.6875rem/1 var(--sans); letter-spacing: 0.13em; text-transform: uppercase; color: var(--ink-3); margin: var(--s5) 0 var(--s2); }
.codebook ol { margin: var(--s1) 0 0; padding-left: 1.4em; }
.chips { display: inline-flex; flex-wrap: wrap; gap: 4px; align-self: start; }
.family-note { font-size: 0.8125rem; color: var(--ink-2); margin: calc(-1 * var(--s2)) 0 var(--s3); max-width: 75ch; }
.home-section { margin-top: var(--s7, 4rem); }
.home-section > h2 { font: 500 1.5rem/1.2 var(--serif); margin: 0 0 var(--s3); }
.home-section > p { max-width: 80ch; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr)); gap: var(--s4); margin-top: var(--s4); }
.metric { border: 1px solid var(--rule); border-radius: 6px; padding: var(--s4); }
.metric h3 { margin: 0 0 var(--s2); font: 600 1rem/1.3 var(--sans); }
.metric p { margin: 0; font-size: 0.875rem; color: var(--ink-2); }
.metric ul { margin: var(--s3) 0 0; padding-left: 1.1em; font-size: 0.8125rem; }
.chrono.mapping { margin-top: var(--s4); }
.chrono.mapping tbody th { width: 12rem; white-space: normal; color: var(--ink); }
.chrono.mapping td { font-size: 0.875rem; }
.chrono.mapping td.items { min-width: 22rem; }
.chrono.mapping dl { margin: 0; }
.chrono.mapping dt { font-weight: 600; color: var(--ink); }
.chrono.mapping dd { margin: 0 0 var(--s2); color: var(--ink-2); }
.chrono.mapping .note { display: block; margin-top: var(--s2); color: var(--ink-3); font-size: 0.8125rem; }
.server { border: 1px solid var(--rule); border-radius: 6px; padding: var(--s5); margin-top: var(--s5); background: var(--paper-2); }
.server h2 { margin-top: 0; }
.server pre { font: 400 0.8125rem/1.5 var(--mono); overflow-x: auto; background: var(--paper); padding: var(--s3); border-radius: 4px; }
.server code, .specs code { font: 400 0.8125rem var(--mono); overflow-wrap: anywhere; }
.signal-link { font-weight: 600; display: block; margin-top: var(--s2); }
.cmp { display: grid; grid-template-columns: 9rem minmax(18rem, 1.4fr) minmax(14rem, 1fr); gap: var(--s4); padding: var(--s3) 0; }
.cmp + .cmp { border-top: 1px dashed var(--rule); }
.cmp__ds { font-weight: 600; color: var(--ink); }
.cmp__ds .target { display: block; margin-top: var(--s1); font-weight: 400; font-size: 0.8125rem; color: var(--ink-3); }
.cmp__items { margin: 0; }
.chrono.mapping > tbody > tr > td { padding-top: 0; padding-bottom: 0; }
.routes a { flex-direction: column; gap: 0.35em; min-height: 6rem; }
.routes a b { font-weight: 600; }
.routes a small { font: 400 0.8125rem/1.3 var(--sans); color: var(--ink-3); }
.routes a[aria-current="page"] small { color: var(--paper); opacity: 0.8; }
.routes + .specs { margin-top: var(--s6); }
.intro dl.kv { margin-top: var(--s4); }
/* Home page: one colour per metric */
.m-solidity { --m: oklch(48% 0.075 205); --m-bg: oklch(96.3% 0.014 205); --m-line: oklch(85% 0.035 205); }\n.m-contact { --m: oklch(48% 0.075 158); --m-bg: oklch(96.3% 0.014 158); --m-line: oklch(85% 0.035 158); }\n.m-persistence { --m: oklch(48% 0.075 182); --m-bg: oklch(96.3% 0.014 182); --m-line: oklch(85% 0.035 182); }\n.m-robot_deformation { --m: oklch(48% 0.075 235); --m-bg: oklch(96.3% 0.014 235); --m-line: oklch(85% 0.035 235); }\n.m-object_deformation { --m: oklch(48% 0.075 128); --m-bg: oklch(96.3% 0.014 128); --m-line: oklch(85% 0.035 128); }
.mdot { display: inline-block; width: 0.7em; height: 0.7em; border-radius: 50%; background: var(--m, var(--ink-3)); margin-right: 0.45em; vertical-align: 0.05em; flex: none; }
.hero { padding: var(--s7) 0 var(--s5); max-width: 62rem; }
.hero h1 { font: 400 clamp(2.2rem, 5vw, 3.6rem)/1.05 var(--serif); margin: var(--s2) 0 var(--s4); letter-spacing: -0.01em; }
.hero .lede { max-width: 58ch; }
.jump { display: flex; flex-wrap: wrap; gap: var(--s2); margin-top: var(--s5); }
.jump a { text-decoration: none; color: var(--ink); font: 500 0.875rem/1 var(--sans); padding: 0.6em 0.95em; border: 1px solid var(--rule); border-radius: 999px; background: var(--paper); transition: background-color 0.2s var(--ease), border-color 0.2s var(--ease); }
.jump a:hover { background: var(--paper-2); border-color: var(--ink-3); }
.home-section { margin-top: var(--s7); scroll-margin-top: var(--s5); }
.home-section > h2 { font: 400 1.75rem/1.2 var(--serif); margin: 0 0 var(--s4); padding-bottom: var(--s2); border-bottom: 1px solid var(--rule); }
table.glance { width: 100%; border-collapse: separate; border-spacing: 0 6px; font-size: 0.9375rem; }
.glance thead th { text-align: left; font: 600 0.75rem/1.2 var(--sans); letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink-2); padding: 0 var(--s3) var(--s1); }
.glance thead th span { display: block; margin-top: 3px; font: 400 0.8125rem var(--mono); letter-spacing: 0; text-transform: none; color: var(--ink-3); }
.glance tbody th, .glance tbody td { background: var(--m-bg); padding: var(--s3); vertical-align: top; }
.glance tbody th { width: 30%; text-align: left; font: 600 1rem/1.3 var(--sans); color: var(--ink); border-left: 4px solid var(--m); border-radius: 6px 0 0 6px; }
.glance tbody td:last-child { border-radius: 0 6px 6px 0; }
.glance tbody th small { display: block; margin-top: 4px; font: 400 0.875rem/1.4 var(--sans); color: var(--ink-2); }
.glance .none { color: var(--ink-3); }
.ichip { display: inline-block; margin: 0 4px 4px 0; padding: 0.3em 0.6em; border-radius: 999px; font: 500 0.8125rem/1.2 var(--sans); color: var(--ink); background: var(--paper); border: 1px solid var(--m-line, var(--rule)); text-decoration: none; transition: background-color 0.2s var(--ease), border-color 0.2s var(--ease); }
a.ichip:hover { border-color: var(--m, var(--ink)); background: var(--paper-2); }
.ichip--primary { background: var(--m, var(--ink)); border-color: var(--m, var(--ink)); color: var(--paper); font-weight: 600; }
a.ichip--primary:hover { background: var(--m); color: var(--paper); filter: brightness(1.08); }
.legend { margin-top: var(--s2); font-size: 0.875rem; color: var(--ink-3); }
.legend .ichip--primary { background: var(--accent); border-color: var(--accent); }
details.more { margin-top: var(--s3); }
details.more > summary { cursor: pointer; list-style: none; display: inline-flex; align-items: center; gap: 0.4em; font: 500 0.875rem/1 var(--sans); color: var(--ink-2); padding: 0.45em 0.75em; border: 1px solid var(--rule); border-radius: 999px; background: var(--paper); }
details.more > summary::-webkit-details-marker { display: none; }
details.more > summary::before { content: "+"; font-weight: 600; color: var(--m, var(--accent)); }
details.more[open] > summary::before { content: "−"; }
details.more > summary:hover { border-color: var(--ink-3); }
details.more[open] > :not(summary) { animation: reveal 0.25s var(--ease); }
details.more dl { margin: var(--s3) 0 0; font-size: 0.875rem; }
details.more dt { font-weight: 600; color: var(--ink); }
details.more dd { margin: 0 0 var(--s2); color: var(--ink-2); }
details.more p { font-size: 0.875rem; color: var(--ink-2); max-width: 75ch; }
details.more .caveat { color: var(--ink); border-left: 3px solid var(--m, var(--exception)); padding-left: var(--s2); }
details.scales { margin: 0 0 var(--s4); }
@keyframes reveal { from { opacity: 0; transform: translateY(-3px); } to { opacity: 1; transform: none; } }
.panel { border: 1px solid var(--m-line); border-top: 4px solid var(--m); border-radius: 8px; padding: var(--s4) var(--s5) var(--s5); margin-bottom: var(--s5); background: linear-gradient(180deg, var(--m-bg), var(--paper) 7rem); }
.panel__head { display: flex; flex-wrap: wrap; align-items: baseline; gap: var(--s1) var(--s3); margin-bottom: var(--s4); }
.panel__head h3 { margin: 0; font: 500 1.5rem/1.2 var(--serif); }
.panel__head p { margin: 0; font-size: 0.9375rem; color: var(--ink-2); flex-basis: 100%; }
.panel__head .mdot { width: 0.85em; height: 0.85em; }
.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 19rem), 1fr)); gap: var(--s4); }
.tile { background: var(--paper); border: 1px solid var(--rule); border-radius: 8px; padding: var(--s4); display: flex; flex-direction: column; gap: var(--s2); transition: border-color 0.2s var(--ease), box-shadow 0.2s var(--ease); }
.tile:hover { border-color: var(--m-line); box-shadow: 0 2px 10px oklch(30% 0.03 60 / 0.06); }
.tile--primary { border: 2px solid var(--m); box-shadow: 0 0 0 4px var(--m-bg); }
.tile__head { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2); }
.dsbadge { font: 600 0.8125rem/1 var(--sans); letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink); background: var(--paper-2); padding: 0.4em 0.6em; border-radius: 4px; }
.pbadge { font: 600 0.8125rem/1 var(--sans); color: var(--paper); background: var(--m); padding: 0.4em 0.6em; border-radius: 4px; }
.tile__target { font-size: 0.8125rem; color: var(--ink-3); flex-basis: 100%; }
.tile__scale { margin: 0; font: 400 0.8125rem var(--mono); color: var(--ink-2); }
.tile__links { margin-top: auto; display: flex; flex-wrap: wrap; gap: var(--s2); padding-top: var(--s2); }
.btn { display: inline-block; text-decoration: none; font: 600 0.875rem/1 var(--sans); color: var(--m, var(--accent)); padding: 0.55em 0.8em; border: 1.5px solid var(--m-line, var(--rule)); border-radius: 6px; background: var(--paper); transition: background-color 0.2s var(--ease), color 0.2s var(--ease); }
.btn:hover { background: var(--m); color: var(--paper); border-color: var(--m); }
.dscards { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 22rem), 1fr)); gap: var(--s5); }
.dscard { border: 1px solid var(--rule); border-radius: 10px; overflow: hidden; background: var(--paper); display: flex; flex-direction: column; transition: box-shadow 0.2s var(--ease), transform 0.2s var(--ease); }
.dscard:hover { box-shadow: 0 6px 22px oklch(30% 0.03 60 / 0.09); transform: translateY(-2px); }
.dscard__strip { display: grid; grid-template-columns: repeat(4, 1fr); }
.dscard__strip img { aspect-ratio: 4 / 3; object-fit: cover; width: 100%; }
.dscard__body { padding: var(--s4) var(--s5) var(--s5); }
.dscard__body h3 { margin: 0; font: 500 1.5rem/1.2 var(--serif); }
.dscard__body h3 a { color: var(--ink); text-decoration: none; }
.dscard__body .quiet { margin: var(--s1) 0 var(--s3); font-size: 0.875rem; }
.dscard__stats { font-size: 0.875rem; color: var(--ink-2); margin: 0 0 var(--s3); }
.gbtns { display: flex; flex-wrap: wrap; gap: var(--s2); }
.gbtn { display: inline-flex; align-items: center; text-decoration: none; font: 500 0.875rem/1 var(--sans); color: var(--ink); padding: 0.55em 0.8em; border-radius: 999px; background: var(--m-bg); border: 1px solid var(--m-line); transition: background-color 0.2s var(--ease), color 0.2s var(--ease); }
.gbtn:hover { background: var(--m); color: var(--paper); }
.gbtn:hover .mdot { background: var(--paper); }
.server { border: 1px solid var(--rule); border-radius: 10px; padding: var(--s5); background: var(--paper-2); }
.server h2 { border-bottom: 0 !important; padding-bottom: 0 !important; }
.server__top { display: flex; flex-wrap: wrap; justify-content: space-between; gap: var(--s4); align-items: flex-start; }
.server__path { margin: 0; font-size: 1.125rem; }
.server__path code { font: 500 1.125rem var(--mono); background: var(--paper); padding: 0.3em 0.55em; border-radius: 5px; border: 1px solid var(--rule); }
.pills { display: flex; flex-wrap: wrap; gap: var(--s2); }
.pill { font-size: 0.8125rem; padding: 0.45em 0.8em; border-radius: 999px; background: var(--paper); border: 1px solid var(--rule); }
.pill b { font: 600 0.9375rem var(--sans); margin-right: 0.2em; }
.server__grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 15rem), 1fr)); gap: var(--s3); margin: var(--s4) 0 var(--s2); }
.fact { background: var(--paper); border-radius: 6px; padding: var(--s3) var(--s4); border: 1px solid var(--rule); }
.fact h4 { margin: 0 0 var(--s1); font: 600 0.6875rem/1 var(--sans); letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3); }
.fact p { margin: 0; font-size: 0.875rem; }
.routes a { border-left: 5px solid var(--m); }
.routes a[aria-current="page"] { background: var(--m); border-color: var(--m); }
.routes a[aria-current="page"] .mdot { background: var(--paper); }
@media (prefers-reduced-motion: reduce) { .dscard, .tile, details.more[open] > :not(summary) { transition: none; animation: none; } .dscard:hover { transform: none; } }
"""
PLAYER_EVAL = (PLAYER.replace("Action generation prompt", "Generation prompt").replace("Publisher annotation", "Human annotation")
               .replace("Full annotation and files →", "Full human annotation and files →"))


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def task_title(task):
    """Readable title from a task string; Humanoid Everyday tasks are ids like 'deformable/squeeze_a_ball'."""
    task = task.rstrip(".")
    if "_" in task and " " not in task:
        task = task.split("/")[-1].split(".")[-1].replace("_", " ").capitalize()
    return task


def score_class(value):
    """Chip colour for a 1–4 or 1–5 human score: 4 and up pass, 3 middling, 2 and below fail."""
    return "pass" if value >= 4 else "exception" if value == 3 else "fail"


def eris_info():
    """The staged copy on ERIS: summary JSON and {upstream_path: manifest row}; empty when not staged yet."""
    summary = json.loads(ERIS_JSON.read_text(encoding="utf-8")) if ERIS_JSON.exists() else None
    rows = {}
    if ERIS_MANIFEST.exists():
        for r in csv.DictReader(open(ERIS_MANIFEST, encoding="utf-8")):
            rows[r["upstream_path"]] = r
    return summary, rows


def collect(ds, out):
    """Error-group videos of one dataset, one item per video folder, with the groups it belongs to."""
    in_groups, order = {}, []
    for r in csv.DictReader(open(ROOT / ds / "metamanifest" / EVAL_DATASETS[ds]["groups"], encoding="utf-8")):
        if r["folder"] not in in_groups:
            order.append(r)
        in_groups.setdefault(r["folder"], []).append(r)
    items = []
    for row in order:
        md = ROOT / ds / row["folder"] / "original" / "metadata"
        meta = json.loads((md / "metadata.json").read_text(encoding="utf-8"))
        human = json.loads((md / "human_annotation.json").read_text(encoding="utf-8"))
        raw = next((p for p in sorted(md.glob("*.json")) if p.name not in ("metadata.json", "human_annotation.json")), None)
        item = dict(dataset=ds, folder=row["folder"], meta=meta, human=human, raw_json=raw, video=ROOT / ds / row["video"],
                    generator=meta["generator"], page=out / ds / row["folder"], label_groups=in_groups[row["folder"]])
        if ds == "PhysicsLENS":
            p = human["physical_plausibility_1_4"]
            hidden = " · hidden-property prompt" if human["observability"] == "unobservable" else ""
            cats = f" · {', '.join(human['physics_category'])}" if human["physics_category"] else ""
            norm = human["normalized"]
            tags = [(lab, "") for lab in norm["label_order"]]
            tags.append(("task done" if norm["action_completed"] else "task not done", "muted"))
            if norm["hidden_property_followed_1_4"] is not None:
                tags.append((f"property followed {norm['hidden_property_followed_1_4']}/4", "muted"))
            item.update(tags=tags, group=meta["robot_group"], title=task_title(human["task"]), input=f"{meta['testset_id']:03d}",
                        sub=f"{meta['robot']} · #{meta['testset_id']:03d}{hidden}", prompt=meta["prompt"],
                        score=("Plausibility", f"{p}/4", score_class(p)),
                        summary=human["description_of_issue"] or "No issue described by the annotator.",
                        verdict=f"Plausibility {p}/4 · action {'completed' if human['action_completed'] == 'yes' else 'not completed'}{cats}")
        elif ds == "RoboJudge_Bench":
            pa, ia, sub = human["physical_adherence"], human["instruction_alignment"], human["sub_scores"]
            v1 = human["labels_20260916_review_repo"]
            instr = meta["instruction"] or ""
            item.update(group=meta["embodiment_group"], title=instr if len(instr) <= 90 else instr[:87].rstrip() + "…",
                        full_title=instr, input=f"{meta['source_corpus']}/{meta['task']}",
                        tags=[(f"{meta['prompt_variant']} prompt", "muted")], sub=f"{meta['embodiment']} · {meta['task']}",
                        prompt="", score=("Agent int. v2", str(sub["agent_consistency"]), "fail" if sub["agent_consistency"] == 0 else "exception"),
                        score2=("v1", str(v1["agent_consistency"]), "fail" if v1["agent_consistency"] == 0 else "exception"),
                        summary=f"PA {pa}/5 · IA {ia}/5 · sub-scores (0–2, v2): " + ", ".join(f"{label} {sub[key]}" for key, _, label in RJB_SUBS),
                        verdict=f"Agent integrity v2 {sub['agent_consistency']} · v1 {v1['agent_consistency']} · PA {pa}/5 · IA {ia}/5")
        else:
            head = human["headline"]
            v = int(float(head["score"]))
            flagged = [f"{i['label']} ({SEVERITY[i['grade']]})" for i in human["items"] if i["grade"] in ("C", "D")]
            item.update(group=meta["scene_group"], title=meta["input_id"].replace("_", " "), input=meta["input_id"],
                        sub=f"{meta['dimension']} · subset {meta['subset']}", prompt="", score=("Overall", f"{v}/5", score_class(v)),
                        summary=("Clear or severe: " + "; ".join(flagged)) if flagged else "No clear or severe violations marked.",
                        verdict=f"{head['label']}: {head['answer_en']}")
        items.append(item)
    return items


def page(title, rel, body, crumbs=()):
    trail = '<span aria-hidden="true">/</span>'.join(f'<a href="{esc(h)}">{esc(t)}</a>' for t, h in crumbs)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap">
<link rel="stylesheet" href="{rel}assets/style.css">
<link rel="stylesheet" href="{rel}assets/eval.css">
<script src="{rel}assets/app.js" defer></script>
</head>
<body>
<header class="masthead"><a class="wordmark" href="{rel}index.html">Robotvideo<span>evaluation review</span></a>
<nav class="crumbs">{trail}</nav></header>
<main>
{body}
</main>
<footer class="foot"><p>Error groups of three published world-model evaluation sets, compared with our physics metrics.
Previews are reduced H.264 copies; the unchanged videos are on Hugging Face (linked from each video page) and the full
filtered sets with every annotation are on ERIS scratch. Videos keep their source licenses.
<a href="{rel}../index.html">Dataset viewer</a></p></footer>
</body>
</html>
"""


def chips(i):
    return '<span class="chips">' + chip(i["score"]) + (chip(i["score2"]) if i.get("score2") else "") + "</span>"


def chip(score):
    label, value, cls = score
    return f'<span class="chip chip--{cls}">{esc(label)}<b>{esc(value)}</b></span>'


def card(i, d):
    base = quote(os.path.relpath(i["page"], d))
    name = f"{i['generator']} · {i['sub']}"
    return f"""<article class="card" data-gate="{slug(i['generator'])}" data-text="{esc((i['title'] + ' ' + name + ' ' + i['folder'] + ' ' + i['summary']).lower())}"
 data-title="{esc(i['title'])}" data-id="{esc(i['folder'])}" data-href="{base}/index.html" data-clip="{base}/clip.mp4"
 data-poster="{base}/frame.jpg" data-prompt="{esc(i['prompt'])}" data-prompt-missing="{esc(prompt_missing(i))}"
 data-publisher="{esc(i['verdict'] + '. ' + i['summary'])}">
<button type="button" class="card__img" aria-label="Play {esc(i['title'])} by {esc(i['generator'])}"><img src="{base}/poster.jpg" alt="" loading="lazy"><span class="card__play" aria-hidden="true"></span></button>
<span class="card__meta"><a class="card__title" href="{base}/index.html">{esc(i['title'])}</a>{chips(i)}</span>
<span class="card__id">{esc(name)}</span>{tag_line(i.get("tags"))}
<span class="card__text">{esc(i['summary'])}</span></article>"""


def prompt_missing(i):
    if i["dataset"] == "RoboJudge_Bench":
        return f"Instruction: {i['meta']['instruction']} (the exact {i['meta']['prompt_variant']} generation prompt is not released)."
    return "No text prompt: generated from the first frame and a robot action sequence."


def tag_line(tags):
    if not tags:
        return ""
    return '<span class="tags">' + "".join(f'<span class="tag{" tag--" + k if k else ""}">{esc(t)}</span>' for t, k in tags) + "</span>"


def ordinal(n):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def scale_list(levels, chosen):
    def cls(level):
        return "lvl-hi" if int(level) >= 4 else "lvl-lo" if int(level) <= 2 else ""
    current = ' aria-current="true"'
    return '<ol class="scale">' + "".join(
        f'<li class="{cls(lv)}"{current if chosen is not None and int(lv) == chosen else ""}>{lv} — {esc(text)}</li>'
        for lv, text in levels.items()) + "</ol>"


def kv(rows):
    return '<dl class="kv">' + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows) + "</dl>"


def kv_html(rows):
    return '<dl class="kv">' + "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>"


def physicslens_annotation(h, cb):
    """Scores on their official scales, all eight violation labels as 0/1 with the paper's definitions, the
    normalized record, the issue description and, for hidden-property prompts, the stated property."""
    n, f = h["normalized"], cb["fields"]
    p, a, hh = n["plausibility_1_4"], n["action_completed"], n["hidden_property_followed_1_4"]
    scores = [f'<div class="score"><h4>Physical plausibility (P)<b>{p} / 4</b></h4>{scale_list(f["physical_plausibility"]["levels"], p)}'
              f'<p class="norm">P01 = {n["plausibility_01"]:g} · plausible (P≥3) = {n["plausible"]}</p></div>',
              f'<div class="score"><h4>Action completion (A)<b>{"yes" if a else "no"}</b></h4>'
              f'<p class="quiet">{esc(f["action_completed"]["instructions"])}</p><p class="norm">A = {a}</p></div>']
    if hh is None:
        scores.append('<div class="score"><h4>Hidden-property adherence (H)<b>—</b></h4><p class="quiet">Not collected: '
                      'observable prompt. The paper does not treat the missing value as 0.</p></div>')
    else:
        scores.append(f'<div class="score"><h4>Hidden-property adherence (H)<b>{hh} / 4</b></h4>'
                      f'{scale_list(f["property_followed"]["levels"], hh)}'
                      f'<p class="norm">H01 = {n["hidden_followed_01"]:g} · followed (H≥3) = {n["hidden_followed"]}</p></div>')
    order = n["label_order"]
    rows = []
    for lab, definition in cb["labels"].items():
        on = n[lab.lower()]
        where = f" · listed {ordinal(order.index(lab) + 1)}" if on else ""
        rows.append(f'<tr class="{"on" if on else ""}"><th scope="row">{esc(lab)}</th>'
                    f'<td><span class="flag{" flag--on" if on else ""}">{on}</span>{esc(where)}</td><td>{esc(definition)}</td></tr>')
    m = n["collision_or_contact"]
    rows.append(f'<tr class="merged{" on" if m else ""}"><th scope="row">Collision ∪ Contact</th>'
                f'<td><span class="flag{" flag--on" if m else ""}">{m}</span></td>'
                f'<td>The paper merges Contact into Collision in all analyses (its seven domains have one collision family).</td></tr>')
    labels = (f'<table class="chrono labels"><thead><tr><th scope="col">Label</th><th scope="col">Selected</th>'
              f'<th scope="col">Definition (paper, Table 9)</th></tr></thead><tbody>{"".join(rows)}</tbody></table>')
    if not order:
        labels += f'<p class="quiet">{esc(cb["label_notes"][1])}</p>'
    if n["note"]:
        labels += f'<p class="quiet">Note: {esc(n["note"])}.</p>'
    cols = [("P", p), ("P01", f'{n["plausibility_01"]:g}'), ("P≥3", n["plausible"]), ("A", a),
            ("H", "—" if hh is None else hh), ("H01", "—" if hh is None else f'{n["hidden_followed_01"]:g}'),
            ("H≥3", "—" if hh is None else n["hidden_followed"])] + \
           [(lab, n[lab.lower()]) for lab in cb["labels"]] + [("Coll∪Cont", m)]
    record = ('<div class="matrix-wrap"><table class="chrono matrix"><thead><tr>' + "".join(f'<th scope="col">{esc(c)}</th>' for c, _ in cols)
              + '</tr></thead><tbody><tr>' + "".join(f'<td class="c">{esc(v)}</td>' for _, v in cols) + "</tr></tbody></table></div>")
    parts = [f'<section class="anno anno--wide"><h3>Scores</h3><div class="scores">{"".join(scores)}</div></section>',
             f'<section class="anno anno--wide"><h3>Violation labels · multi-select, 8 labels</h3>{labels}</section>',
             f'<section class="anno"><h3>Issue description</h3><p>{esc(h["description_of_issue"] or "None given (the form asks for one only when P<4).")}</p>'
             f'<dl class="kv"><dt>Annotator</dt><dd>{esc(h["annotator"])}</dd><dt>Prompt set</dt><dd>{esc(h["observability"])}</dd></dl></section>']
    if h["observability"] == "unobservable":
        hidden = kv([("Hidden property", f"{h['hidden_property']}: {h['hidden_property_value']}"),
                     ("Expected outcome", h["expected_outcome"]), ("Failure signature", h["failure_signature"])])
        parts.append(f'<section class="anno"><h3>Hidden-property prompt</h3>{hidden}</section>')
    parts.append(f'<section class="anno anno--wide"><h3>Normalized record</h3>{record}'
                 '<p class="quiet">P01 = (P−1)/3, H01 = (H−1)/3; thresholds P≥3 and H≥3 are the paper\'s. Label columns: '
                 '1 = selected by the annotator, 0 = not selected (not a confirmation of correctness).</p></section>')
    return parts


def robojudge_bench_annotation(h):
    """PA and IA on their scales, then the six sub-scores in both label versions (v2 released, v1 review repo)."""
    v1 = h["labels_20260916_review_repo"]
    scores = []
    for axis, title in (("physical_adherence", "Physical adherence (PA)"), ("instruction_alignment", "Instruction alignment (IA)")):
        v = h[axis]
        scores.append(f'<div class="score"><h4>{title}<b>{v} / 5</b></h4>{scale_list(RJ_SCALES[axis], v)}'
                      f'<p class="norm">v2 (released) {v} · v1 (review repo, 16 Sep) {v1[axis]}</p></div>')
    criteria = {c: t for axis in RJ_CRITERIA for c, t in RJ_CRITERIA[axis]}
    rows = "".join(
        f'<tr class="{"on" if h["sub_scores"][key] == 0 else ""}"><th scope="row">{esc(label)}<span class="zh">{esc(key)} · {"PA" if axis == "physical_adherence" else "IA"}</span></th>'
        f'<td><span class="flag{" flag--on" if h["sub_scores"][key] == 0 else ""}">{h["sub_scores"][key]}</span></td>'
        f'<td><span class="flag{" flag--on" if v1[key] == 0 else ""}">{v1[key]}</span>{" ≠" if v1[key] != h["sub_scores"][key] else ""}</td>'
        f'<td>{esc(criteria[label])}</td></tr>'
        for key, axis, label in RJB_SUBS)
    table = (f'<table class="chrono labels"><thead><tr><th scope="col">Sub-score</th><th scope="col">v2 released</th>'
             f'<th scope="col">v1 review repo</th><th scope="col">Criterion (judge prompt)</th></tr></thead><tbody>{rows}</tbody></table>'
             f'<p class="quiet">{esc(RJB_CODE_NOTE)}</p>')
    return [f'<section class="anno anno--wide"><h3>Scores</h3><div class="scores">{"".join(scores)}</div></section>',
            f'<section class="anno anno--wide"><h3>Diagnostic sub-scores · 0–2 · both label versions</h3>{table}</section>',
            f'<section class="anno"><h3>Instruction</h3><p>{esc(h["instruction"])}</p></section>']


def grade_text(item):
    score = None if item["score"] is None else f"{item['score']:g}"
    return " · ".join(x for x in (item["grade"], score) if x)


def annotation(i):
    """The human annotation: PhysicsLENS fields, RoboJudge-Bench scores, or the MiraBench questionnaire by family."""
    h = i["human"]
    if i["dataset"] == "RoboJudge_Bench":
        parts = robojudge_bench_annotation(h)
        intro = (f"Human ratings from the benchmark (<a href=\"{RJ_PROJECT}\">project page</a>: “800 human-annotated test "
                 "videos”), in both label versions: v2 = released data/test.jsonl, v1 = the authors' review-repo file of "
                 "16 Sep 2026. No rationales are released. Scales and criteria are the RoboJudge judge prompts.")
    elif i["dataset"] == "PhysicsLENS":
        parts = physicslens_annotation(h, CODEBOOK)
        intro = (f"One annotator's final record from the publisher's consolidated_annotations.csv (one annotation per clip; "
                 f"no inter-annotator agreement exists). Scales and label definitions are quoted from the "
                 f'<a href="{PL_PAPER}">PhysicsLENS paper</a>, Appendix B, Tables 8–9.')
    else:
        families = {}
        for it in h["items"]:
            families.setdefault(it["family"], []).append(it)
        parts = []
        for fam, its in families.items():
            body = "".join(f'<tr><th scope="row">{esc(it["label"])}<span class="zh">{esc(it["title_zh"])}</span></th>'
                           f'<td>{esc(it["answer_en"])}<span class="zh">{esc(it["answer_zh"])}</span></td>'
                           f'<td class="grade">{esc(grade_text(it))}</td></tr>' for it in its)
            note = f'<p class="family-note">{esc(FAMILY_NOTES[fam])}</p>' if fam in FAMILY_NOTES else ""
            parts.append(f'<section class="anno anno--wide"><h3>{esc(fam)}</h3>{note}<table class="chrono indicators"><thead><tr>'
                         f'<th scope="col">Indicator</th><th scope="col">Answer</th><th scope="col">Grade · score</th></tr></thead>'
                         f'<tbody>{body}</tbody></table></section>')
        intro = (f"{esc(h['dimension'])} questionnaire ({esc(h['dimension_zh'])}). The Chinese text is the annotation; English "
                 "labels and answers are this repository's gloss. Grades and scores come from the publisher's parsed CSV "
                 "(A 1 · B 0.67 · C 0.33 · D 0: higher score, lower error). "
                 f'Indicator scopes follow the <a href="{MIRA_PAPER}">MiraBench paper</a>, Appendix G.1 (Protocol A).')
    details = ""
    if i["raw_json"]:
        record = json.dumps(json.loads(i["raw_json"].read_text(encoding="utf-8")), indent=2, ensure_ascii=False)
        details = (f'<details class="review"><summary>Publisher record <span>{esc(i["raw_json"].name)}</span></summary>'
                   f"<pre>{esc(record)}</pre></details>")
    return f"""<section class="annotation"><h2>Human annotation</h2>
<p class="quiet">{intro} <a href="annotation.json">JSON</a></p>
<div class="annotation__grid">{''.join(parts)}</div>
{details}</section>"""


def stills(src, outdir):
    """frame.jpg (1280 px) and poster.jpg (560 px) from the first decoded frame of a generated video, plus init.jpg
    from the conditioning image when the video folder has prompt/init_frame.png (RoboJudge-Bench)."""
    init_png, init = src.parent / "prompt" / "init_frame.png", outdir / "init.jpg"
    if init_png.exists() and stale(init, init_png):
        outdir.mkdir(parents=True, exist_ok=True)
        im = Image.open(init_png).convert("RGB")
        im.thumbnail((1280, 1280), Image.LANCZOS)
        im.save(init, "JPEG", quality=88, optimize=True, progressive=True)
    frame, poster = outdir / "frame.jpg", outdir / "poster.jpg"
    if not stale(frame, src) and not stale(poster, src):
        return
    outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-frames:v", "1", "-vf",
                    "scale='min(1280,iw)':'min(1280,ih)':force_original_aspect_ratio=decrease", "-q:v", "3", str(frame)],
                   check=True)
    im = Image.open(frame).convert("RGB")
    im.thumbnail((560, 560), Image.LANCZOS)
    im.save(poster, "JPEG", quality=80, optimize=True, progressive=True)


def video_page(i, prev, nxt, eris):
    rel = "../" * len(Path(i["dataset"], i["folder"]).parts)
    ds, meta, info = EVAL_DATASETS[i["dataset"]], i["meta"], i["info"]
    if i["prompt"]:
        prompt = f'<section class="field"><h2>Generation prompt</h2><p class="prompt">{esc(i["prompt"])}</p></section>'
    else:
        cond = meta.get("conditioning") or (
            f"first frame and a {meta['prompt_variant']} prompt; the benchmark releases the instruction, not the exact generation prompt")
        prompt = f'<section class="field"><h2>Conditioning</h2><p>{esc(cond[0].upper() + cond[1:])}.</p></section>'
    groups = "".join(f'<p><a href="{rel}{i["dataset"]}/{g["group"]}/index.html">{esc(g["group_title"])}</a></p>'
                     for g in i["label_groups"])
    blocks = [f'<section class="field"><h2>Generator</h2><p>{esc(i["generator"])}</p>'
              f'<p class="quiet">{esc(meta.get("checkpoint") or meta.get("dimension") or meta.get("generator_folder"))}</p></section>',
              f'<section class="field"><h2>Error group</h2>{groups}</section>', prompt,
              f'<section class="field"><h2>Human verdict</h2><div class="gate">{chips(i)}<p>{esc(i["verdict"])}</p></div>'
              f'<p class="quiet">{esc(i["summary"])}</p></section>']
    if meta.get("instruction"):
        blocks.insert(2, f'<section class="field"><h2>Instruction</h2><p>{esc(meta["instruction"])}</p></section>')
    facts = [("Resolution", f"{info['width']} × {info['height']}"), ("Frames", fmt_num(info["frames"])),
             ("Frame rate", fmt_num(info["fps"], " fps")), ("Duration", fmt_num(info["duration"], " s")),
             ("Codec", info["codec"]), ("First frame", meta.get("robot") or meta.get("first_frame_view") or meta.get("embodiment"))]
    blocks.append('<section class="field"><h2>Clip</h2><dl class="facts">' +
                  "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in facts) + "</dl></section>")
    (i["page"] / "annotation.json").write_text(json.dumps(i["human"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    links = [("Human annotation (JSON)", "annotation.json"),
             ("Original video on Hugging Face (unchanged)", f"{ds['source']}/blob/{meta['revision']}/{quote(meta['upstream_path'])}")]
    files = ('<section class="field"><h2>Files</h2><ul class="links">' +
             "".join(f'<li><a href="{esc(h)}">{esc(t)}</a></li>' for t, h in links) + "</ul>")
    row = eris[1].get(meta["upstream_path"])
    if row:
        files += (f'<p class="quiet">On ERIS: <code>{esc(ERIS_PATH)}/{esc(row["video"])}</code>, annotation '
                  f'<code>{esc(row["annotation"])}</code>, record <code>{esc(row["json"])}</code></p>')
    blocks.append(files + "</section>")
    nav = '<nav class="pager">' + "".join(
        f'<a rel="{r}" href="{rel}{esc(quote(e["dataset"] + "/" + e["folder"]))}/index.html">{a}{esc(e["title"] + " · " + e["generator"])}{b}</a>'
        if e else "<span></span>" for e, r, a, b in ((prev, "prev", "← ", ""), (nxt, "next", "", " →"))) + "</nav>"
    still = (i["video"].parent / "prompt" / "init_frame.png").exists()
    body = f"""<article class="episode">
<header class="episode__head"><p class="kicker">{esc(ds['title'])} · {esc(i['label_groups'][0]['group_title'])}</p>
<h1>{esc(i.get('full_title') or i['title'])}</h1><p class="quiet">{esc(i['generator'] + ' · ' + i['sub'])}</p></header>
<div class="episode__grid">
<div class="episode__media">
<figure class="media"><video controls playsinline preload="metadata" poster="frame.jpg" src="clip.mp4"></video>
<figcaption>Generated video · {fmt_num(info['frames'])} frames · {fmt_num(info['duration'], ' s')}</figcaption></figure>
<figure class="media media--still"><img src="{'init.jpg' if still else 'frame.jpg'}" alt="First frame" loading="lazy">
<figcaption>{'Conditioning first frame (prompt/init_frame.png)' if still else 'First frame of the generated video'}</figcaption></figure>
</div>
<aside class="episode__side">{''.join(blocks)}</aside>
</div>
{annotation(i)}
{nav}
</article>"""
    first = i["label_groups"][0]["group"]
    crumbs = [("All sets", f"{rel}index.html"), (ds["title"], f"{rel}{i['dataset']}/index.html"),
              (GROUP_SHORT.get(first, first), f"{rel}{i['dataset']}/{first}/index.html")]
    (i["page"] / "index.html").write_text(page(f"{i['title']} · {i['generator']} · {ds['title']}", rel, body, crumbs))


def group_lists(items):
    """{group: [items]} in the order of the dataset's group CSV."""
    ds = items[0]["dataset"]
    by_folder = {i["folder"]: i for i in items}
    lists = {}
    for r in csv.DictReader(open(ROOT / ds / "metamanifest" / EVAL_DATASETS[ds]["groups"], encoding="utf-8")):
        lists.setdefault(r["group"], []).append(by_folder[r["folder"]])
    return lists


def codebook_html():
    f = CODEBOOK["fields"]
    levels = lambda key: "<ol>" + "".join(f"<li>{esc(t)}</li>" for t in f[key]["levels"].values()) + "</ol>"  # noqa: E731
    fields = kv_html([
        ("Physical plausibility (P), 1–4, all clips", levels("physical_plausibility") + f'<p class="quiet">{esc(f["physical_plausibility"]["interpretation"])}</p>'),
        ("Action completion (A), yes/no, all clips", f'<p>{esc(f["action_completed"]["instructions"])}</p><p class="quiet">{esc(f["action_completed"]["interpretation"])}</p>'),
        ("Hidden-property adherence (H), 1–4, unobservable clips", levels("property_followed") + f'<p class="quiet">{esc(f["property_followed"]["interpretation"])}</p>'),
        ("Issue description, clips with P<4", f'<p>{esc(f["description_of_issue"]["instructions"])}</p>'),
        ("Violation labels, multi-select, clips with P<4", f'<p>{esc(f["physics_category"]["instructions"])}</p>')])
    labels = kv_html([(lab, f"<p>{esc(t)}</p>") for lab, t in CODEBOOK["labels"].items()])
    notes = "".join(f'<p class="quiet">{esc(t)}</p>' for t in CODEBOOK["label_notes"])
    return f"""<details class="review codebook"><summary>Annotation codebook <span>quoted from arXiv:2610.01162, Appendix B, Tables 8–9</span></summary>
<p class="quiet">Official definitions from the <a href="{PL_PAPER}">PhysicsLENS paper</a>. The benchmark is organised into seven physical
domains (collision, gravity, momentum, friction, deformation, fluid, causality), which the paper names but does not define
separately; annotators used the eight labels below, which add Contact.</p>
<h3>Form fields</h3>{fields}<h3>Violation labels</h3>{labels}{notes}
<p class="quiet">{esc(CODEBOOK["annotators"])}</p></details>"""


def server_specs(ds, eris):
    """<dt>/<dd> rows on the full set staged on ERIS, for a signal page header."""
    summary = eris[0]
    meta = EVAL_DATASETS[ds]
    if not summary or ds not in summary["per_dataset"]:
        return f"<dt>Full set</dt><dd>Not staged on ERIS yet. Filter: {esc(meta['eris_filter'])}</dd>"
    base = f"{ERIS_PATH}/{ds}"
    gens = " · ".join(f"{k.split('/', 1)[1]} {v}" for k, v in summary["per_generator"].items() if k.startswith(ds + "/"))
    return (f"<dt>Full set on ERIS</dt><dd><code>{esc(base)}/</code> · {summary['per_dataset'][ds]} videos "
            f"({esc(gens)})</dd><dt>Filter</dt><dd>{esc(meta['eris_filter'])}</dd>"
            f"<dt>Annotations on ERIS</dt><dd>Per video <code>&lt;generator&gt;/annotations/NNN_&lt;id&gt;.json</code>; "
            f"publisher tables and a flat per-video table in <code>{esc(base)}/annotations/</code></dd>")


def group_metrics(key):
    names = {k: t for k, t, _ in METRICS}
    return " and ".join(names[k] for keys, _, groups, _, _, _ in MAPPING if key in groups for k in keys)


def route_tabs(lists, current, base):
    """Large buttons, one per error group; base is the path from the current page to the dataset folder."""
    cur = ' aria-current="page"'
    return '<nav class="tabs routes" aria-label="Error groups">' + "".join(
        f'<a class="{metric_class(next(ks for ks, _, gs, _, _, _ in MAPPING if k in gs))}" href="{base}{k}/index.html"'
        f'{cur if k == current else ""}><b><span class="mdot"></span>{esc(GROUP_SHORT.get(k, k))}</b>'
        f'<small>{len(v)} videos · {esc(group_metrics(k))}</small></a>' for k, v in lists.items()) + "</nav>"


def signal_page(ds, items, out, eris):
    """The dataset page routes to one page per error group; each group page shows only that group."""
    d = out / ds
    meta = EVAL_DATASETS[ds]
    lists = group_lists(items)
    info = {g["group"]: g for i in items for g in i["label_groups"]}
    crumbs = [("All sets", "../../index.html"), (meta["title"], "../index.html")]
    for key, members in lists.items():
        g, gd = info[key], d / key
        gd.mkdir(parents=True, exist_ok=True)
        if ds == "PhysicsLENS":
            cards, table, defs = physicslens_group(key, members, gd)
        elif ds == "MiraBench":
            cards, table, defs = mira_group(key, members, gd)
        else:
            cards, table, defs = robojudge_group(members, gd)
        rule = g.get("group_rule") or ("Agent integrity (agent_consistency) = " + key[-1] + " in both label versions; robot "
                                       "first frames only.")
        c = Counter(i["generator"] for i in members)
        filters = "".join(f'<button type="button" data-filter="{slug(gen)}" aria-pressed="false">{esc(gen)} <span>{c[gen]}</span></button>'
                          for gen in meta["order"] if c[gen])
        body = f"""{route_tabs(lists, key, "../")}<header class="intro">
<p class="kicker">{esc(meta['title'])} · error group · compared with {esc(group_metrics(key))}</p>
<h1>{esc(g['group_title'])}</h1>
<p class="lede">Selection: {esc(rule)}</p>{defs}
</header>
<div class="toolbar" role="search"><div class="filters"><button type="button" data-filter="all" aria-pressed="true">All <span>{len(members)}</span></button>{filters}</div>
<input type="search" placeholder="Search tasks, generators and annotations" aria-label="Search videos"><output class="count" aria-live="polite"></output></div>
<section class="group" id="{key}"><h2 class="group__title">{esc(GROUP_SHORT.get(key, key))}<span>{len(members)} videos</span></h2>
<div class="cards">{''.join(cards)}</div>
<details class="review" open><summary>Labels <span>{"released (v2) · v1 in brackets where different" if ds == "RoboJudge_Bench" else "per video"}</span></summary>{table}</details></section>
<p class="empty" hidden>No videos match.</p>
{PLAYER_EVAL}"""
        (gd / "index.html").write_text(page(f"{g['group_title']} · {meta['title']} · evaluation review", "../../", body,
                                            crumbs + [(GROUP_SHORT.get(key, key), "index.html")]))
    extra = codebook_html() if ds == "PhysicsLENS" else ""
    body = f"""<header class="intro">
<p class="kicker">Signal page · {esc(meta['kind'])}</p>
<h1>{esc(meta['title'])} error groups</h1>
<p class="lede">{esc(LEDES[ds])} Choose a group:</p>
</header>
{route_tabs(lists, None, "")}
<dl class="specs">
<dt>Videos here</dt><dd>{len(items)} in {len(lists)} groups</dd>
{server_specs(ds, eris)}
<dt>Selection</dt><dd><code>{esc(meta['selection'])}</code> → <code>metamanifest/{esc(meta['groups'])}</code></dd>
<dt>Annotations</dt><dd><a href="annotations.jsonl">JSONL</a> (the {len(items)} videos here)</dd>
<dt>Source</dt><dd><a href="{esc(meta['source'])}">{esc(meta['source'].split('//')[1])}</a> · {esc(meta['license'])}</dd></dl>
{extra}"""
    (d / "index.html").write_text(page(f"{meta['title']} error groups · evaluation review", "../", body, [("All sets", "../index.html")]))
    with open(d / "annotations.jsonl", "w", encoding="utf-8") as fh:
        for i in items:
            fh.write(json.dumps(dict(dataset=ds, folder=i["folder"], generator=i["generator"],
                                     groups=[g["group"] for g in i["label_groups"]], prompt=i["prompt"] or None,
                                     metadata=i["meta"], human_annotation=i["human"]), ensure_ascii=False) + "\n")


LEDES = {
    "PhysicsLENS": "Ten videos per violation label, from all 442 upstream annotations, for the labels that map to our metrics: "
                   "Contact & Collision (rigid manipulated objects only), Momentum and Deformation. Annotations with P = 4 "
                   "that still carry labels are skipped, and so are first frames without a robot (EgoDex). Within a group, "
                   "annotations with the fewest other labels come first, spread across generators and first frames.",
    "MiraBench": "Physical-consistency videos of at most 12.1 s whose items are graded D, the most severe answer (score 0), "
                 "from the publisher's parsed CSV. Interaction: every in-scope video with action effect, contact causality or "
                 "response direction at D (12; the 3 with all three at D first). Interpenetration D: 10 of 15. Appearance D "
                 "(colour, shape for the material, size or material behaviour of the manipulated object): 10 of 40. "
                 "All are humanoid head-camera clips; the publisher does not name the robot.",
    "RoboJudge_Bench": "Ten robot videos with agent integrity (agent_consistency) = 0 and ten with = 1, the failures and partial "
                       "failures (higher is better). Only videos whose agent integrity is the same in both label versions are "
                       "used, and both versions are shown: v2 = released data/test.jsonl, v1 = the authors' review-repo file "
                       "of 16 Sep 2026. No GR1 humanoid video qualifies for code 1.",
}


def physicslens_group(key, members, d):
    labels = list(CODEBOOK["labels"])
    defs = '<dl class="kv codebook-defs">' + "".join(f"<dt>{esc(lab)}</dt><dd>{esc(CODEBOOK['labels'][lab])}</dd>"
                                                    for lab in PL_GROUP_LABELS[key]) + "</dl>"
    cards, rows = [], []
    for i in members:
        row = next(x for x in i["label_groups"] if x["group"] == key)
        extra_tags = [(f"object: {row['manipulated_object']}", "muted")] if row["manipulated_object"] else []
        cards.append(card(dict(i, tags=extra_tags + i.get("tags", [])), d))
        n = i["human"]["normalized"]
        flags = "".join(f'<td class="c"><span class="flag{" flag--on" if n[lab.lower()] else ""}">{n[lab.lower()]}</span></td>'
                        for lab in labels)
        rows.append(f'<tr><th scope="row"><a href="{quote(os.path.relpath(i["page"], d))}/index.html">{esc(i["title"])}</a></th>'
                    f'<td>{esc(i["generator"])}</td><td>{esc(row["manipulated_object"] or "—")}</td>'
                    f'<td class="c">{n["plausibility_1_4"]}</td><td class="c">{n["action_completed"]}</td>{flags}'
                    f'<td>{esc(i["human"]["description_of_issue"] or "")}</td></tr>')
    head = "".join(f'<th scope="col" title="{esc(lab)}">{PL_SHORT[lab]}</th>' for lab in labels)
    table = (f'<div class="matrix-wrap"><table class="chrono matrix"><thead><tr><th scope="col">Video</th><th scope="col">Generator</th>'
             f'<th scope="col">Object</th><th scope="col">P</th><th scope="col">A</th>{head}<th scope="col">Issue description</th>'
             f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
             '<p class="quiet">P plausibility 1–4, A action completed 1/0; label columns 1 = selected by the annotator.</p>')
    return cards, table, defs


def mira_group(key, members, d):
    codes = MIRA_GROUP_CODES[key]
    by_code = {it["code"]: it for it in members[0]["human"]["items"] if it["code"]}
    defs = '<dl class="kv">' + "".join(f'<dt>{esc(by_code[c]["label"])}<span class="zh">{esc(by_code[c]["title_zh"])}</span></dt>'
                                       f'<dd>{esc(by_code[c]["question_zh"])}</dd>' for c in codes) + "</dl>"
    cards, rows = [], []
    for i in members:
        its = {it["code"]: it for it in i["human"]["items"]}
        tags = [(f"{its[c]['label'].split(' (')[0]}: {its[c]['grade'] or 'NA'}", "" if its[c]["grade"] == "D" else "muted")
                for c in codes if c in its]
        cards.append(card(dict(i, tags=tags), d))
        cells = "".join(f'<td><span class="flag{" flag--on" if its.get(c, {}).get("grade") == "D" else ""}">'
                        f'{esc(its.get(c, {}).get("grade") or "NA")}</span> {esc(its.get(c, {}).get("answer_en") or "")}</td>'
                        for c in codes)
        rows.append(f'<tr><th scope="row"><a href="{quote(os.path.relpath(i["page"], d))}/index.html">{esc(i["title"])}</a></th>'
                    f'<td>{esc(i["generator"])}</td><td class="c">{esc(i["score"][1])}</td>{cells}</tr>')
    head = "".join(f'<th scope="col">{esc(by_code[c]["label"])}</th>' for c in codes)
    table = (f'<div class="matrix-wrap"><table class="chrono indicators"><thead><tr><th scope="col">Video</th>'
             f'<th scope="col">Model</th><th scope="col">Overall</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
             '<p class="quiet">A = no violation (1.0) · B = minor (0.67) · C = clear (0.33) · D = severe (0.0) · NA = event not '
             'present. For interpenetration the question is reversed (does penetration occur?), but D is still the most severe.</p>')
    return cards, table, defs


def robojudge_group(members, d):
    integrity = dict(RJ_CRITERIA["physical_adherence"])["Agent integrity"]
    defs = f'<dl class="kv"><dt>Agent integrity</dt><dd>{esc(integrity)} {esc(RJB_CODE_NOTE)}</dd></dl>'
    cards, rows = [], []
    for i in members:
        h = i["human"]
        e = h["labels_20260916_review_repo"]
        changed = [lab for k, _, lab in RJB_SUBS if e[k] != h["sub_scores"][k]]
        tags = i.get("tags", [])[:1] + [(f"{lab}: v1 ≠ v2", "") for lab in changed]
        cards.append(card(dict(i, tags=tags), d))
        subs = "".join(f'<td class="c">{h["sub_scores"][k]}{"" if e[k] == h["sub_scores"][k] else f" ({e[k]})"}</td>' for k, _, _ in RJB_SUBS)
        cell = lambda k: f'{h[k]}{"" if e[k] == h[k] else f" ({e[k]})"}'  # noqa: E731
        rows.append(f'<tr><th scope="row"><a href="{quote(os.path.relpath(i["page"], d))}/index.html">{esc(i["title"][:60])}</a></th>'
                    f'<td>{esc(i["generator"])}</td><td>{esc(i["meta"]["embodiment"])}</td>'
                    f'<td class="c">{cell("physical_adherence")}</td><td class="c">{cell("instruction_alignment")}</td>{subs}</tr>')
    head = "".join(f'<th scope="col" title="{esc(lab)}">{RJB_SHORT[k]}</th>' for k, _, lab in RJB_SUBS)
    table = (f'<div class="matrix-wrap"><table class="chrono matrix"><thead><tr><th scope="col">Video</th><th scope="col">Generator</th>'
             f'<th scope="col">Embodiment</th><th scope="col">PA</th><th scope="col">IA</th>{head}</tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></div><p class="quiet">v2 (released) value; the v1 value in brackets where it differs.</p>')
    return cards, table, defs


def metric_class(keys):
    return "m-" + keys[0]


def group_titles(sets):
    return {g["group"]: g["group_title"] for v in sets.values() for i in v for g in i["label_groups"]}


def glance_html(sets):
    """Compact matrix: our metrics down, datasets across, annotation items as chips linking to the examples."""
    head = "".join(f'<th scope="col">{esc(meta["title"])}<span>{esc(SCALE_SHORT[ds])}</span></th>' for ds, meta in EVAL_DATASETS.items())
    rows = []
    for key, title, text in METRICS:
        cells = []
        for ds in EVAL_DATASETS:
            chips_ = []
            for keys, d, groups, items, _, _ in MAPPING:
                if d != ds or key not in keys:
                    continue
                star = groups[0] in PRIMARY and key == "contact"
                href = f"{quote(ds)}/{groups[0]}/index.html"
                tip = f' title="{esc(PRIMARY[groups[0]])}"' if star else ""
                chips_ += [f'<a class="ichip{" ichip--primary" if star else ""}" href="{href}"{tip}>'
                           f'{"★ " if star else ""}{esc(ITEM_SHORT.get(n, n))}</a>' for n, _, _ in items]
            cells.append("<td>" + ("".join(chips_) or '<span class="none">—</span>') + "</td>")
        rows.append(f'<tr class="m-{key}"><th scope="row"><span class="mdot"></span>{esc(title)}<small>{esc(text)}</small></th>{"".join(cells)}</tr>')
    return (f'<div class="matrix-wrap"><table class="glance"><thead><tr><th scope="col">Our metric</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            '<p class="legend"><span class="ichip ichip--primary">★ Primary</span> the main comparison for that metric · '
            'chips open the error-group examples</p>')


def mapping_html(sets):
    """One colored panel per metric (or metric pair); one tile per dataset comparison, definitions collapsed."""
    names = {k: t for k, t, _ in METRICS}
    texts = {k: t for k, _, t in METRICS}
    titles = group_titles(sets)
    panels = []
    for metric_keys in METRIC_ROWS:
        tiles = []
        for keys, ds, groups, items, scale, note in MAPPING:
            if not set(keys) & set(metric_keys):
                continue
            primary = groups[0] in PRIMARY and "contact" in metric_keys
            target = (f'<span class="tile__target">→ {esc(" + ".join(names[k] for k in keys))}</span>'
                      if len(metric_keys) > 1 else "")
            chips_ = "".join(f'<span class="ichip">{esc(ITEM_SHORT.get(n, n))}</span>' for n, _, _ in items)
            caveat = f'<p class="caveat">{esc(note)}</p>' if note else ""
            defs = "".join(f"<dt>{esc(n)}" + (f'<span class="zh">{esc(z)}</span>' if z else "") + f"</dt><dd>{esc(t)}</dd>"
                           for n, z, t in items)
            links = "".join(f'<a class="btn" href="{quote(ds)}/{g}/index.html">{esc(GROUP_SHORT.get(g, g))} examples →</a>'
                            for g in groups if g in titles)
            tiles.append(f"""<article class="tile{" tile--primary" if primary else ""}">
<header class="tile__head"><span class="dsbadge">{esc(EVAL_DATASETS[ds]["title"])}</span>{'<span class="pbadge">★ Primary</span>' if primary else ""}{target}</header>
<div class="tile__chips">{chips_}</div>
<p class="tile__scale">{esc(SCALE_SHORT[ds])}</p>
<details class="more"><summary>Definitions &amp; caveats</summary><dl>{defs}</dl><p>{esc(scale)}</p>{caveat}</details>
<div class="tile__links">{links}</div></article>""")
        title = " + ".join(names[k] for k in metric_keys)
        sub = " · ".join(texts[k] for k in metric_keys)
        panels.append(f"""<section class="panel {metric_class(metric_keys)}"><header class="panel__head"><span class="mdot"></span><h3>{esc(title)}</h3>
<p>{esc(sub)}</p></header><div class="tiles">{"".join(tiles)}</div></section>""")
    return "".join(panels)


LAYOUT_ROWS = [("<dataset>/annotations/", "publisher annotation tables + one flat per-video table"),
               ("<dataset>/<generator>/prompts.csv", "generator,dataset,number,id,prompt,task_description"),
               ("<dataset>/<generator>/NNN_<id>.mp4", "unchanged upstream video"),
               ("<dataset>/<generator>/json/NNN_<id>.json", "provenance, prompt or task, checksum, clip facts"),
               ("<dataset>/<generator>/annotations/NNN_<id>.json", "the human annotation (RoboJudge-Bench: v1 and v2)"),
               ("RoboJudge_Bench/<generator>/first_frame/NNN_<id>.png", "conditioning frame")]
LAYOUT = esc("existing_annotations/\n" + "\n".join(f"  {path:<56}{text}" for path, text in LAYOUT_ROWS))


def server_section(eris):
    summary = eris[0]
    if not summary:
        return ('<section class="server home-section" id="eris"><h2>Data on ERIS</h2><p>Not staged yet: run '
                '<code>scripts/stage_existing_annotations.py</code> and upload.</p></section>')
    pills = "".join(f'<span class="pill"><b>{summary["per_dataset"].get(ds, 0)}</b> {esc(meta["title"])}</span>'
                    for ds, meta in EVAL_DATASETS.items())
    rows = []
    for ds, meta in EVAL_DATASETS.items():
        gens = ", ".join(f"{k.split('/', 1)[1]} ({v})" for k, v in summary["per_generator"].items() if k.startswith(ds + "/"))
        ann = {"PhysicsLENS": "consolidated_annotations.csv · annotations_flat.csv (normalized 0/1) · annotation_codebook.json",
               "MiraBench": "parsed/physical_consistency.csv · physical_consistency_in_scope.csv · annotations_flat.csv",
               "RoboJudge_Bench": "test_v2_released.jsonl · test_final_800_20260916_v1_review_repo.jsonl · annotations_flat_v1_v2.csv"}[ds]
        rows.append(f'<tr><th scope="row">{esc(meta["title"])}</th><td class="c">{summary["per_dataset"].get(ds, 0)}</td>'
                    f'<td>{esc(meta["eris_filter"])}</td><td>{esc(gens)}</td><td><code>{esc(ds)}/annotations/</code> {esc(ann)}</td></tr>')
    notes = "; ".join(f"{k}: {v}" for k, v in summary.get("notes", {}).items() if v)
    return f"""<section class="server home-section" id="eris"><h2>Data on ERIS</h2>
<div class="server__top"><div><p class="server__path"><code>{esc(ERIS_PATH)}/</code></p>
<p class="quiet">ERIS scratch · one folder per dataset, then per generator · {summary['videos']} videos, every one SHA-256-verified on the server</p></div>
<div class="pills">{pills}</div></div>
<div class="server__grid">
<div class="fact"><h4>Per video</h4><p><code>NNN_&lt;id&gt;.mp4</code> · <code>json/</code> prompt or task + provenance · <code>annotations/</code> human annotation</p></div>
<div class="fact"><h4>Per dataset</h4><p><code>annotations/</code> publisher tables + one flat per-video table</p></div>
<div class="fact"><h4>Index</h4><p><code>README.md</code> · <code>MANIFEST.csv</code> · <code>summary.json</code></p></div>
</div>
<details class="more"><summary>How each set was filtered and where its annotations are</summary>
<div class="matrix-wrap"><table class="chrono mapping"><thead><tr><th scope="col">Dataset</th><th scope="col">Videos</th>
<th scope="col">Filter</th><th scope="col">Generator folders</th><th scope="col">Annotation files</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>{f'<p class="quiet">{esc(notes)}.</p>' if notes else ''}</details>
<details class="more"><summary>Folder layout</summary><pre>{LAYOUT}</pre></details>
<details class="more"><summary>Prompts and provenance</summary><p>PhysicsLENS has the generator's text prompt
(“Scene: … Action: …”) and the task. RoboJudge-Bench releases the instruction, not the exact prefix/rewrite prompt. MiraBench
has no text prompt (first frame plus a robot action sequence), so its task description names the subset and scene. Files were
downloaded from pinned Hugging Face revisions to a temporary folder, checked against the Hub checksums, copied to ERIS and
checked again there with <code>sha256sum</code>; built by <code>scripts/stage_existing_annotations.py</code>.</p></details>
</section>"""


def index_page(sets, out, eris):
    titles = group_titles(sets)
    cards = []
    for ds, items in sets.items():
        meta = EVAL_DATASETS[ds]
        step = max(1, len(items) // 4)
        strip = "".join(f'<img src="{quote(os.path.relpath(i["page"] / "poster.jpg", out))}" alt="" loading="lazy">'
                        for i in items[::step][:4])
        groups = list(dict.fromkeys(g["group"] for i in items for g in i["label_groups"]))
        buttons = "".join(f'<a class="gbtn {metric_class(next(k for k, _, gs, _, _, _ in MAPPING if g in gs))}" '
                          f'href="{quote(ds)}/{g}/index.html"><span class="mdot"></span>{esc(GROUP_SHORT.get(g, g))}</a>'
                          for g in groups if g in titles)
        staged = (eris[0] or {}).get("per_dataset", {}).get(ds)
        cards.append(f"""<article class="dscard"><a class="dscard__strip" href="{quote(ds)}/index.html">{strip}</a>
<div class="dscard__body"><h3><a href="{quote(ds)}/index.html">{esc(meta['title'])}</a></h3>
<p class="quiet">{esc(meta['kind'])}</p>
<p class="dscard__stats"><b>{len(items)}</b> example videos · <b>{staged or 0}</b> on ERIS · {meta['upstream_total']} upstream</p>
<div class="gbtns">{buttons}</div></div></article>""")
    body = f"""<header class="hero">
<p class="kicker">World-model evaluation</p>
<h1>Our physics metrics vs. published human annotations</h1>
<p class="lede">Five metrics on our side, three human-annotated benchmarks on the other: which annotation compares with which
metric, on what scale, and the clips where annotators marked a violation.</p>
<nav class="jump" aria-label="On this page"><a href="#glance">At a glance</a><a href="#mapping">Mapping in detail</a>
<a href="#sets">Signal pages</a><a href="#eris">Data on ERIS</a></nav>
</header>
<section class="home-section" id="glance"><h2>At a glance</h2>{glance_html(sets)}</section>
<section class="home-section" id="mapping"><h2>Mapping in detail</h2>
<details class="more scales"><summary>How to read the scales</summary><p>A violation on their side should line up with a low score on
ours. PhysicsLENS labels are 0/1 flags (1 = the annotator selected the violation; 0 is not a confirmation of correctness).
MiraBench grades A–D score 1, 0.67, 0.33 and 0: higher score, lower error. RoboJudge-Bench sub-scores are 0–2, higher is
better, kept in two label versions (v1 review repo, v2 release).</p></details>
{mapping_html(sets)}</section>
<section class="home-section" id="sets"><h2>Signal pages</h2><div class="dscards">{''.join(cards)}</div></section>
{server_section(eris)}"""
    (out / "index.html").write_text(page("Robotvideo evaluation review", "", body))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "site" / "eval")
    args = ap.parse_args()
    out = args.out.resolve()
    (out / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "site" / "assets" / "style.css", out / "assets" / "style.css")
    js = (ROOT / "site" / "assets" / "app.js").read_text(encoding="utf-8")
    patched = js.replace('d.prompt || "No reviewed action prompt yet."', 'd.prompt || d.promptMissing || "No reviewed action prompt yet."')
    assert patched != js, "app.js changed; update the prompt fallback patch"
    (out / "assets" / "app.js").write_text(patched, encoding="utf-8")
    (out / "assets" / "eval.css").write_text(EVAL_CSS, encoding="utf-8")
    for stale_dir in [p for p in out.iterdir() if p.is_dir() and p.name not in EVAL_DATASETS and p.name != "assets"]:
        shutil.rmtree(stale_dir)  # pages of datasets no longer reviewed (RoboJudge-Train)
    eris = eris_info()
    sets = {}
    for ds in EVAL_DATASETS:
        items = collect(ds, out)
        keep = {i["page"] for i in items}
        for idx in list((out / ds).rglob("index.html")) if (out / ds).exists() else []:
            if idx.parent != out / ds and idx.parent not in keep and idx.parent.name not in GROUP_SHORT:
                shutil.rmtree(idx.parent)
        for folder in sorted((p for p in (out / ds).rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True) \
                if (out / ds).exists() else []:
            if not any(folder.iterdir()):
                folder.rmdir()
        with ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(lambda i: (transcode(i["video"], i["page"] / "clip.mp4"), stills(i["video"], i["page"])), items))
        for i in items:
            i["info"] = probe(i["video"])
        done = set()
        for seq in group_lists(items).values():  # the pager runs within the first group of each video
            for n, i in enumerate(seq):
                if i["folder"] not in done:
                    done.add(i["folder"])
                    video_page(i, seq[n - 1] if n else None, seq[n + 1] if n + 1 < len(seq) else None, eris)
        signal_page(ds, items, out, eris)
        sets[ds] = items
    index_page(sets, out, eris)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{sum(len(v) for v in sets.values())} error-group videos in {len(sets)} sets, {size / 1e6:.1f} MB -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
