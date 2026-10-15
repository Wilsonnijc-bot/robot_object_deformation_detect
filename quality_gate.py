"""Sharpness gate for world-model conditioning frames (see data_quality_gate.md).

Usage: .venv/bin/python quality_gate.py [--check PNG ...]

Without arguments, measures every generation input in the repository and
  * writes quality_gate.csv at the repository root,
  * writes the `data_quality_gate` block into each input's metadata/metadata.json,
  * sets the gate_* columns in every dataset's generation manifest.
With --check, prints the measurement for the given PNGs only.

The measured image is the one the generator receives: the first frame is
cover-resized (Lanczos) to the nearest-aspect 720p-class bucket and center-cropped,
exactly as deformationdetection/robot/experiments/world_model_generation/input_quality.py.
Sharpness is the variance of the 4-neighbour Laplacian of its luma (scipy.ndimage.laplace,
reflect mode), reimplemented in NumPy so only Pillow and NumPy are needed.
"""
import csv
import io
import json
import os
import re
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = {"third_person": 120, "first_person": 60}
# Datasets admitted by visual review despite failing the metric; the measured score is still recorded.
EXCEPTIONS = {
    "RoboCOIN_Airbot_MMK2_stack_block": (
        "Visual-review exception (2026-10-15): edges of the hands and blocks are judged acceptable; "
        "the low score is driven mostly by the large untextured table."),
    "RoboMIND": (
        "Accepted as an exception by project decision (2026-10-15); the Franka camera_right frames "
        "score below the third-person threshold."),
    "Humanoid_Everyday": (
        "Accepted as an exception by project decision (2026-10-15); the score is below the "
        "third-person threshold."),
}
RULE = ("Variance of the Laplacian of the luma of the model-view frame (cover-resize to the "
        "nearest-aspect 720p-class bucket, center crop); pass if >= 120 third-person, >= 60 first-person.")
BUCKETS = [(16 / 9, (1280, 704)), (4 / 3, (960, 704)), (1.0, (832, 832)),
           (3 / 4, (704, 960)), (9 / 16, (704, 1280))]
MANIFESTS = [
    "TAVP/world_model_manifest.csv",
    "Humanoid_Everyday/metamanifest/world_model_prompts.csv",
    "NVIDIA_GR1_100/metamanifest/world_model_prompts.csv",
    "ActionNet/metamanifest/world_model_prompts.csv",
    "RoboCOIN_Airbot_MMK2_stack_block/metamanifest/world_model_prompts.csv",
    "RoboMIND/metamanifest/world_model_prompts.csv",
]
OLD_COLUMNS = ("upscale_cosmos3", "upscale_cosmos25", "upscale_abot_lvp", "passes_quality_gate_all_models")
GATE_COLUMNS = ("gate_view", "gate_sharpness", "gate_threshold", "gate_result")


def nat(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def bucket(w, h):
    return min(BUCKETS, key=lambda b: abs(b[0] - w / h))[1]


def model_view(im, tw, th):
    s = max(tw / im.width, th / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    left, top = (im.width - tw) // 2, (im.height - th) // 2
    return im.crop((left, top, left + tw, top + th))


def luma(im):
    rgb = np.asarray(im.convert("RGB"), float)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def laplace_var(y):
    p = np.pad(y, 1, mode="symmetric")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * y
    return float(lap.var())


def measure(png):
    im = Image.open(png).convert("RGB")
    tw, th = bucket(im.width, im.height)
    return dict(width=im.width, height=im.height, bucket=f"{tw}x{th}",
                upscale=round(max(tw / im.width, th / im.height), 3),
                sharpness=round(laplace_var(luma(model_view(im, tw, th))), 1))


def view_of(dataset, folder):
    if dataset in ("TAVP", "Humanoid_Everyday"):
        return "third_person"
    if dataset == "RoboMIND" and folder.startswith("franka_"):
        return "third_person"
    return "first_person"


def inputs():
    """(dataset, folder, first_frame) for every generation input, one per video folder."""
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted([d for d in dirnames if not d.startswith(".") and d not in ("metadata", "site")], key=nat)
        if os.path.basename(dirpath) != "generation input":
            continue
        sub = os.path.join(dirpath, "first_substep")
        frame = os.path.join(sub if os.path.isdir(sub) else dirpath, "first_frame.png")
        if not os.path.exists(frame):
            continue
        parts = os.path.relpath(dirpath, ROOT).split(os.sep)
        folder = parts[1] if parts[0] == "Humanoid_Everyday" else "/".join(parts[1:-1])
        yield parts[0], folder, frame


def gate_row(dataset, folder, frame):
    view = view_of(dataset, folder)
    m = measure(frame)
    result = "pass" if m["sharpness"] >= THRESHOLD[view] else "fail"
    if result == "fail" and dataset in EXCEPTIONS:
        result = "exception"
    return dict(dataset=dataset, folder=folder, view=view, **m, threshold=THRESHOLD[view],
                gate=result, first_frame=os.path.relpath(frame, ROOT))


def write_json_block(row):
    path = os.path.join(ROOT, os.path.dirname(row["first_frame"]), "metadata", "metadata.json")
    if not os.path.exists(path):
        return
    text = open(path).read()
    indent = re.search(r"\n( +)\"", text)
    data = json.loads(text)
    data.pop("data_quality_gate", None)
    data["data_quality_gate"] = {
        "rule": RULE, "documentation": "data_quality_gate.md (repository root)",
        "view": row["view"], "model_view_resolution": row["bucket"], "upscale_factor": row["upscale"],
        "sharpness": row["sharpness"], "threshold": row["threshold"], "result": row["gate"]}
    if row["gate"] == "exception":
        data["data_quality_gate"]["exception_reason"] = EXCEPTIONS[row["dataset"]]
    out = json.dumps(data, indent=len(indent.group(1)) if indent else 2, ensure_ascii=False)
    with open(path, "w") as fh:
        fh.write(out + ("\n" if text.endswith("\n") else ""))


def patch_manifest(rel, by_frame):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return
    text = open(path, newline="").read()
    eol = "\r\n" if "\r\n" in text else "\n"
    rows = list(csv.DictReader(io.StringIO(text, newline="")))
    if not rows:
        return
    col = "relative_first_frame_path" if "relative_first_frame_path" in rows[0] else "first_frame"
    fields = [f for f in rows[0] if f not in OLD_COLUMNS and f not in GATE_COLUMNS] + list(GATE_COLUMNS)
    here = os.path.dirname(path)
    for r in rows:
        for base in (here, os.path.dirname(here)):
            key = os.path.relpath(os.path.normpath(os.path.join(base, r[col])), ROOT)
            if key in by_frame:
                break
        g = by_frame[key]
        for f in OLD_COLUMNS:
            r.pop(f, None)
        r.update(gate_view=g["view"], gate_sharpness=g["sharpness"],
                 gate_threshold=g["threshold"], gate_result=g["gate"])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator=eol)
    w.writeheader()
    w.writerows(rows)
    with open(path, "w", newline="") as fh:
        fh.write(buf.getvalue())


def main(argv):
    if argv[:1] == ["--check"]:
        for png in argv[1:]:
            print(png, measure(png))
        return
    rows = [gate_row(*i) for i in inputs()]
    with open(os.path.join(ROOT, "quality_gate.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        write_json_block(r)
    by_frame = {r["first_frame"]: r for r in rows}
    for rel in MANIFESTS:
        patch_manifest(rel, by_frame)
    count = {g: sum(r["gate"] == g for r in rows) for g in ("pass", "exception", "fail")}
    print(f"{len(rows)} inputs: {count}")
    for r in rows:
        if r["gate"] != "pass":
            print(f"  {r['gate']:9s} {r['dataset']}/{r['folder']}: {r['sharpness']} vs {r['threshold']}")


if __name__ == "__main__":
    main(sys.argv[1:])
