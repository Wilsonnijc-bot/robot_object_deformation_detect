"""Arrange generated videos for Google Drive as <generator>/<dataset>/NN_<robot>-<action>.{mp4,json}.

Items whose source first frame had to be upscaled >= 1.5x to the generation size go one level deeper,
in <generator>/<dataset>/low_res_input/ (Humanoid 06, 11, 14, 15 and the TAVP Dobot clips 09-15).

NN is the item's 1-based position within its dataset in manifest order (the first-15 natural-sort
selection). It keeps names unique where a robot repeats an action (e.g. two franka-collect_fruits).
Source outcome labels (success/failure) are not part of the name; the original source id stays in
each sidecar as `source_id` and in naming.csv.

Usage: organize_drive.py <flat_dir> <out_dir>
  flat_dir holds manifest.json and <model>/<source_id>.{mp4,json}; out_dir is created.
"""
import csv
import json
import os
import re
import shutil
import sys

GENERATORS = ["cosmos2.5", "cosmos3", "abot_physworld"]
LOW_RES_DIR = "low_res_input"
LOW_RES_UPSCALE = 1.5
DATASETS = ["TAVP", "Humanoid_Everyday", "NVIDIA_GR1_100"]

# NVIDIA_GR1_100 folders are bare numbers; actions are summarised from each item's prompt.
GR1_ACTIONS = {
    "1": "place_bok_choy", "2": "place_rubiks_cube", "3": "place_banana", "4": "place_dragonfruit",
    "6": "place_orange", "9": "place_purple_glass", "10": "place_grapes", "12": "place_potato",
    "13": "place_pink_mug", "14": "place_red_apple", "15": "place_orange_cup",
    "16": "place_tomato_in_bag", "17": "place_milk_carton", "18": "place_juice_carton",
    "19": "place_green_apple",
}
TAVP_SUFFIX = re.compile(r"_(standard|failure_case|[a-z]+_generalization|heavy_occlusion|illumination_variation)"
                         r"_(success|failure)$")


def short_name(item):
    ds, name = item["dataset"], item["name"]
    if ds == "TAVP":
        _, robot, action = TAVP_SUFFIX.sub("", name).split("_", 2)
        return f"{robot}-{action}"
    if ds == "Humanoid_Everyday":
        return "humanoid-" + name.split("__", 1)[1]
    if ds == "NVIDIA_GR1_100":
        return "gr1-" + GR1_ACTIONS[name]
    raise ValueError(ds)


def upscale(item):
    """Cover-resize factor from the source first frame to the shared generation bucket."""
    return max(item["gen_width"] / item["width"], item["gen_height"] / item["height"])


def main(flat, out):
    items = json.load(open(os.path.join(flat, "manifest.json")))
    names, pos = {}, {}
    for it in items:
        pos[it["dataset"]] = pos.get(it["dataset"], 0) + 1
        names[it["id"]] = f"{pos[it['dataset']]:02d}_{short_name(it)}"
    assert len(set(names.values())) == len(names), "names must be unique"
    by_id = {it["id"]: it for it in items}
    low_res = {it["id"] for it in items if upscale(it) >= LOW_RES_UPSCALE}
    for g in GENERATORS:
        for ds in DATASETS:
            os.makedirs(os.path.join(out, g, ds), exist_ok=True)
            if any(by_id[i]["dataset"] == ds for i in low_res):
                os.makedirs(os.path.join(out, g, ds, LOW_RES_DIR), exist_ok=True)
    moved = 0
    for g in GENERATORS:
        src = os.path.join(flat, g)
        if not os.path.isdir(src):
            continue
        for f in sorted(os.listdir(src)):
            sid, ext = os.path.splitext(f)
            if ext not in (".mp4", ".json") or sid not in names:
                continue
            if not os.path.exists(os.path.join(src, sid + ".json")):
                continue  # mp4 without its sidecar = unfinished; skip
            sub = [LOW_RES_DIR] if sid in low_res else []
            dst = os.path.join(out, g, by_id[sid]["dataset"], *sub, names[sid] + ext)
            if ext == ".json":
                meta = json.load(open(os.path.join(src, f)))
                meta = {"name": names[sid], "source_id": sid,
                        "source_size": f"{by_id[sid]['width']}x{by_id[sid]['height']}",
                        "input_upscale": round(upscale(by_id[sid]), 2), "low_res_input": sid in low_res,
                        **{k: v for k, v in meta.items() if k != "id"}}
                json.dump(meta, open(dst, "w"), indent=1, ensure_ascii=False)
            else:
                shutil.copy2(os.path.join(src, f), dst)
            moved += 1
    for it in items:
        it["drive_name"] = names[it["id"]]
        it["input_upscale"] = round(upscale(it), 2)
        it["low_res_input"] = it["id"] in low_res
    json.dump(items, open(os.path.join(out, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(out, "naming.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "drive_name", "low_res_input", "source_size", "input_upscale", "source_id",
                    "source_frame", "prompt"])
        for it in items:
            w.writerow([it["dataset"], names[it["id"]], it["id"] in low_res, f"{it['width']}x{it['height']}",
                        round(upscale(it), 2), it["id"], it["source_frame"], it["prompt"]])
    print(f"{moved} files arranged into {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
