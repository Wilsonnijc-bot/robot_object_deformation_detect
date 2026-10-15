"""Write the Robotvideo initial-image quality report (markdown) from input_quality.py's CSV.

Usage: input_quality_report.py <input_quality.csv> <out.md>

Grades use lap_var_model, the Laplacian variance of the exact image the generators received (after the
cover-resize + center crop). Thresholds were set by inspecting 1:1 crops of each source-size group.
"""
import collections
import csv
import statistics as st
import sys

GRADES = [("A", 400, "Good", "crisp edges and fine texture at model resolution"),
          ("B", 120, "Fair", "usable; mild softness or compression, small detail smeared"),
          ("C", 40, "Poor", "visibly soft and/or blocky; upscaled or low-bitrate source"),
          ("D", 0, "Very poor", "heavily blurred after upscaling; fine structure (fingers, edges) lost")]
DATASETS = ["TAVP", "Humanoid_Everyday", "NVIDIA_GR1_100"]
BLOCKY = 1.35


def grade(r):
    v = float(r["lap_var_model"])
    return next(g for g, lo, _, _ in GRADES if v >= lo)


def med(rows, k):
    return st.median(float(r[k]) for r in rows)


def main(src, out):
    rows = list(csv.DictReader(open(src)))
    for r in rows:
        r["grade"] = grade(r)
        r["size"] = f"{r['width']}x{r['height']}"
        r["gen"] = r["generated"] == "True"
    L = []
    w = L.append
    total = len(rows)
    done = sum(r["gen"] for r in rows)
    w("# Robotvideo world-model inputs: inventory and initial-image pixel quality\n")
    w("Source: `~/Downloads/Robotvideo` on the Mac, byte-identical to `/PHShome/zy992/Wilson/generation input/` "
      "on ERIS (all 353 generation-input PNG/TXT files match by MD5). One input = one video folder's "
      "generation first frame + generation prompt.\n")

    w("## 1. How many inputs are left\n")
    w("| Dataset | Inputs available | Generated (first 15) | Not generated |")
    w("|---|---:|---:|---:|")
    for ds in DATASETS:
        d = [r for r in rows if r["dataset"] == ds]
        g = sum(r["gen"] for r in d)
        w(f"| {ds} | {len(d)} | {g} | {len(d) - g} |")
    w(f"| **Total** | **{total}** | **{done}** | **{total - done}** |\n")
    w("Each generated input was run through 3 generators (Cosmos 2.5, Cosmos 3, ABot-PhysWorld), so "
      f"{done} inputs = {3 * done} videos. Running the remaining {total - done} inputs through the same 3 "
      f"generators would add {3 * (total - done)} videos.\n")
    w("Not counted as inputs:\n")
    w("- TAVP: 10 videos also have `prompt for original multi step action(reproducing the original video).txt`, "
      "a full multi-step prompt for the same first frame; only the `first_substep/` pair was used.")
    w("- TAVP Dobot: `source_first_frame.png` is the full 1920x480 three-camera frame; the input is the "
      "cropped middle view (640x480).")
    w("- Humanoid_Everyday: `first_person/first_frame.png` exists but has no generation prompt.\n")

    w("## 2. Method\n")
    w("Every metric is computed on luminance (0-255). Script: "
      "`robot/experiments/world_model_generation/input_quality.py`; per-image values in "
      "`robotvideo_input_quality.csv` next to this file.\n")
    w("| Metric | Meaning |")
    w("|---|---|")
    w("| Upscale | Cover-resize factor from the source frame to the generation size (1280x704, 960x704, 704x1280 or 704x960, nearest aspect). >1 means the model saw an enlarged image. |")
    w("| Sharpness (model view) | Variance of the Laplacian of the exact image the generators received. **Grades are based on this.** |")
    w("| Sharpness (native) | Same at the source's own size. Not comparable across sizes. |")
    w("| Detail bandwidth | Fraction of the frequency limit (Nyquist) holding 99% of spectral energy. ~0.5 = full detail (sharp GR1 frames); ~0.25 = the frame resolves only about half its nominal linear resolution, i.e. it is soft even at its own size (often an upsampled or low-bitrate video frame). |")
    w("| Noise σ | Immerkaer fast noise estimate. |")
    w(f"| Blockiness | Mean step across 8-pixel grid lines / elsewhere. ~1.0 none; >{BLOCKY} visible compression blocks. |")
    w("| Exposure | Mean luma; % of pixels clipped dark (≤5) / bright (≥250). |")
    w("| Contrast, colorfulness | RMS contrast; Hasler-Suesstrunk colorfulness. |\n")
    w("No input has black letterbox/pillarbox borders.\n")

    w("## 3. Grades\n")
    w("Thresholds on model-view sharpness, set by inspecting 1:1 crops of every source-size group:\n")
    w("| Grade | Sharpness (model view) | Label | What it looks like |")
    w("|---|---|---|---|")
    hi = None
    for g, lo, label, desc in GRADES:
        rng = f"≥ {lo}" if hi is None else (f"{lo}–{hi}" if lo else f"< {hi}")
        w(f"| **{g}** | {rng} | {label} | {desc} |")
        hi = lo
    w("")
    w("| Dataset | A | B | C | D | Generated: A/B/C/D | Remaining: A/B/C/D |")
    w("|---|---:|---:|---:|---:|---|---|")
    for ds in DATASETS:
        d = [r for r in rows if r["dataset"] == ds]
        c = collections.Counter(r["grade"] for r in d)
        cg = collections.Counter(r["grade"] for r in d if r["gen"])
        cr = collections.Counter(r["grade"] for r in d if not r["gen"])
        f = lambda cc: "/".join(str(cc[g]) for g in "ABCD")
        w(f"| {ds} | {c['A']} | {c['B']} | {c['C']} | {c['D']} | {f(cg)} | {f(cr)} |")
    c = collections.Counter(r["grade"] for r in rows)
    w(f"| **All** | **{c['A']}** | **{c['B']}** | **{c['C']}** | **{c['D']}** | | |\n")

    w("## 4. Quality by source group\n")
    w("| Dataset | Source size | n | Upscale | Sharpness model view | Detail bandwidth | Noise σ | Blockiness | Dark clip % | Grades | Assessment |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    notes = {
        ("NVIDIA_GR1_100", "1280x800"): "Best set. Native resolution, full detail bandwidth, mild sensor noise. Dark clipping comes from black robot/background regions, not underexposure.",
        ("Humanoid_Everyday", "1440x1920"): "High-resolution portrait; soft at native size but downscaled 2x for the model, so the model view is crisp.",
        ("Humanoid_Everyday", "1920x1440"): "As above, landscape.",
        ("TAVP", "1280x720"): "Franka. Native size, but detail bandwidth ~0.29: the source videos are ~0.16 Mbps, so edges are smoothed and fine texture is gone. Blocking is mild.",
        ("Humanoid_Everyday", "720x1280"): "Nominally native, but bandwidth ~0.25: soft content (likely a resized or low-bitrate source). Mostly grade B.",
        ("Humanoid_Everyday", "592x1280"): "Slight 1.19x upscale on top of soft content; some frames blocky.",
        ("TAVP", "640x480"): "Dobot middle view cut from the 1920x480 three-camera video, upscaled 1.5x. Strong 8-px compression blocking, low light (mean luma ~78).",
        ("TAVP", "960x720"): "Dobot illumination-variation clip: dim (mean luma ~56, 14% crushed blacks) and soft.",
        ("Humanoid_Everyday", "448x960"): "Upscaled 1.57x from an already soft frame; blocky.",
        ("Humanoid_Everyday", "360x480"): "Upscaled 2x.",
        ("Humanoid_Everyday", "320x568"): "Upscaled 2.25x. Very blurry at model resolution; hands, edges and small objects lose definition. Worst group.",
    }
    grp = collections.defaultdict(list)
    for r in rows:
        grp[(r["dataset"], r["size"])].append(r)
    order = sorted(grp, key=lambda k: -med(grp[k], "lap_var_model"))
    for k in order:
        d = grp[k]
        gc = collections.Counter(r["grade"] for r in d)
        gs = " ".join(f"{g}×{gc[g]}" for g in "ABCD" if gc[g])
        w(f"| {k[0]} | {k[1]} | {len(d)} | {med(d, 'upscale'):.2f} | {med(d, 'lap_var_model'):.0f} | "
          f"{med(d, 'bandwidth99'):.2f} | {med(d, 'noise_sigma'):.2f} | {med(d, 'blockiness'):.2f} | "
          f"{med(d, 'clip_lo_pct'):.1f} | {gs} | {notes.get(k, '')} |")
    w("\nValues are medians within the group.\n")

    w("## 5. Key findings\n")
    w("1. **NVIDIA_GR1_100 is uniformly good**: all 92 inputs are grade A, so the 77 remaining GR1 inputs can be generated without quality concerns.")
    w("2. **All remaining TAVP inputs are grade C**: the 8 not yet generated (016-023) are Dobot clips (640x480 middle-view crops, plus one dim 960x720 clip). Generating them adds only low-quality, blocky conditioning frames.")
    w("3. **Humanoid_Everyday is mixed and source size is misleading**: 720x1280 frames have detail bandwidth ~0.25 versus ~0.5 for sharp frames, i.e. they resolve roughly half their nominal linear resolution (about what a sharp 360x640 frame holds), while the 1440x1920 frames are the sharpest Humanoid inputs once downscaled. Of 34 remaining Humanoid inputs, 8 are grade D (320x568, upscaled 2.25x).")
    w("4. **Compression blocking** above the visible threshold appears in all 14 TAVP 640x480 Dobot frames and in 4 Humanoid frames (592x1280 and 448x960).")
    w("5. **Exposure** is acceptable everywhere except TAVP 023 (illumination-variation clip, deliberately dark). Highlight clipping is ≤2% for every image.")
    w("6. The `low_res_input/` folders in the generated outputs (TAVP 09-15, Humanoid 06/11/14/15) are exactly the generated grade C/D items with ≥1.5x upscaling; grade C also includes some native-size but soft or blocky frames that are not in those folders.\n")

    w("## 6. Per-image table\n")
    for ds in DATASETS:
        w(f"### {ds}\n")
        w("| # | Folder | Generated | Size | Upscale | Sharpness model | Bandwidth | Noise σ | Blockiness | Mean luma | Dark clip % | Grade |")
        w("|---:|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|")
        for r in [r for r in rows if r["dataset"] == ds]:
            blk = f"**{float(r['blockiness']):.2f}**" if float(r["blockiness"]) > BLOCKY else f"{float(r['blockiness']):.2f}"
            w(f"| {r['index']} | `{r['name']}` | {'yes' if r['gen'] else ''} | {r['size']} | {float(r['upscale']):.2f} | "
              f"{float(r['lap_var_model']):.0f} | {float(r['bandwidth99']):.2f} | {float(r['noise_sigma']):.2f} | {blk} | "
              f"{float(r['mean_luma']):.0f} | {float(r['clip_lo_pct']):.1f} | {r['grade']} |")
        w("")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
