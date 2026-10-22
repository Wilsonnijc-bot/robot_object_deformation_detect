"""Embodiment composition of PhysicsLENS and RoboJudge-Bench, with the share of single-arm serial manipulators
(standalone arms such as Franka), a few downloaded examples of them, and one pie chart per dataset.

Usage: .venv/bin/python -I scripts/serial_manipulators.py /tmp/new_empty_download_directory [--no-download]

Every robot arm here is a serial chain; "single-arm serial manipulator" means one standalone arm (Franka, WidowX,
DLR SARA, the RH20T arms), as opposed to dual-arm systems, humanoids and human hands. Classes come from the first
frame's source corpus, checked on the first frames:
  PhysicsLENS      the source token in each file name (consolidated_prompts_80.csv, manifest.csv); unit = the 439
                   annotated generated videos staged on ERIS (metamanifest/existing_annotations_manifest.csv).
  RoboJudge-Bench  the source corpus of each of the 800 videos (metamanifest/test.jsonl). Open X-Embodiment mixes
                   robots, so its 69 first frames were inspected one by one: all are single arms, either a black
                   WidowX-type arm in Bridge-style toy kitchens (BRIDGE_TASKS) or a white single arm over a blue
                   tabletop whose model cannot be identified at the 512 px source resolution.
Writes metamanifest/serial_manipulators/{composition.csv, summary.json, pies.html}, and per dataset
<dataset>/metamanifest/serial_manipulator_examples.csv with the downloaded examples in the usual
<...>/original/ layout of that dataset's fetch_sample.py.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "metamanifest" / "serial_manipulators"
CLASSES = [  # fixed order = categorical palette slots 1-5 (validated, adjacent pairs)
    ("single", "Single-arm serial manipulator", "#2a78d6"),
    ("dual_fixed", "Dual-arm, fixed base", "#eb6834"),
    ("dual_mobile", "Dual-arm, mobile base", "#1baf7a"),
    ("humanoid", "Humanoid", "#eda100"),
    ("human", "Human hands", "#e87ba4"),
]
# PhysicsLENS source token -> (class, robot)
PL_SOURCES = {
    "rh20t_cfg1": ("single", "RH20T cfg 1 arm"), "dlr_sara": ("single", "DLR SARA arm"),
    "aloha_static": ("dual_fixed", "ALOHA (static)"), "unitree_z1_dual": ("dual_fixed", "Unitree Z1 dual arms"),
    "aloha_mobile": ("dual_mobile", "Mobile ALOHA"), "agibot": ("dual_mobile", "AgiBot G1"),
    "ai2_alphabot": ("dual_mobile", "AI2 AlphaBot 2"), "galbot_g1": ("dual_mobile", "Galbot G1"),
    "unitree_g1dex1": ("humanoid", "Unitree G1 · Dex1"), "unitree_g1dex3": ("humanoid", "Unitree G1 · Dex3"),
    "unitree_g1edu": ("humanoid", "Unitree G1 EDU"), "humanoid_everyday": ("humanoid", "Humanoid Everyday (Unitree)"),
    "gr00t_gr1": ("humanoid", "Fourier GR1"), "gr00t_egodex": ("human", "EgoDex human hands"),
}
RJB_SOURCES = {
    "droid": ("single", "Franka (DROID)"), "robotwin": ("dual_fixed", "AgileX dual arms (RoboTwin, simulated)"),
    "agibot_world": ("dual_mobile", "AgiBot G1 (AgiBot World)"), "gr1_inlab": ("humanoid", "Fourier GR1 (in-lab)"),
    "egodex_human": ("human", "EgoDex"), "egoscaler_human": ("human", "EgoScaler"),
    "epickitchens_human": ("human", "EPIC-KITCHENS"),
}
# Open X-Embodiment tasks whose first frame shows the black WidowX-type arm in a Bridge-style toy kitchen.
BRIDGE_TASKS = {f"task_{n:04d}" for n in (2, 3, 4, 9, 11, 12, 14, 16, 17, 18, 19, 21, 22, 23, 45, 53, 61, 64, 84, 87, 94)}
PER_DATASET = 3


def load(ds, name):
    sys.modules.pop("fetch_sample", None)
    spec = importlib.util.spec_from_file_location(f"{ds}_{name}", ROOT / ds / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pl_token(stem):
    return next(t for t in PL_SOURCES if t in stem)


def physicslens_rows():
    prompts = {int(r["id"]): r["video_file"] for r in csv.DictReader(open(ROOT / "PhysicsLENS" / "metamanifest" / "consolidated_prompts_80.csv", encoding="utf-8"))}
    prompts[91] = "091_unitree_z1_dual_pour_coffee_ep330.mp4"  # no prompt row upstream; name from the annotation CSV
    rows = []
    for r in csv.DictReader(open(ROOT / "metamanifest" / "existing_annotations_manifest.csv", encoding="utf-8")):
        if r["dataset"] != "PhysicsLENS":
            continue
        tid = int(r["id"][:3])
        cls, robot = PL_SOURCES[pl_token(prompts[tid])]
        rows.append(dict(dataset="PhysicsLENS", id=r["id"], first_frame=f"{tid:03d}", generator=r["generator"],
                         source=pl_token(prompts[tid]), cls=cls, robot=robot))
    return rows


def robojudge_rows():
    rows = []
    for r in map(json.loads, open(ROOT / "RoboJudge_Bench" / "metamanifest" / "test.jsonl", encoding="utf-8")):
        if r["dataset"] == "open_x_embodiment":
            cls, robot = ("single", "WidowX-type arm (Bridge toy kitchen)") if r["task"] in BRIDGE_TASKS else \
                ("single", "White single arm, model not identified")
        else:
            cls, robot = RJB_SOURCES[r["dataset"]]
        rows.append(dict(dataset="RoboJudge_Bench", id=r["item_id"], first_frame=f"{r['dataset']}/{r['task']}/{r['episode']}",
                         generator=r["model"], source=r["dataset"], cls=cls, robot=robot))
    return rows


def spread(cands, key, n):
    """n candidates, one per distinct key value first (e.g. one per robot), ties broken by a hash of the id."""
    pool = sorted(cands, key=lambda r: hashlib.sha1(str(r["_id"]).encode()).hexdigest())
    chosen, used, gens = [], set(), set()
    while len(chosen) < n and pool:
        best = min(pool, key=lambda r: (key(r) in used, r["_gen"] in gens, pool.index(r)))
        pool.remove(best)
        chosen.append(best)
        used.add(key(best))
        gens.add(best["_gen"])
    return chosen


def download(work):
    """Three single-arm examples per dataset, filed with each dataset's own store()."""
    examples = {}
    fs = load("PhysicsLENS", "fetch_sample")
    record = {}
    for member in fs.SHARED:
        fs.fetch(member, work / "pl", record)
    prompts = {int(r["id"]): r for r in csv.DictReader(open(work / "pl" / fs.DATA / "consolidated_prompts_80.csv", encoding="utf-8"))}
    annotations = list(csv.DictReader(open(work / "pl" / fs.DATA / "consolidated_annotations.csv", encoding="utf-8")))
    manifest = list(csv.DictReader(open(work / "pl" / "manifest.csv", encoding="utf-8")))
    model_to_folder = {v[2]: k for k, v in fs.GENERATORS.items()}
    cands = [dict(_id=a["video_file"] + a["model"], _gen=a["model"], tid=int(a["testset_id"]), a=a) for a in annotations
             if int(a["testset_id"]) in prompts and PL_SOURCES[pl_token(prompts[int(a["testset_id"])]["video_file"])][0] == "single"]
    rows = []
    for c in spread(cands, lambda r: r["tid"], PER_DATASET):
        a = c["a"]
        row = fs.store(c["tid"], model_to_folder[a["model"]], a["observability"] == "unobservable", work / "pl", record,
                       prompts, annotations, manifest)
        rows.append(dict(row, serial_robot=PL_SOURCES[pl_token(prompts[c["tid"]]["video_file"])][1]))
    examples["PhysicsLENS"] = rows
    fs = load("RoboJudge_Bench", "fetch_sample")
    record = {}
    released = [json.loads(line) for line in open(ROOT / "RoboJudge_Bench" / "metamanifest" / "test.jsonl", encoding="utf-8")]
    robot_of = {r["id"]: r["robot"] for r in robojudge_rows()}
    cands = [dict(_id=r["item_id"], _gen=fs.base_of(r["model"])[0], r=r) for r in released if robot_of[r["item_id"]] in
             ("Franka (DROID)", "WidowX-type arm (Bridge toy kitchen)", "White single arm, model not identified")]
    rows = []
    for c in spread(cands, lambda x: robot_of[x["_id"]], PER_DATASET):
        rows.append(dict(fs.store(c["r"], work / "rjb", record), serial_robot=robot_of[c["_id"]]))
    groups = load("RoboJudge_Bench", "fetch_agent_integrity_groups")
    v1 = {json.loads(line)["item_id"]: json.loads(line) for line in
          open(ROOT / "RoboJudge_Bench" / "metamanifest" / "test_final_800_20260916_review_repo.jsonl", encoding="utf-8")}
    for row in rows:  # both label versions, as on every other RoboJudge-Bench video (fetch_agent_integrity_groups.py)
        md = ROOT / "RoboJudge_Bench" / row["folder"] / "original" / "metadata"
        item = json.loads((md / "metadata.json").read_text(encoding="utf-8"))["item_id"]
        human = json.loads((md / "human_annotation.json").read_text(encoding="utf-8"))
        human["labels_20260916_review_repo"] = dict(
            source=f"github.com/{groups.REVIEW_REPO} @ {groups.REVIEW_COMMIT[:7]}, {groups.REVIEW_FILE}",
            note="The authors' earlier final labels; main scores mostly match the release, sub-scores often do not.",
            **{k: v1[item][k] for k in groups.LABEL_KEYS})
        (md / "human_annotation.json").write_text(json.dumps(human, indent=2, ensure_ascii=False) + "\n")
    examples["RoboJudge_Bench"] = rows
    for ds, rows in examples.items():
        with open(ROOT / ds / "metamanifest" / "serial_manipulator_examples.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return examples


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def pie_svg(counts, total, title):
    """Pie with a 2px surface gap between slices, slices in the fixed class order starting at 12 o'clock."""
    cx = cy = 160
    r = 120
    parts, angle = [], -math.pi / 2
    for key, label, color in CLASSES:
        n = counts.get(key, 0)
        if not n:
            continue
        sweep = 2 * math.pi * n / total
        x1, y1 = cx + r * math.cos(angle), cy + r * math.sin(angle)
        x2, y2 = cx + r * math.cos(angle + sweep), cy + r * math.sin(angle + sweep)
        big = 1 if sweep > math.pi else 0
        d = (f"M{cx},{cy} L{x1:.2f},{y1:.2f} A{r},{r} 0 {big} 1 {x2:.2f},{y2:.2f} Z" if n < total
             else f"M{cx},{cy - r} A{r},{r} 0 1 1 {cx - 0.01},{cy - r} Z")
        tip = f"{label}: {n} of {total} videos ({100 * n / total:.1f}%)"
        parts.append(f'<path d="{d}" fill="{color}" stroke="var(--surface)" stroke-width="2" stroke-linejoin="round"'
                     f' data-tip="{esc(tip)}"><title>{esc(tip)}</title></path>')
        mid = angle + sweep / 2
        share = 100 * n / total
        text = f"{share:.0f}%" if share >= 10 else f"{share:.1f}%"
        if share >= 8:  # direct label inside the slice, in text ink on a surface pill
            lx, ly = cx + r * 0.62 * math.cos(mid), cy + r * 0.62 * math.sin(mid)
            parts.append(f'<g class="lbl" pointer-events="none"><rect x="{lx - 24:.1f}" y="{ly - 11:.1f}" width="48" height="22" rx="11"/>'
                         f'<text x="{lx:.1f}" y="{ly + 4.5:.1f}" text-anchor="middle">{text}</text></g>')
        else:  # small slice: label just outside, with a short leader
            ex, ey = cx + (r + 4) * math.cos(mid), cy + (r + 4) * math.sin(mid)
            lx, ly = cx + (r + 22) * math.cos(mid), cy + (r + 22) * math.sin(mid)
            parts.append(f'<g class="lbl lbl--out" pointer-events="none"><line x1="{ex:.1f}" y1="{ey:.1f}" x2="{lx:.1f}" y2="{ly:.1f}"/>'
                         f'<text x="{lx:.1f}" y="{ly + (12 if math.sin(mid) > 0.3 else -4 if math.sin(mid) < -0.3 else 4):.1f}" '
                         f'text-anchor="{"start" if math.cos(mid) > 0.2 else "end" if math.cos(mid) < -0.2 else "middle"}">{text}</text></g>')
        angle += sweep
    return f'<svg viewBox="0 0 320 320" role="img" aria-label="{esc(title)}">{"".join(parts)}</svg>'


def chart_page(by_ds, examples):
    blocks = []
    for ds, rows in by_ds.items():
        total = len(rows)
        counts = Counter(r["cls"] for r in rows)
        robots = Counter((r["cls"], r["robot"]) for r in rows)
        frames = {k: len({r["first_frame"] for r in rows if r["cls"] == k}) for k, _, _ in CLASSES}
        single = counts.get("single", 0)
        legend = "".join(
            f'<tr><th scope="row"><span class="sw" style="background:{c}"></span>{esc(label)}</th><td class="n">{counts.get(k, 0)}</td>'
            f'<td class="n">{100 * counts.get(k, 0) / total:.1f}%</td><td class="n">{frames[k]}</td>'
            f'<td class="robots">{esc(" · ".join(f"{rb} {n}" for (kk, rb), n in sorted(robots.items(), key=lambda x: -x[1]) if kk == k)) or "—"}</td></tr>'
            for k, label, c in CLASSES)
        ex = "".join(f"<li>{esc(e['serial_robot'])} · {esc(e['generator'])} · <code>{esc(e['folder'])}</code></li>" for e in examples.get(ds, []))
        name = "PhysicsLENS" if ds == "PhysicsLENS" else "RoboJudge-Bench"
        blocks.append(f"""<section class="ds"><h2>{name}</h2>
<p class="headline"><b>{single}</b> of {total} videos ({100 * single / total:.1f}%) show a single-arm serial manipulator</p>
<div class="row">{pie_svg(counts, total, f"{name}: embodiment of {total} videos")}
<table><thead><tr><th scope="col">Embodiment</th><th scope="col">Videos</th><th scope="col">Share</th><th scope="col">First frames</th><th scope="col">Robots (videos)</th></tr></thead>
<tbody>{legend}</tbody></table></div>
{f'<p class="ex">Downloaded examples:</p><ul class="ex">{ex}</ul>' if ex else ""}</section>""")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Serial manipulators in PhysicsLENS and RoboJudge-Bench</title>
<style>
:root {{ color-scheme: light; --surface: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e; --rule: #e4e2dc; }}
body {{ margin: 0; background: var(--surface); color: var(--text-primary); font: 16px/1.5 "Instrument Sans", "Helvetica Neue", Arial, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 40px 32px 64px; }}
h1 {{ font: 500 30px/1.2 Georgia, serif; margin: 0 0 6px; }}
.sub {{ color: var(--text-secondary); margin: 0 0 32px; max-width: 80ch; }}
.ds {{ border-top: 1px solid var(--rule); padding-top: 24px; margin-top: 28px; }}
h2 {{ font: 500 24px/1.2 Georgia, serif; margin: 0 0 4px; }}
.headline {{ margin: 0 0 16px; color: var(--text-secondary); font-size: 17px; }}
.headline b {{ color: var(--text-primary); font-size: 22px; }}
.row {{ display: grid; grid-template-columns: 320px 1fr; gap: 32px; align-items: center; }}
svg {{ width: 300px; height: 300px; overflow: visible; }}
svg path {{ transition: opacity .15s; }} svg:hover path {{ opacity: .55; }} svg path:hover {{ opacity: 1; }}
.lbl rect {{ fill: var(--surface); opacity: .92; }} .lbl--out line {{ stroke: var(--text-secondary); stroke-width: 1; }} .lbl text {{ font: 600 13px sans-serif; fill: var(--text-primary); }}
table {{ border-collapse: collapse; width: 100%; font-size: 15px; }}
th, td {{ text-align: left; padding: 7px 10px 7px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }}
thead th {{ font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--text-secondary); font-weight: 600; }}
tbody th {{ font-weight: 600; white-space: nowrap; }}
td.n {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
td.robots {{ color: var(--text-secondary); font-size: 14px; }}
.sw {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 8px; vertical-align: -1px; }}
.ex {{ color: var(--text-secondary); font-size: 14px; margin: 12px 0 4px; }} ul.ex {{ margin-top: 0; }}
code {{ font: 13px ui-monospace, Menlo, monospace; }}
@media (max-width: 760px) {{ .row {{ grid-template-columns: 1fr; }} }}
</style></head><body><main>
<h1>Serial manipulators in PhysicsLENS and RoboJudge-Bench</h1>
<p class="sub">Embodiment in the first frame of every video. “Single-arm serial manipulator” = one standalone arm such as
Franka; dual-arm systems are also built from serial arms but are counted separately. Built by
<code>scripts/serial_manipulators.py</code>; per-video classes in <code>metamanifest/serial_manipulators/composition.csv</code>.</p>
{''.join(blocks)}
</main></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    by_ds = {"PhysicsLENS": physicslens_rows(), "RoboJudge_Bench": robojudge_rows()}
    examples = {}
    if not args.no_download:
        args.work.mkdir(parents=True, exist_ok=True)
        examples = download(args.work)
    else:
        for ds in by_ds:
            p = ROOT / ds / "metamanifest" / "serial_manipulator_examples.csv"
            examples[ds] = list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for v in by_ds.values() for r in v]
    with open(OUT / "composition.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {ds: dict(videos=len(v), first_frames=len({r["first_frame"] for r in v}),
                        classes={label: Counter(r["cls"] for r in v).get(k, 0) for k, label, _ in CLASSES},
                        robots=dict(Counter(r["robot"] for r in v).most_common()))
               for ds, v in by_ds.items()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    (OUT / "pies.html").write_text(chart_page(by_ds, examples), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
