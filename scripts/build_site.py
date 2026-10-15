"""Build the static dataset viewer into site/.

Usage: .venv/bin/python build_site.py [--jobs N]

Run quality_gate.py first. One page per dataset folder and one per episode folder, mirroring
the repository layout. Media are small browser-friendly derivatives (H.264 yuv420p, long side
<= 854 px, JPEG posters); full-resolution files are linked on GitHub. Derivatives newer than
their sources are reused. Links are relative, so the site works from disk and from
https://wilsonnijc-bot.github.io/robot_object_deformation_detect/robotvideo/.
"""
import argparse
import csv
import html
import json
import os
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
REPO = "https://github.com/Wilsonnijc-bot/robot_object_deformation_detect"
BRANCH = "data/robotvideo"

DATASETS = {
    "NVIDIA_GR1_100": dict(
        title="NVIDIA GR1", robot="Fourier GR1 humanoid, dexterous hands", view="First-person head camera",
        license="CC BY 4.0", source="https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1",
        blurb="Short single-task tabletop clips from the robot's forward-looking head camera, kept whole and unmodified. "
              "Prompts follow humanoid_action_v1."),
    "Humanoid_Everyday": dict(
        title="Humanoid Everyday", robot="Full-size humanoid", view="Third-person",
        license="Unclear (card: Apache-2.0; code: MIT)", source="https://humanoideveryday.github.io/",
        blurb="Everyday manipulation, hand-overs and loco-manipulation filmed from a fixed third-person camera. "
              "Complete original clips; prompts follow humanoid_action_v1."),
    "TAVP": dict(
        title="TAVP", robot="Franka Research 3, parallel gripper", view="Third-person",
        license="Not stated by the publisher", source="https://hcplab-sysu.github.io/TAVP/",
        blurb="Real-robot evaluation videos from the TAVP project page. Multi-step videos are shortened to the first "
              "complete manipulation; the full video is shown beside it. Prompts follow concise_action_v1."),
    "ActionNet": dict(
        title="ActionNet", robot="Fourier humanoid, dexterous hands", view="First-person top camera",
        license="CC BY-NC-SA 4.0", source="https://huggingface.co/datasets/FourierIntelligence/ActionNet",
        blurb="Pick-and-place, shelf and container episodes from the close top camera over the hands and work "
              "surface. Complete episodes; prompts follow humanoid_action_v1."),
    "RoboMIND": dict(
        title="RoboMIND", robot="Franka Panda (right camera) · Tien Kung humanoid", view="Third- and first-person",
        license="Apache-2.0", source="https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND",
        blurb="Franka episodes keep only the right RGB camera; Tien Kung episodes use the head camera, with red and blue "
              "swapped back because RoboMIND stores them channel-swapped. Frames are 640 × 480 JPEGs from HDF5 at a "
              "nominal 30 FPS."),
    "RoboCOIN_Airbot_MMK2_stack_block": dict(
        title="RoboCOIN · Airbot MMK2", robot="Dual-arm Airbot MMK2, five-finger hands", view="First-person head camera",
        license="Apache-2.0", source="https://huggingface.co/datasets/RoboCOIN/Airbot_MMK2_stack_block",
        blurb="Two-handed stacking: the left hand places the base object, then the right hand stacks on top. "
              "Admitted to the gate as a documented visual-review exception. Prompts follow humanoid_action_v1."),
}
# Episodes shown on the site per dataset (first N in folder order); all remain in the repository.
SITE_LIMIT = {"NVIDIA_GR1_100": 15}
GATE_LABEL = {"pass": "Pass", "exception": "Exception", "fail": "Fail"}
CONSTRAINT = ("Keep the camera fixed; preserve robot geometry, physically valid joint motion, "
              "and normal gripper opening and closing.")


def esc(s):
    return html.escape(str(s), quote=True)


def gh(kind, path):
    return f"{REPO}/{kind}/{BRANCH}/{quote(path.as_posix() if isinstance(path, Path) else path)}"


def read(path):
    return path.read_text(encoding="utf-8").strip() if path and path.exists() else None


def first(*paths):
    return next((p for p in paths if p.exists()), None)


def humanize(folder):
    leaf = folder.split("/")[0]
    if "__" in leaf:
        name = leaf.split("__", 1)[1]
        if leaf.split("_")[0] in ("franka", "tienkung"):
            name = name.split("__")[-1]
        return name.replace("_", " ").strip().capitalize()
    if folder.isdigit():
        return f"Video {folder}"
    leaf = folder.split("/")[-1]
    if leaf[:3].isdigit() and leaf[3:4] == "_":
        leaf = leaf[4:]
    return leaf.replace("_", " ").capitalize()


def probe(path):
    out = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,codec_name:format=duration", "-of", "json", str(path)]))
    s, f = out["streams"][0], out.get("format", {})
    num, den = (int(x) for x in s["avg_frame_rate"].split("/"))
    return dict(width=s["width"], height=s["height"], fps=round(num / den, 2) if den else None,
                frames=int(s["nb_frames"]) if s.get("nb_frames", "N/A").isdigit() else None,
                duration=round(float(f["duration"]), 2) if f.get("duration") else None, codec=s["codec_name"])


def stale(out, src):
    return not out.exists() or out.stat().st_mtime < src.stat().st_mtime


def transcode(src, out):
    if not stale(out, src):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-vf",
        "scale='min(854,iw)':'min(854,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,format=yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", "28", "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(out)


def stills(png, outdir):
    poster, frame = outdir / "poster.jpg", outdir / "frame.jpg"
    if stale(poster, png) or stale(frame, png):
        outdir.mkdir(parents=True, exist_ok=True)
        im = Image.open(png).convert("RGB")
        for out, side, q in ((poster, 560, 80), (frame, 1280, 88)):
            c = im.copy()
            c.thumbnail((side, side), Image.LANCZOS)
            c.save(out, "JPEG", quality=q, optimize=True, progressive=True)


def collect():
    episodes = []
    for row in csv.DictReader(open(ROOT / "metamanifest" / "quality_gate.csv")):
        ds, folder = row["dataset"], row["folder"]
        png = ROOT / row["first_frame"]
        sel = png.parent
        gen = sel.parent if sel.name == "first_substep" else sel
        ep_root = gen.parent
        meta = json.loads(read(sel / "metadata" / "metadata.json") or "{}")
        full_prompt = first(gen / "full_sequence_actiongeneration_prompt.txt",
                            gen / "prompt for original multi step action(reproducing the original video).txt")
        full_video = None
        if sel.name == "first_substep":
            full_video = first(ep_root / "original" / "video.mp4")
        elif full_prompt and meta.get("view"):
            full_video = first(ep_root / "original" / "decoded_videos" / f"{meta['view']}.mp4")
        review = first(sel / "metadata" / "action_review.json", gen / "metadata" / "action_review.json")
        video = sel / "video.mp4"
        if not video.exists():  # deduplicated: the generation video is the unchanged source in original/
            path = meta["relative_video_path"]
            video = ((sel / "metadata" / path) if path.startswith("../") else (ROOT / ds / path)).resolve()
        episodes.append(dict(
            dataset=ds, folder=folder, gate=row, png=png, sel=sel, gen=gen, ep_root=ep_root,
            video=video, full_video=full_video, meta=meta,
            prompt=read(first(sel / "actiongeneration_prompt.txt", sel / "prompt.txt")),
            full_prompt=read(full_prompt),
            publisher=read(first(ep_root / "original" / "task_prompt.txt", ep_root / "task_prompt.txt")),
            review=read(review), review_path=review,
            page=SITE / ds / folder, title=humanize(folder)))
    return episodes


def build_media(episodes, jobs):
    tasks = []
    for e in episodes:
        stills(e["png"], e["page"])
        tasks.append((e["video"], e["page"] / "clip.mp4"))
        if e["full_video"]:
            tasks.append((e["full_video"], e["page"] / "full.mp4"))
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(lambda t: transcode(*t), tasks))
    for e in episodes:
        e["info"] = probe(e["video"])


ANNOTATION_SECTIONS = [
    ("Observed outcome", ["observed_outcome", "selected_action_outcome", "outcome", "terminal_state",
                          "observed_terminal_state", "selected_execution_status_basis"]),
    ("Hand roles", ["arm_roles", "arm_hand_roles"]),
    ("Coordination", ["coordination"]),
    ("Robot-side evidence", ["anatomical_side_evidence"]),
    ("Chronology", ["chronology", "observed_sequence"]),
    ("Clip selection", ["selection_reason"]),
    ("Differences from the publisher annotation", ["source_annotation_discrepancies", "annotation_discrepancies",
                                                    "annotation_notes"]),
    ("Uncertainties", ["uncertainties"]),
]
FRAME_KEYS = ("frames", "source_frames_inclusive", "source_frames")
TEXT_KEYS = ("observation", "description", "action", "event", "summary")


def label(key):
    return key.replace("_", " ").capitalize()


def frame_span(item):
    for k in FRAME_KEYS:
        if isinstance(item.get(k), list) and item[k]:
            return "–".join(str(x) for x in item[k][:2])
    start = item.get("start_frame")
    end = next((item[k] for k in ("end_frame_inclusive", "end_frame", "end_frame_exclusive") if k in item), None)
    return f"{start}–{end}" if start is not None else ""


def is_chronology(value):
    return isinstance(value, list) and value and all(isinstance(i, dict) and frame_span(i) for i in value)


def chrono_text(item):
    text = next((item[k] for k in TEXT_KEYS if isinstance(item.get(k), str)), None)
    if text is None:
        rest = {k: v for k, v in item.items() if k not in FRAME_KEYS and not k.startswith(("start_frame", "end_frame"))}
        text = "; ".join(f"{label(k)}: {v}" for k, v in rest.items())
    return text


def render(value, depth=0):
    """Readable HTML for a review value of any shape."""
    if value is None or value == [] or value == {}:
        return '<p class="quiet">None recorded.</p>'
    if isinstance(value, (str, int, float, bool)):
        return f"<p>{esc(value)}</p>"
    if is_chronology(value):
        rows = "".join(f'<tr><th scope="row">{esc(frame_span(i))}</th><td>{esc(chrono_text(i))}</td></tr>' for i in value)
        return f'<table class="chrono"><thead><tr><th scope="col">Frames</th><th scope="col">Observation</th></tr></thead><tbody>{rows}</tbody></table>'
    if isinstance(value, list):
        return "<ul>" + "".join(f"<li>{esc(v) if not isinstance(v, (dict, list)) else render(v, depth + 1)}</li>" for v in value) + "</ul>"
    if depth > 2:
        return f"<p>{esc(json.dumps(value, ensure_ascii=False))}</p>"
    return '<dl class="kv">' + "".join(f"<dt>{esc(label(k))}</dt><dd>{render(v, depth + 1)}</dd>" for k, v in value.items()) + "</dl>"


def annotation_sections(review):
    out = []
    for title, keys in ANNOTATION_SECTIONS:
        values = [(k, review[k]) for k in keys if k in review and review[k] not in (None, "", [], {})]
        if values:
            out.append((title, values))
    return out


def annotation_text(e):
    """Plain-text annotation for one episode (published as annotation.txt)."""
    lines = [f"{DATASETS[e['dataset']]['title']} / {e['folder']}", ""]
    for head, text in (("Action generation prompt", e["prompt"]), ("Full-sequence prompt", e["full_prompt"]),
                       ("Publisher annotation", e["publisher"])):
        if text:
            lines += [head, text, ""]
    g = e["gate"]
    lines += ["Quality gate", f"{g['gate']}: sharpness {g['sharpness']} vs {g['threshold']} ({g['view']})", ""]
    review = json.loads(e["review"]) if e["review"] else {}

    def flat(value, indent=""):
        if is_chronology(value):
            return [f"{indent}frames {frame_span(i)}: {chrono_text(i)}" for i in value]
        if isinstance(value, list):
            return [x for v in value for x in (flat(v, indent + "  ") if isinstance(v, (dict, list)) else [f"{indent}- {v}"])]
        if isinstance(value, dict):
            return [x for k, v in value.items() for x in ([f"{indent}{label(k)}:"] + flat(v, indent + "  ")
                    if isinstance(v, (dict, list)) else [f"{indent}{label(k)}: {v}"])]
        return [f"{indent}{value}"]
    for title, values in annotation_sections(review):
        lines.append(title)
        for k, v in values:
            if len(values) > 1:
                lines.append(f"{label(k)}:")
            lines += flat(v)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_row(e):
    return {"dataset": e["dataset"], "folder": e["folder"], "title": e["title"],
            "action_generation_prompt": e["prompt"], "full_sequence_prompt": e["full_prompt"],
            "publisher_annotation": e["publisher"],
            "quality_gate": {k: e["gate"][k] for k in ("view", "sharpness", "threshold", "gate")},
            "repository_folder": e["ep_root"].relative_to(ROOT).as_posix(),
            "review": json.loads(e["review"]) if e["review"] else None}


def write_exports(directory, eps):
    with open(directory / "annotations.jsonl", "w", encoding="utf-8") as fh:
        for e in eps:
            fh.write(json.dumps(export_row(e), ensure_ascii=False) + "\n")
    with open(directory / "annotations.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "folder", "action_generation_prompt", "full_sequence_prompt", "publisher_annotation",
                    "gate_result", "gate_sharpness", "gate_threshold"])
        for e in eps:
            w.writerow([e["dataset"], e["folder"], e["prompt"] or "", e["full_prompt"] or "", e["publisher"] or "",
                        e["gate"]["gate"], e["gate"]["sharpness"], e["gate"]["threshold"]])


def chip(gate):
    g = gate["gate"]
    return (f'<span class="chip chip--{g}" title="Sharpness {esc(gate["sharpness"])} vs threshold {esc(gate["threshold"])}">'
            f'{GATE_LABEL[g]}<b>{float(gate["sharpness"]):.0f}</b></span>')


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
<script src="{rel}assets/app.js" defer></script>
</head>
<body>
<header class="masthead"><a class="wordmark" href="{rel}index.html">Robotvideo<span>dataset viewer</span></a>
<nav class="crumbs">{trail}</nav></header>
<main>
{body}
</main>
<footer class="foot"><p>Conditioning inputs for first-frame world-model generation. Previews are reduced H.264 copies;
full-resolution files live in the <a href="{gh('tree', '')}">repository</a>.</p></footer>
</body>
</html>
"""


def fmt_num(v, suffix=""):
    return "—" if v is None else f"{v:g}{suffix}"


def episode_page(e, prev, nxt):
    depth = len(Path(e["dataset"], e["folder"]).parts)
    rel = "../" * depth
    info, gate, ds = e["info"], e["gate"], DATASETS[e["dataset"]]
    rel_ep = e["ep_root"].relative_to(ROOT)
    prompt = (f'<p class="prompt">{esc(e["prompt"])}</p>' if e["prompt"] else
              '<p class="prompt prompt--missing">No reviewed action prompt yet. The publisher instruction is below.</p>')
    blocks = [f'<section class="field"><h2>Action prompt</h2>{prompt}</section>']
    if e["full_prompt"]:
        blocks.append(f'<section class="field"><h2>Full-sequence prompt</h2><p>{esc(e["full_prompt"])}</p></section>')
    if e["publisher"]:
        blocks.append(f'<section class="field"><h2>Publisher annotation</h2><p class="quiet">{esc(e["publisher"])}</p></section>')
    reason = e["meta"].get("data_quality_gate", {}).get("exception_reason")
    blocks.append(f"""<section class="field"><h2>Quality gate</h2>
<div class="gate">{chip(gate)}<p>Sharpness <b>{float(gate['sharpness']):.1f}</b> against {esc(gate['threshold'])}
({esc(gate['view'].replace('_', '-'))}). Measured on the {esc(gate['bucket'].replace('x', ' × '))} model view,
upscale {float(gate['upscale']):.2f}×.</p></div>{f'<p class="quiet">{esc(reason)}</p>' if reason else ''}</section>""")
    facts = [("Source frame", f"{gate['width']} × {gate['height']}"),
             ("Frames", fmt_num(info["frames"])), ("Frame rate", fmt_num(info["fps"], " fps")),
             ("Duration", fmt_num(info["duration"], " s")), ("Codec", info["codec"])]
    if e["meta"].get("variant"):
        facts.append(("Variant", e["meta"]["variant"].replace("_", " ")))
    blocks.append('<section class="field"><h2>Clip</h2><dl class="facts">' +
                  "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in facts) + "</dl></section>")
    links = [("Episode folder on GitHub", gh("tree", rel_ep)),
             ("First frame (PNG)", gh("raw", e["png"].relative_to(ROOT))),
             ("Generation video (original file)", gh("raw", e["video"].relative_to(ROOT)))]
    (e["page"] / "annotation.txt").write_text(annotation_text(e), encoding="utf-8")
    links.insert(0, ("Annotation (text)", "annotation.txt"))
    if e["review"]:
        (e["page"] / "review.json").write_text(json.dumps(json.loads(e["review"]), indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
        links.insert(1, ("Visual review (JSON)", "review.json"))
    blocks.append('<section class="field"><h2>Files</h2><ul class="links">' +
                  "".join(f'<li><a href="{esc(h)}">{esc(t)}</a></li>' for t, h in links) + "</ul></section>")
    full = ""
    if e["full_video"]:
        full = f"""<figure class="media media--secondary"><video controls playsinline preload="none" poster="frame.jpg" src="full.mp4"></video>
<figcaption>Complete source video. The generation input above is its first {fmt_num(info['frames'])} frames.</figcaption></figure>"""
    review = ""
    if e["review"]:
        data = json.loads(e["review"])
        parts = []
        for title, values in annotation_sections(data):
            inner = "".join((f"<h4>{esc(label(k))}</h4>" if len(values) > 1 else "") + render(v) for k, v in values)
            parts.append(f'<section class="anno"><h3>{esc(title)}</h3>{inner}</section>')
        review = f"""<section class="annotation"><h2>Annotation</h2>
<p class="quiet">From the dedicated visual review of this video ({esc(data.get('prompt_protocol', 'review'))}).
<a href="annotation.txt">Plain text</a> · <a href="review.json">JSON</a></p>
<div class="annotation__grid">{''.join(parts)}</div>
<details class="review"><summary>Complete review record <span>{esc(e['review_path'].name)}</span></summary>
<pre>{esc(json.dumps(data, indent=2, ensure_ascii=False))}</pre></details></section>"""
    nav = '<nav class="pager">' + (
        f'<a rel="prev" href="{rel}{esc(quote(prev["dataset"] + "/" + prev["folder"]))}/index.html">← {esc(prev["title"])}</a>' if prev else "<span></span>") + (
        f'<a rel="next" href="{rel}{esc(quote(nxt["dataset"] + "/" + nxt["folder"]))}/index.html">{esc(nxt["title"])} →</a>' if nxt else "<span></span>") + "</nav>"
    body = f"""<article class="episode">
<header class="episode__head"><p class="kicker">{esc(ds['title'])} · {esc(e['folder'])}</p>
<h1>{esc(e['title'])}</h1></header>
<div class="episode__grid">
<div class="episode__media">
<figure class="media"><video controls playsinline preload="metadata" poster="frame.jpg" src="clip.mp4"></video>
<figcaption>Generation input · {fmt_num(info['frames'])} frames · {fmt_num(info['duration'], ' s')}</figcaption></figure>
<figure class="media media--still"><img src="frame.jpg" alt="Conditioning first frame" loading="lazy">
<figcaption>Conditioning first frame</figcaption></figure>
{full}
</div>
<aside class="episode__side">{''.join(blocks)}</aside>
</div>
{review}
{nav}
</article>"""
    crumbs = [("All datasets", f"{rel}index.html"), (ds["title"], f"{rel}{e['dataset']}/index.html")]
    (e["page"] / "index.html").write_text(page(f"{e['title']} · {ds['title']}", rel, body, crumbs))


def card(e, rel_from, grouped=False):
    base = quote(os.path.relpath(e["page"], rel_from))
    text = (e["prompt"] or e["publisher"] or "").replace(CONSTRAINT, "").strip()
    name = e["folder"].split("/")[-1] if grouped else e["folder"]
    return f"""<article class="card" data-gate="{esc(e['gate']['gate'])}" data-text="{esc((e['title'] + ' ' + e['folder'] + ' ' + text).lower())}"
 data-title="{esc(e['title'])}" data-id="{esc(e['folder'])}" data-href="{base}/index.html" data-clip="{base}/clip.mp4"
 data-poster="{base}/frame.jpg" data-prompt="{esc(e['prompt'] or '')}" data-publisher="{esc(e['publisher'] or '')}">
<button type="button" class="card__img" aria-label="Play {esc(e['title'])} and show its prompt"><img src="{base}/poster.jpg" alt="" loading="lazy"><span class="card__play" aria-hidden="true"></span></button>
<span class="card__meta"><a class="card__title" href="{base}/index.html">{esc(e['title'])}</a>{chip(e['gate'])}</span>
<span class="card__id">{esc(name)}</span>
<span class="card__text">{esc(text)}</span></article>"""


PLAYER = """<dialog class="player" aria-labelledby="player-title">
<div class="player__frame">
<video controls playsinline></video>
<div class="player__text">
<p class="kicker player__id"></p>
<h2 id="player-title"></h2>
<h3>Action generation prompt</h3>
<p class="prompt player__prompt"></p>
<div class="player__pub"><h3>Publisher annotation</h3><p class="quiet player__publisher"></p></div>
<p class="player__links"><a class="player__open" href="#">Full annotation and files →</a></p>
<p class="player__hint">← → previous / next · Esc closes</p>
</div>
<button type="button" class="player__close" aria-label="Close player" autofocus>×</button>
</div>
</dialog>"""


def gate_summary(eps):
    c = Counter(e["gate"]["gate"] for e in eps)
    return " · ".join(f'<span class="dot dot--{g}"></span>{c[g]} {GATE_LABEL[g].lower()}' for g in GATE_LABEL if c[g])


def dataset_page(ds, eps, total):
    meta, d = DATASETS[ds], SITE / ds
    groups = {}
    for e in eps:
        groups.setdefault(e["folder"].split("/")[0] if "/" in e["folder"] else "", []).append(e)
    sections = []
    for name, items in groups.items():
        head = f'<h2 class="group__title">{esc(humanize(name))}<span>{esc(name)}</span></h2>' if name else ""
        sections.append(f'<section class="group">{head}<div class="cards">{"".join(card(e, d, bool(name)) for e in items)}</div></section>')
    c = Counter(e["gate"]["gate"] for e in eps)
    filters = "".join(f'<button type="button" data-filter="{g}" aria-pressed="false">{GATE_LABEL[g]} <span>{c[g]}</span></button>'
                      for g in GATE_LABEL if c[g])
    body = f"""<header class="intro">
<p class="kicker">{esc(meta['view'])} · {esc(meta['robot'])}</p>
<h1>{esc(meta['title'])}</h1>
<p class="lede">{esc(meta['blurb'])}</p>
<dl class="specs"><dt>Episodes</dt><dd>{len(eps) if total == len(eps) else f"{len(eps)} shown of {total}; all {total} are in the repository folder"}</dd><dt>Gate</dt><dd>{gate_summary(eps)}</dd>
<dt>License</dt><dd>{esc(meta['license'])}</dd><dt>Source</dt><dd><a href="{esc(meta['source'])}">{esc(meta['source'].split('//')[1].rstrip('/'))}</a></dd>
<dt>Folder</dt><dd><a href="{gh('tree', ds)}">{esc(ds)}/</a></dd>
<dt>Annotations</dt><dd><a href="annotations.jsonl">JSONL</a> · <a href="annotations.csv">CSV</a> ({total} episodes, with full reviews)</dd></dl>
</header>
<div class="toolbar" role="search"><div class="filters"><button type="button" data-filter="all" aria-pressed="true">All <span>{len(eps)}</span></button>{filters}</div>
<input type="search" placeholder="Search prompts and folders" aria-label="Search episodes"><output class="count" aria-live="polite"></output></div>
{''.join(sections)}
<p class="empty" hidden>No episodes match. Clear the search or choose another gate filter.</p>
{PLAYER}"""
    (d / "index.html").write_text(page(f"{meta['title']} · Robotvideo", "../", body, [("All datasets", "../index.html")]))


def index_page(by_ds, totals):
    rows = []
    for ds, eps in by_ds.items():
        meta = DATASETS[ds]
        step = max(1, len(eps) // 5)
        strip = "".join(f'<img src="{quote(os.path.relpath(e["page"] / "poster.jpg", SITE))}" alt="" loading="lazy">' for e in eps[::step][:5])
        rows.append(f"""<a class="set" href="{quote(ds)}/index.html">
<span class="set__strip">{strip}</span>
<span class="set__body"><span class="set__title">{esc(meta['title'])}</span>
<span class="set__robot">{esc(meta['robot'])} · {esc(meta['view'])}</span>
<span class="set__stats"><b>{len(eps)}</b> {"episodes" if totals[ds] == len(eps) else f"of {totals[ds]} episodes shown"} · {gate_summary(eps)} · {esc(meta['license'])}</span></span></a>""")
    total = [e for eps in by_ds.values() for e in eps]
    body = f"""<header class="intro intro--home">
<h1>Robot videos for first-frame world-model generation</h1>
<p class="lede">{sum(totals.values())} reviewed episodes from {len(by_ds)} datasets, {len(total)} shown here. Each input is a first frame, an action prompt
written from the video, and the reference clip. Every input is scored on the sharpness of its upscaled
conditioning frame: third-person at least 120, first-person at least 60.</p>
<p class="gate-line">{gate_summary(total)} · <a href="{gh('blob', 'data_quality_gate.md')}">How the gate works</a>
· All annotations: <a href="annotations.jsonl">JSONL</a> · <a href="annotations.csv">CSV</a></p>
</header>
<div class="sets">{''.join(rows)}</div>"""
    (SITE / "index.html").write_text(page("Robotvideo dataset viewer", "", body))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    episodes = collect()
    totals = Counter(e["dataset"] for e in episodes)
    everything = episodes
    episodes = [e for ds in DATASETS for e in [x for x in episodes if x["dataset"] == ds][:SITE_LIMIT.get(ds)]]
    unknown = set(totals) - set(DATASETS)
    if unknown:
        raise SystemExit(f"Add {sorted(unknown)} to DATASETS in build_site.py")
    keep = {e["page"] for e in episodes}
    for ds in DATASETS:
        for idx in list((SITE / ds).rglob("index.html")) if (SITE / ds).exists() else []:
            if idx.parent != SITE / ds and idx.parent not in keep and idx.parent.exists():
                shutil.rmtree(idx.parent)
    for stale_ds in (p for p in SITE.iterdir() if p.is_dir() and p.name not in DATASETS and p.name != "assets") if SITE.exists() else []:
        shutil.rmtree(stale_ds)
    build_media(episodes, args.jobs)
    by_ds = {ds: [e for e in episodes if e["dataset"] == ds] for ds in DATASETS if any(e["dataset"] == ds for e in episodes)}
    for eps in by_ds.values():
        for i, e in enumerate(eps):
            episode_page(e, eps[i - 1] if i else None, eps[i + 1] if i + 1 < len(eps) else None)
    for ds, eps in by_ds.items():
        dataset_page(ds, eps, totals[ds])
        write_exports(SITE / ds, [e for e in everything if e["dataset"] == ds])
    write_exports(SITE, [e for ds in by_ds for e in everything if e["dataset"] == ds])
    index_page(by_ds, totals)
    (SITE / ".nojekyll").touch()
    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"{len(episodes)} episode pages, {len(by_ds)} dataset pages, {size / 1e6:.1f} MB -> {SITE}")


if __name__ == "__main__":
    main()
