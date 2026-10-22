"""Stage the full filtered MiraBench, PhysicsLENS and RoboJudge-Bench sets, with their human annotations, in the
worldgen organized_by_generator layout for upload to ERIS scratch (existing_annotations/).

Usage: .venv/bin/python -I scripts/stage_existing_annotations.py /tmp/new_empty_work_dir [--jobs 4]
Then:  tar -C <work> -cf - existing_annotations | ssh <eris> 'tar -C ~/scratch -xf -'   (ERIS has no rsync),
       check MANIFEST.csv with sha256sum on the server, and delete <work>.

The Mac is only a transfer step: everything is written under the work directory, nothing in this repository changes.
Every file is checked against the Hub's X-Linked-Etag by each dataset's own fetch() (pinned revisions).

Filters (the same scopes as the local review site, scripts/build_eval_site.py):
  MiraBench        physical_consistency only (no action-following), duration <= 12.125 s, i.e. the 12.1 s pour20
                   videos are kept and the 24.1 s DreamDojo-14B/2B occ68 videos are not: 88 videos.
  PhysicsLENS      every annotated generated video (observable and hidden-property prompts); the real source videos,
                   the upscaled_720p copies and generated videos without an annotation row are not included.
  RoboJudge-Bench  robot first frames only (EgoDex-human, EgoScaler, EPIC-KITCHENS excluded): 585 of 800. Both label
                   versions are kept: v1 = the authors' review-repo file test_final_800_20260916.jsonl (869bebe),
                   v2 = the released data/test.jsonl (Hub revision 13a0c19, 2026-09-29).

Layout of <work>/existing_annotations/ (generator subfolders as in worldgen/organized_by_generator/<gen>/<dataset>/):
  README.md, MANIFEST.csv, summary.json
  <dataset>/README.md, <dataset>/annotations/            publisher tables and a flat per-video table
  <dataset>/<generator>/prompts.csv                      generator,dataset,number,id,prompt,task_description
  <dataset>/<generator>/NNN_<id>.mp4                     unchanged upstream video
  <dataset>/<generator>/json/NNN_<id>.json               provenance, prompt/task, checksum, clip facts
  <dataset>/<generator>/annotations/NNN_<id>.json        the human annotation
  <dataset>/<generator>/first_frame/NNN_<id>.png         RoboJudge-Bench only: the conditioning frame
"""
import argparse
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "~/scratch/existing_annotations"  # on the ERIS cluster (scratch space of the uploading user)


def retrying(fetch):
    """The Hub answers bursts of anonymous requests with 429: back off and retry."""
    def wrapped(*args, **kwargs):
        for attempt in range(8):
            try:
                return fetch(*args, **kwargs)
            except (urllib.error.HTTPError, urllib.error.URLError, ConnectionError, TimeoutError) as error:
                if getattr(error, "code", 429) not in (429, 500, 502, 503, 504) or attempt == 7:
                    raise
                time.sleep(min(120, 5 * 2 ** attempt))
    return wrapped


def load(ds, name):
    sys.modules.pop("fetch_sample", None)  # each dataset's group script imports its own fetch_sample by that name
    spec = importlib.util.spec_from_file_location(f"{ds}_{name}", ROOT / ds / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "fetch"):
        mod.fetch = retrying(mod.fetch)  # store() looks fetch up in its module at call time
    return mod


def probe(path):
    out = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=width,height,r_frame_rate,nb_read_packets,codec_name:format=duration", "-of", "json", str(path)]))
    s, num_den = out["streams"][0], out["streams"][0]["r_frame_rate"].split("/")
    return dict(width=s["width"], height=s["height"], frames=int(s["nb_read_packets"]), codec=s["codec_name"],
                fps=round(int(num_den[0]) / int(num_den[1]), 3), duration_s=round(float(out["format"]["duration"]), 3))


def place(entries, out, ds):
    """entries: dicts with generator (folder name), id, video, prompt, task, json, annotation, first_frame (optional).
    Numbers videos per generator (sorted by id) and writes the worldgen-style files; returns manifest rows."""
    rows = []
    by_gen = {}
    for e in entries:
        by_gen.setdefault(e["generator"], []).append(e)
    for gen, items in sorted(by_gen.items()):
        d = out / ds / gen
        (d / "json").mkdir(parents=True, exist_ok=True)
        (d / "annotations").mkdir(exist_ok=True)
        prompts = []
        for n, e in enumerate(sorted(items, key=lambda e: e["id"]), 1):
            name = f"{n:03d}_{e['id']}"
            shutil.copy2(e["video"], d / f"{name}.mp4")
            facts = probe(d / f"{name}.mp4")
            record = dict(id=f"{ds}__{gen}__{e['id']}", number=n, file=f"{name}.mp4", **e["json"], clip=facts)
            if e.get("first_frame"):
                (d / "first_frame").mkdir(exist_ok=True)
                shutil.copy2(e["first_frame"], d / "first_frame" / f"{name}.png")
                record["first_frame"] = f"first_frame/{name}.png"
            (d / "json" / f"{name}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
            (d / "annotations" / f"{name}.json").write_text(json.dumps(e["annotation"], indent=1, ensure_ascii=False) + "\n")
            prompts.append(dict(generator=gen, dataset=ds, number=n, id=e["id"], prompt=e["prompt"] or "",
                                task_description=e["task"]))
            rows.append(dict(dataset=ds, generator=gen, number=n, id=e["id"], video=f"{ds}/{gen}/{name}.mp4",
                             json=f"{ds}/{gen}/json/{name}.json", annotation=f"{ds}/{gen}/annotations/{name}.json",
                             sha256=e["json"]["sha256"], bytes=(d / f"{name}.mp4").stat().st_size,
                             duration_s=facts["duration_s"], upstream_path=e["json"]["upstream_path"],
                             prompt=e["prompt"] or "", task_description=e["task"], key_labels=e["key"]))
        with open(d / "prompts.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(prompts[0]))
            w.writeheader()
            w.writerows(prompts)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------------------------------- MiraBench
def mirabench(work, out, jobs):
    fs, groups = load("MiraBench", "fetch_sample"), load("MiraBench", "fetch_label_groups")
    raw = work / "raw" / "MiraBench"
    fs.DATASET = raw  # store() files into the work directory, not the repository
    record = {}
    for member in fs.SHARED:
        fs.fetch(member, work / "dl_mira", record)
    parsed = {d: list(csv.DictReader(open(work / "dl_mira" / fs.ROOT / "parsed" / f"{d}.csv", encoding="utf-8")))
              for d in fs.DIMENSIONS}
    model_of = {r["rel_path"]: r["model"] for r in parsed[fs.PC]}
    scoped = sorted(k for k in model_of if fs.in_scope(fs.PC, *groups.locate(k, model_of[k])[:2]))
    with ThreadPoolExecutor(jobs) as pool:
        rows = list(pool.map(lambda k: fs.store(fs.PC, *groups.locate(k, model_of[k]), work / "dl_mira", record, parsed),
                             scoped))
    entries, flat = [], []
    for row in rows:
        md = raw / row["folder"] / "original" / "metadata"
        meta = json.loads((md / "metadata.json").read_text(encoding="utf-8"))
        human = json.loads((md / "human_annotation.json").read_text(encoding="utf-8"))
        publisher = json.loads(next(p for p in md.glob("*.json") if p.name not in ("metadata.json", "human_annotation.json"))
                               .read_text(encoding="utf-8"))
        model_key = row["folder"].split("/")[1].rsplit("_", 1)[0]
        grades = {i["code"]: i for i in human["items"] if i["code"]}
        task = (f"{meta['subset']} subset, input {meta['input_id']}: {meta['scene_group']}. No text prompt is published; "
                f"the video is conditioned on the first frame and a robot action sequence.")
        key = "; ".join([f"overall {human['headline']['answer_en']}"] + [
            f"{c} {grades[c]['grade'] or 'NA'}" for c in ("IC_0", "IC_1", "IC_2", "IC_3") if c in grades])
        entries.append(dict(generator=model_key, id=meta["input_id"], video=raw / row["video"], prompt="", task=task, key=key,
                            json=dict(meta, prompt=None, task_description=task, subset_scene=meta["scene_group"]),
                            annotation=dict(human, publisher_record=publisher)))
        flat.append(dict(generator=model_key, id=meta["input_id"], subset=meta["subset"], upstream_path=meta["upstream_path"],
                         **{f"{c}_grade": (grades[c]["grade"] or "") if c in grades else "" for c in sorted(grades)},
                         **{f"{c}_score": ("" if grades[c]["score"] is None else grades[c]["score"]) for c in sorted(grades)}))
    placed = place(entries, out, "MiraBench")
    ann = out / "MiraBench" / "annotations"
    (ann / "parsed").mkdir(parents=True, exist_ok=True)
    for f in ["summary.txt"] + [f"{d}.csv" for d in fs.DIMENSIONS]:
        shutil.copy2(work / "dl_mira" / fs.ROOT / "parsed" / f, ann / "parsed" / f)
    write_csv(ann / "physical_consistency_in_scope.csv", [r for r in parsed[fs.PC] if r["rel_path"] in set(scoped)])
    cols = sorted({k for r in flat for k in r}, key=lambda k: (k not in ("generator", "id", "subset", "upstream_path"), k))
    write_csv(ann / "annotations_flat.csv", [{k: r.get(k, "") for k in cols} for r in flat])
    shutil.copy2(work / "dl_mira" / "README.md", ann / "source_README.md")
    shutil.copy2(work / "dl_mira" / fs.ROOT / "README.md", ann / "source_dataset_README.md")
    (ann / "download_record.json").write_text(json.dumps(dict(repository=fs.REPO, revision=fs.REVISION, members=record), indent=1) + "\n")
    return placed


# -------------------------------------------------------------------------------------------------- PhysicsLENS
def tree(repo, revision, path):
    files, url = [], f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{path}?recursive=true&limit=1000"
    while url:
        with urllib.request.urlopen(url) as r:
            files += json.load(r)
            link = r.headers.get("Link", "")
        url = link.split(";")[0].strip("<>") if 'rel="next"' in link else None
    return [f["path"] for f in files if f["type"] == "file"]


def physicslens(work, out, jobs):
    fs, norm = load("PhysicsLENS", "fetch_sample"), load("PhysicsLENS", "normalize_annotations")
    dl, record = work / "dl_pl", {}
    for member in fs.SHARED:
        fs.fetch(member, dl, record)
    prompts = {int(r["id"]): r for r in csv.DictReader(open(dl / fs.DATA / "consolidated_prompts_80.csv", encoding="utf-8"))}
    annotations = list(csv.DictReader(open(dl / fs.DATA / "consolidated_annotations.csv", encoding="utf-8")))
    manifest = list(csv.DictReader(open(dl / "manifest.csv", encoding="utf-8")))
    codebook = json.loads((ROOT / "PhysicsLENS" / "metamanifest" / "annotation_codebook.json").read_text(encoding="utf-8"))
    labels = list(codebook["labels"])
    folders = {}  # upstream folder -> (generator folder, unobservable?)
    for gen, (title, unobs_dir, model, tag, suffix) in fs.GENERATORS.items():
        folders[gen], folders[unobs_dir] = (gen, False), (gen, True)
    files = tree(fs.REPO, fs.REVISION, fs.DATA)
    mp4s = sorted(p for p in files if p.endswith(".mp4") and p.split("/")[1] in folders)
    jsons = set(files)

    def stage(path):
        upstream, name = path.split("/")[1], path.split("/")[2]
        gen, unobs = folders[upstream]
        title, _, model, _, suffix = fs.GENERATORS[gen]
        names = {name, name.replace(suffix + ".mp4", ".mp4")} if suffix else {name}
        matches = [a for a in annotations if a["video_file"] in names and a["model"] == model
                   and a["observability"] == ("unobservable" if unobs else "observable")]
        assert len(matches) <= 1, (path, len(matches))
        if not matches:
            return dict(missing=path)  # no human annotation upstream: listed, not staged
        a = matches[0]
        mp4 = fs.fetch(path, dl, record)
        gpath = path[:-4] + ".json"
        gmeta = json.loads(fs.fetch(gpath, dl, record).read_text(encoding="utf-8")) if gpath in jsons else None
        tid = int(a["testset_id"])
        p = prompts.get(tid)
        try:
            group, robot = fs.robot_of(name)
        except StopIteration:
            group, robot = ("Human hands · EgoDex", "EgoDex human hands (GR00T)") if "egodex" in name else \
                ("Other robots", name.split("_", 1)[1].rsplit("_ep", 1)[0])
        human = dict(
            source="physicslens_robot_data/consolidated_annotations.csv", annotator=a["annotator"],
            physical_plausibility_1_4=int(a["physical_plausibility_1_4"]),
            hidden_property_followed_1_4=int(a["hidden_property_followed_1_4"]) if a["hidden_property_followed_1_4"] else None,
            action_completed=a["action_completed"], observability=a["observability"],
            physics_category=[c.strip() for c in a["physics_category"].split(",") if c.strip()],
            description_of_issue=a["description_of_issue"] or None, task=p["task"] if p else None,
            hidden_property=p["hidden_property"] if (p and unobs) else None,
            hidden_property_value=p["hidden_property_value"] if (p and unobs) else None,
            expected_outcome=p["expected_outcome"] if (p and unobs) else None,
            failure_signature=p["failure_signature"] if (p and unobs) else None,
            raw_row=a, normalized=norm.normalize(a, labels),
            codebook="annotations/annotation_codebook.json (quoted from arXiv:2610.01162, Tables 8 and 9)")
        prompt = (gmeta or {}).get("prompt") or ""
        task = p["task"] if p else "(no row in consolidated_prompts_80.csv upstream)"
        meta = dict(dataset="PhysicsLENS", repository=fs.REPO, revision=fs.REVISION, upstream_path=path,
                    sha256=record[path]["sha256"], generator=title, generator_folder=upstream,
                    prompt_set="unobservable" if unobs else "observable", prompt=prompt or None, task_description=task,
                    testset_id=tid, source_dataset=next((r["source_dataset"] for r in manifest if int(r["frame_id"]) == tid), None),
                    robot_group=group, robot=robot, publisher_generation_record=gmeta,
                    license="Annotations and prompts CC BY-NC-SA 4.0; frames keep their source license")
        n = human["normalized"]
        key = f"P {n['plausibility_1_4']}/4; labels: {', '.join(n['label_order']) or 'none'}"
        return dict(generator=gen, id=Path(name).stem, video=mp4, prompt=prompt, task=task, key=key, json=meta,
                    annotation=human, flat={**{k: a[k] for k in ("video_file", "testset_id", "model", "observability")},
                                            "generator_folder": gen, "robot_group": group,
                                            **{k: ("; ".join(v) if isinstance(v, list) else "" if v is None else v)
                                               for k, v in n.items()}})

    with ThreadPoolExecutor(jobs) as pool:
        entries = list(pool.map(stage, mp4s))
    unannotated = [e["missing"] for e in entries if "missing" in e]
    entries = [e for e in entries if "missing" not in e]
    used = {(e["annotation"]["raw_row"]["video_file"], e["annotation"]["raw_row"]["model"],
             e["annotation"]["raw_row"]["observability"]) for e in entries}
    orphans = [a for a in annotations if (a["video_file"], a["model"], a["observability"]) not in used]
    placed = place(entries, out, "PhysicsLENS")
    ann = out / "PhysicsLENS" / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    for member, name in zip(fs.SHARED, ["source_README.md", "manifest.csv", "consolidated_annotations.csv",
                                        "consolidated_prompts_80.csv"]):
        shutil.copy2(dl / member, ann / name)
    shutil.copy2(ROOT / "PhysicsLENS" / "metamanifest" / "annotation_codebook.json", ann / "annotation_codebook.json")
    write_csv(ann / "annotations_flat.csv", [e["flat"] for e in entries])
    if orphans:
        write_csv(ann / "annotation_rows_without_video.csv", orphans)
    if unannotated:
        write_csv(ann / "videos_without_annotation.csv", [dict(upstream_path=p) for p in unannotated])
    (ann / "download_record.json").write_text(json.dumps(dict(repository=fs.REPO, revision=fs.REVISION, members=record), indent=1) + "\n")
    return placed, len(orphans), len(unannotated)


# --------------------------------------------------------------------------------------------- RoboJudge-Bench
def robojudge(work, out, jobs):
    fs, groups = load("RoboJudge_Bench", "fetch_sample"), load("RoboJudge_Bench", "fetch_agent_integrity_groups")
    raw = work / "raw" / "RoboJudge_Bench"
    fs.DATASET = raw
    dl, record = work / "dl_rjb", {}
    for member in ["README.md", "data/test.jsonl", "metadata/summary.json"]:
        fs.fetch(member, dl, record)
    released = [json.loads(line) for line in open(dl / "data" / "test.jsonl", encoding="utf-8")]
    v1_path = ROOT / "RoboJudge_Bench" / "metamanifest" / "test_final_800_20260916_review_repo.jsonl"
    v1 = {json.loads(line)["item_id"]: json.loads(line) for line in open(v1_path, encoding="utf-8")}
    robots = [r for r in released if r["dataset"] not in groups.HUMAN_HAND]
    with ThreadPoolExecutor(jobs) as pool:
        rows = list(pool.map(lambda r: (r, fs.store(r, dl, record)), robots))
    keys = groups.LABEL_KEYS
    entries, flat = [], []
    for r, row in rows:
        orig = raw / row["folder"] / "original"
        meta = json.loads((orig / "metadata" / "metadata.json").read_text(encoding="utf-8"))
        human = json.loads((orig / "metadata" / "human_annotation.json").read_text(encoding="utf-8"))
        e1 = v1[r["item_id"]]
        human["label_versions"] = {
            "v1": dict(source=f"github.com/{groups.REVIEW_REPO} @ {groups.REVIEW_COMMIT[:7]}, {groups.REVIEW_FILE} "
                              "(the authors' earlier final labels, 2026-09-16)", **{k: e1[k] for k in keys}),
            "v2": dict(source=f"huggingface.co/datasets/{fs.REPO} @ {fs.REVISION[:7]}, data/test.jsonl "
                              "(released labels, Hub commit 2026-09-29)", **{k: r[k] for k in keys})}
        human["agent_integrity"] = dict(v1=e1["agent_consistency"], v2=r["agent_consistency"],
                                        note="agent_consistency, 0-2, higher is better; v1 and v2 agree for "
                                             "443 of all 800 benchmark videos")
        base, variant = fs.base_of(r["model"])
        run = meta["run"]
        vid = f"{r['dataset']}__{r['task']}__{r['episode']}__{variant}" + ("" if run == "1" else f"__run{run}")
        instruction = (orig / "prompt" / "instruction.txt")
        task = r["instruction"]
        meta.update(prompt=None, task_description=task, prompt_note=(
            f"The benchmark releases the instruction, not the exact {variant} generation prompt; prompts.csv repeats the "
            "instruction"), instruction_file_text=instruction.read_text(encoding="utf-8").strip() if instruction.exists() else None)
        key = (f"PA v2 {r['physical_adherence']} (v1 {e1['physical_adherence']}); agent integrity v2 "
               f"{r['agent_consistency']} (v1 {e1['agent_consistency']})")
        entries.append(dict(generator=base, id=vid, video=orig / meta["video_file"], prompt=task, task=task, key=key,
                            json=meta, annotation=human, first_frame=orig / "prompt" / "init_frame.png"))
        flat.append(dict(item_id=r["item_id"], generator=base, prompt_variant=variant, source_corpus=r["dataset"],
                         task=r["task"], episode=r["episode"], instruction=task,
                         **{f"{k}_v2": r[k] for k in keys}, **{f"{k}_v1": e1[k] for k in keys}))
    placed = place(entries, out, "RoboJudge_Bench")
    ann = out / "RoboJudge_Bench" / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dl / "data" / "test.jsonl", ann / "test_v2_released.jsonl")
    shutil.copy2(v1_path, ann / "test_final_800_20260916_v1_review_repo.jsonl")
    shutil.copy2(dl / "metadata" / "summary.json", ann / "summary.json")
    shutil.copy2(dl / "README.md", ann / "source_README.md")
    write_csv(ann / "annotations_flat_v1_v2.csv", flat)
    (ann / "download_record.json").write_text(json.dumps(dict(repository=fs.REPO, revision=fs.REVISION, members=record), indent=1) + "\n")
    return placed


FILTERS = {
    "MiraBench": "Physical consistency only (no action-following), duration <= 12.1 s: the 12.125 s DreamDojo-14B pour20 "
                 "videos are kept, the 24.1 s DreamDojo-14B/2B occ68 videos are not.",
    "PhysicsLENS": "Every generated video with a human annotation, observable and hidden-property (unobs) prompts; the "
                   "real source videos, the upscaled_720p copies and unannotated videos are left out.",
    "RoboJudge_Bench": "Robot first frames only (agibot_world, robotwin, gr1_inlab, open_x_embodiment, droid); the "
                       "human-hand corpora egodex_human, egoscaler_human and epickitchens_human are left out.",
}
ANNOTATIONS = {
    "MiraBench": "annotations/NNN_<id>.json: the decoded physical-consistency questionnaire (Chinese answer, English gloss, "
                 "publisher grade A-D and score 1/0.67/0.33/0 per item; higher score, lower error) plus the publisher's raw "
                 "record. Dataset tables: annotations/parsed/*.csv (publisher), physical_consistency_in_scope.csv, "
                 "annotations_flat.csv (one row per video, grade and score per item code).",
    "PhysicsLENS": "annotations/NNN_<id>.json: plausibility 1-4, action completed, hidden-property adherence, issue "
                   "description, the eight violation labels and their 0/1 normalization. Dataset tables: "
                   "consolidated_annotations.csv (publisher), annotations_flat.csv (normalized), annotation_codebook.json "
                   "(definitions quoted from arXiv:2610.01162).",
    "RoboJudge_Bench": "annotations/NNN_<id>.json: PA, IA (1-5) and six sub-scores (0-2) in both label versions, "
                       "label_versions.v1 (authors' review-repo file test_final_800_20260916.jsonl) and label_versions.v2 "
                       "(released data/test.jsonl, 2026-09-29), plus agent_integrity {v1, v2}. Dataset tables: "
                       "test_v2_released.jsonl, test_final_800_20260916_v1_review_repo.jsonl, annotations_flat_v1_v2.csv.",
}
MAPPING = """Proposed comparison with our metrics (solidity, contact-mediated interaction, object persistence, robot
deformation, object deformation):
  PhysicsLENS      Collision, Contact, Momentum (0/1 violation)        -> contact-mediated interaction
                   Deformation (0/1 violation)                        -> object deformation
  MiraBench        IC_0 action effect, IC_1 contact precedes motion,
                   IC_2 response direction (A-D, 1 -> 0)               -> contact-mediated interaction (together or separately)
                   IC_3 interpenetration (A-D)                         -> solidity (general: mixes robot-object and
                                                                          object-environment penetration)
                   SC_A1-SC_A4 colour, shape, size, material (A-D)     -> object deformation and object persistence
  RoboJudge-Bench  agent_consistency, agent integrity (0-2, v1 and v2) -> robot deformation
"""


def write_readmes(out, summary):
    layout = """  <dataset>/annotations/                       publisher annotation tables + one flat per-video table
  <dataset>/<generator>/prompts.csv             generator,dataset,number,id,prompt,task_description
  <dataset>/<generator>/NNN_<id>.mp4            unchanged upstream video (SHA-256 in MANIFEST.csv and json/)
  <dataset>/<generator>/json/NNN_<id>.json      provenance, prompt or task description, checksum, clip facts
  <dataset>/<generator>/annotations/NNN_<id>.json   the human annotation
  RoboJudge_Bench/<generator>/first_frame/NNN_<id>.png   conditioning first frame"""
    lines = [f"# Existing human annotations: MiraBench, PhysicsLENS, RoboJudge-Bench", "",
             f"Location: {summary['remote']}/ on ERIS. Built by "
             "scripts/stage_existing_annotations.py in the Robotvideo repository from pinned Hugging Face revisions; every "
             "file was checked against the Hub checksum, and MANIFEST.csv lists the SHA-256 of each video. "
             f"{summary['videos']} videos.", "", "Layout (per generator, as in worldgen/organized_by_generator):", "", layout, "",
             "Per dataset:", ""]
    for ds, n in summary["per_dataset"].items():
        gens = ", ".join(f"{k.split('/', 1)[1]} {v}" for k, v in summary["per_generator"].items() if k.startswith(ds + "/"))
        lines += [f"## {ds} ({n} videos)", "", f"Filter: {FILTERS[ds]}", "", f"Generators: {gens}", "",
                  f"Annotations: {ANNOTATIONS[ds]}", ""]
        (out / ds / "README.md").write_text("\n".join([f"# {ds} ({n} videos)", "", f"Filter: {FILTERS[ds]}", "",
                                                        f"Generators: {gens}", "", f"Annotations: {ANNOTATIONS[ds]}", "",
                                                        "Layout:", "", layout, ""]) + "\n")
    lines += ["## Metric mapping", "", MAPPING]
    if any(summary["notes"].values()):
        lines += ["Notes: " + "; ".join(f"{k}: {v}" for k, v in summary["notes"].items()), ""]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", choices=["MiraBench", "PhysicsLENS", "RoboJudge_Bench"])
    args = ap.parse_args()
    out = args.work / "existing_annotations"
    out.mkdir(parents=True, exist_ok=True)
    rows, notes = [], {}
    if args.only in (None, "MiraBench"):
        rows += mirabench(args.work, out, args.jobs)
    if args.only in (None, "PhysicsLENS"):
        placed, orphans, unannotated = physicslens(args.work, out, args.jobs)
        rows += placed
        notes["PhysicsLENS annotation rows without a video"] = orphans
        notes["PhysicsLENS generated videos without an annotation (not staged)"] = unannotated
    if args.only in (None, "RoboJudge_Bench"):
        rows += robojudge(args.work, out, args.jobs)
    write_csv(out / "MANIFEST.csv", rows)
    counts = Counter((r["dataset"], r["generator"]) for r in rows)
    summary = dict(remote=REMOTE, videos=len(rows), per_dataset=dict(Counter(r["dataset"] for r in rows)),
                   per_generator={f"{d}/{g}": n for (d, g), n in sorted(counts.items())}, notes=notes)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    write_readmes(out, summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
