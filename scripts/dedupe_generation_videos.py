"""Remove generation-input videos that are byte-identical to a source video in original/.

Usage: .venv/bin/python dedupe_generation_videos.py [--dry-run]

For every `generation input/` (and `generation input/first_substep/`) whose video.mp4 has the same
SHA-256 as an MP4 under the sibling original/ folder, the copy is deleted and every reference is
pointed at the original file:
  * metadata/metadata.json `relative_video_path` (same convention as before: metadata-directory-
    relative when it started with "../", otherwise dataset-root-relative, as in TAVP), plus
    `generation_video_is_source: true`;
  * metadata/action_review.json `generation_video` (review-directory-relative);
  * every cell of the dataset's manifests that resolved to the deleted file.
Derived generation videos (shortened, re-encoded or cropped) are left alone.
"""
import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = {
    "TAVP": ["TAVP/world_model_manifest.csv", "TAVP/manifest.csv"],
    "Humanoid_Everyday": ["Humanoid_Everyday/metamanifest/world_model_prompts.csv"],
    "NVIDIA_GR1_100": ["NVIDIA_GR1_100/metamanifest/world_model_prompts.csv"],
    "ActionNet": ["ActionNet/metamanifest/world_model_prompts.csv"],
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(target, start):
    return os.path.relpath(target, start).replace(os.sep, "/")


def rewrite_json(path, update):
    text = path.read_text(encoding="utf-8")
    indent = re.search(r"\n( +)\"", text)
    data = json.loads(text)
    if not update(data):
        return
    out = json.dumps(data, indent=len(indent.group(1)) if indent else 2, ensure_ascii=False)
    path.write_text(out + ("\n" if text.endswith("\n") else ""), encoding="utf-8")


def find_duplicates():
    pairs = []
    for gen in sorted(ROOT.rglob("generation input")):
        if not gen.is_dir() or {"site", ".git"} & set(gen.relative_to(ROOT).parts):
            continue
        sources = {sha(p): p for p in (gen.parent / "original").rglob("*.mp4")}
        for sel in (gen, gen / "first_substep"):
            video = sel / "video.mp4"
            if video.exists() and (h := sha(video)) in sources:
                pairs.append((video, sources[h]))
    return pairs


def update_metadata(video, source):
    dataset_root = ROOT / video.relative_to(ROOT).parts[0]
    meta = video.parent / "metadata" / "metadata.json"
    if meta.exists():
        def fix(d):
            old = d.get("relative_video_path", "")
            d["relative_video_path"] = (rel(source, meta.parent) if not old or old.startswith("../")
                                        else rel(source, dataset_root))
            d["generation_video_is_source"] = True
            return True
        rewrite_json(meta, fix)
    for review in (video.parent / "metadata" / "action_review.json",):
        if review.exists():
            def fix_review(d):
                if d.get("generation_video") and (review.parent / d["generation_video"]).resolve() == video.resolve():
                    d["generation_video"] = rel(source, review.parent)
                    return True
                return False
            rewrite_json(review, fix_review)


def update_manifest(path, mapping):
    with path.open(encoding="utf-8", newline="") as fh:
        text = fh.read()
    eol = "\r\n" if "\r\n" in text else "\n"
    rows = list(csv.reader(io.StringIO(text, newline="")))
    bases = [path.parent, ROOT / path.relative_to(ROOT).parts[0]]
    changed = 0
    for row in rows[1:]:
        for i, cell in enumerate(row):
            if not cell.endswith("video.mp4"):
                continue
            for base in bases:
                target = (base / cell).resolve()
                if target in mapping:
                    row[i] = rel(mapping[target], base)
                    changed += 1
                    break
    if changed:
        buf = io.StringIO()
        csv.writer(buf, lineterminator=eol).writerows(rows)
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(buf.getvalue())
    return changed


def main(dry_run):
    pairs = find_duplicates()
    total = sum(v.stat().st_size for v, _ in pairs)
    print(f"{len(pairs)} generation videos identical to their source ({total / 1e6:.0f} MB)")
    if dry_run:
        return
    mapping = {v.resolve(): s for v, s in pairs}
    for video, source in pairs:
        update_metadata(video, source)
    for files in MANIFESTS.values():
        for f in files:
            if (ROOT / f).exists():
                print(f, update_manifest(ROOT / f, mapping), "cells repointed")
    for video, _ in pairs:
        video.unlink()


if __name__ == "__main__":
    main("--dry-run" in sys.argv[1:])
