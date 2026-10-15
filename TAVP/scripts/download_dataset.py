#!/usr/bin/env python3
"""Build and fully validate the 23 published real-robot TAVP demonstration clips.

Python 3.9+ standard library only; requires ffmpeg and ffprobe on PATH.
All network bytes are verified against the pinned official Git tree.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import http.client
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path

COMMIT = "fcb05cd3430f238cfe1efd26b1662c309f2d79b8"
PAGE = "https://hcplab-sysu.github.io/TAVP/"
REPO = "https://github.com/HCPLab-SYSU/TAVP"
RAW = "https://raw.githubusercontent.com/HCPLab-SYSU/TAVP/"
PROVENANCES = {"published_verbatim", "transferred_from_matching_task", "inferred_from_task_category"}
PRINT_LOCK = threading.Lock()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def atomic_text(path, text):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def write_json(path, data):
    atomic_text(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash(path):
    h = hashlib.sha1(("blob %d\0" % path.stat().st_size).encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def identity(url):
    """Merge CSVs by full repository-relative path, never basename or row order."""
    parsed = urllib.parse.urlparse(url)
    path = urllib.parse.unquote(parsed.path)
    if "/static/videos/" not in path:
        raise ValueError("Not a video URL/path: " + url)
    return "static/videos/" + path.split("/static/videos/", 1)[1]


class Node:
    def __init__(self, tag="root", attrs=None, parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs or []), parent
        self.children = []

    def nodes(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.nodes()

    def text(self):
        return " ".join(" ".join(c.text() if isinstance(c, Node) else c
                                 for c in self.children).split())

    def has_class(self, name):
        return name in self.attrs.get("class", "").split()


class DOM(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.current = self.root
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.current)
        self.current.children.append(node)
        if tag not in self.VOID:
            self.current = node

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        node = self.current
        while node.parent:
            if node.tag == tag:
                self.current = node.parent
                return
            node = node.parent

    def handle_data(self, data):
        self.current.children.append(data)


def request_bytes(url, timeout, retries):
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TAVP-dataset-builder/1.0", "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (OSError, http.client.HTTPException, urllib.error.URLError) as exc:
            if attempt == retries:
                raise
            log("Retrying source request: %s (%s)" % (url, exc))
            time.sleep(min(2 ** attempt, 30))


def load_csvs(paths):
    indexed = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            # Inspect actual names. Unknown columns survive verbatim in source_csv_records.
            url_columns = [c for c in (reader.fieldnames or [])
                           if c.lower() in {"video_url", "original_video_url", "source_url", "mp4_url"}]
            if not url_columns:
                raise ValueError("CSV has no recognized video URL identity column: " + str(path))
            for line, row in enumerate(reader, 2):
                urls = [row[c] for c in url_columns if row.get(c)]
                if not urls:
                    raise ValueError("CSV row lacks video identity: %s:%d" % (path, line))
                key = identity(urls[0])
                indexed.setdefault(key, []).append({"file": path.name, "line": line, "columns": row})
    return indexed


def discover(root, csv_paths, timeout, retries):
    sources = root / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    tree_path = sources / "repository_tree.json"
    if not tree_path.exists():
        data = request_bytes("https://api.github.com/repos/HCPLab-SYSU/TAVP/git/trees/" + COMMIT + "?recursive=1", timeout, retries)
        atomic_text(tree_path, data.decode())
    tree = read_json(tree_path)
    if tree.get("truncated") or tree.get("sha") != COMMIT:
        raise ValueError("Repository tree is incomplete or does not match the pinned commit")
    entries = {item["path"]: item for item in tree["tree"]}
    index_path = sources / "repository_index.html"
    if not index_path.exists():
        index_path.write_bytes(request_bytes(RAW + COMMIT + "/index.html", timeout, retries))
    if git_hash(index_path) != entries["index.html"]["sha"]:
        raise ValueError("Pinned HTML snapshot failed official Git blob integrity check")
    dom = DOM(index_path.read_text(encoding="utf-8"))
    records = []
    for node in dom.root.nodes():
        if node.tag != "source" or not node.attrs.get("src", "").endswith(".mp4"):
            continue
        path = identity("/" + node.attrs["src"])
        if not path.startswith(("static/videos/Franka_Eval_Video/", "static/videos/Dobot_video264/")):
            continue
        card = node.parent
        while card and not (card.has_class("video-card") or card.has_class("media-item")):
            card = card.parent
        if card is None:
            raise ValueError("Cannot resolve published video card for " + path)
        descendants = list(card.nodes())
        captions = [x.text() for x in descendants if x.has_class("video-caption") or x.has_class("media-caption")]
        titles = [x.text() for x in descendants if x.has_class("video-title")]
        badges = [x.text().lower() for x in descendants if x.has_class("video-badge") or x.has_class("media-badge")]
        if len(captions) != 1 or len(badges) != 1 or badges[0] not in {"success", "failure"}:
            raise ValueError("Ambiguous caption/status for " + path)
        caption, status = captions[0], badges[0]
        robot = "Franka Research 3" if "Franka_Eval_Video" in path else "Dobot Nova 2"
        robust = "RobustnessTesting264/" in path
        scenario = caption.split(":", 1)[0] if robust else ("failure case" if status == "failure" else "standard")
        if titles:
            task = titles[0]
        elif robust:
            task = "Grape Picking"
        elif " Task:" in caption:
            task = caption.split(" Task:", 1)[0]
        else:
            task = "Collect Fruits" if "collect_fruits" in path else "Put Item in Drawer"
        instruction = caption.split("Language instruction:", 1)[1].strip().strip('"') if "Language instruction:" in caption else None
        entry = entries.get(path)
        if not entry or entry["type"] != "blob":
            raise ValueError("Video is absent from the pinned official Git tree: " + path)
        records.append({
            "dataset": "TAVP", "video_id": "%03d" % (len(records) + 1),
            "robot_type": robot, "task_name": slug(task), "task_display_name": task,
            "task_category": "robustness_test" if robust else ("failure_case" if status == "failure" else "standard_evaluation"),
            "scenario": scenario, "robustness_test_type": slug(scenario) if robust else None,
            "task_instruction": instruction, "original_task_instruction": instruction,
            "prompt_provenance": "published_verbatim" if instruction else None,
            "original_webpage_description": caption, "original_caption": caption,
            "execution_status": status, "environment": "real_world",
            "source_page": PAGE, "source_repository": REPO,
            "repository_commit": COMMIT, "original_repository_path": path,
            "original_video_url": RAW + COMMIT + "/" + urllib.parse.quote(path, safe="/"),
            "published_video_url": PAGE + urllib.parse.quote(path, safe="/"),
            "repository_blob_sha1": entry["sha"], "expected_file_size_bytes": entry["size"],
            "instruction_is_explicit": bool(instruction),
            "manual_verification_needed": instruction is None,
            "annotation_notes": None, "source_csv_records": [],
            "prompt_source": {"type": "official_project_page", "url": PAGE,
                              "repository_index_url": RAW + COMMIT + "/index.html"} if instruction else None,
            "quality_selection_notes": "Exact official MP4 linked on project page; no matching unconverted/higher-quality original exists in the pinned repository tree. No local re-encoding.",
        })
    if len(records) != 23 or Counter(r["robot_type"] for r in records) != {"Franka Research 3": 8, "Dobot Nova 2": 15}:
        raise ValueError("Official page discovery did not yield exactly 8 Franka + 15 Dobot videos")
    if len({r["original_repository_path"] for r in records}) != 23:
        raise ValueError("Duplicate video identities on official page")
    indexed = load_csvs(csv_paths)
    known_paths = {r["original_repository_path"] for r in records}
    extra = sorted(set(indexed) - known_paths)
    if extra:
        raise ValueError("CSV contains unmatched video identities: " + repr(extra))
    existing_path = sources / "source_catalog.json"
    existing = {r["original_repository_path"]: r for r in read_json(existing_path)} if existing_path.exists() else {}
    for r in records:
        matches = indexed.get(r["original_repository_path"], [])
        r["source_csv_records"] = matches
        notes = []
        candidate_prompts = []
        for match in matches:
            row = match["columns"]
            for column, official in [("robot", r["robot_type"]), ("status", r["execution_status"]), ("task", r["task_display_name"])]:
                if row.get(column) and row[column] != official:
                    raise ValueError("CSV conflicts with published %s for %s" % (column, r["video_id"]))
            if row.get("original_language_instruction") and row["original_language_instruction"] != r["original_task_instruction"]:
                raise ValueError("CSV author instruction differs from official page: " + r["video_id"])
            if row.get("annotation_notes"):
                notes.append(row["annotation_notes"])
            if row.get("manual_video_verification_needed", "").lower() in {"yes", "true", "1"}:
                r["manual_verification_needed"] = True
            if row.get("usable_task_prompt"):
                provenance = row.get("prompt_provenance")
                if provenance == "transferred_from_matching_success_task":
                    provenance = "transferred_from_matching_task"
                if provenance not in PROVENANCES:
                    raise ValueError("Unknown CSV prompt provenance: " + str(provenance))
                candidate_prompts.append((row["usable_task_prompt"], provenance, match))
        if len({(p, prov) for p, prov, _ in candidate_prompts}) > 1:
            raise ValueError("Conflicting CSV prompts by stable video identity: " + r["video_id"])
        if r["original_task_instruction"]:
            if candidate_prompts and candidate_prompts[0][:2] != (r["task_instruction"], "published_verbatim"):
                raise ValueError("CSV prompt conflicts with published instruction: " + r["video_id"])
        elif candidate_prompts:
            prompt, prov, match = candidate_prompts[0]
            if prov == "published_verbatim":
                raise ValueError("CSV incorrectly claims an unpublished instruction is verbatim")
            r.update(task_instruction=prompt, prompt_provenance=prov,
                     prompt_source={"type": "supplied_csv", "file": "sources/" + match["file"],
                                    "row_identity": match["columns"].get("id"),
                                    "video_identity": r["original_repository_path"]})
        elif r["original_repository_path"] in existing:
            old = existing[r["original_repository_path"]]
            for key in ["task_instruction", "prompt_provenance", "prompt_source", "source_csv_records", "annotation_notes"]:
                r[key] = old[key]
        elif r["robustness_test_type"] and r["robustness_test_type"] != "object_generalization":
            r.update(task_instruction="Put the grapes into the green plate.", prompt_provenance="transferred_from_matching_task",
                     prompt_source={"type": "matching_task", "video_id": "011", "url": PAGE})
            notes.append("Task-level transfer; no individual robustness-clip instruction was published.")
        else:
            fallback = {
                "collect_fruits": "Collect the specified fruits from the table and place them in the target container.",
                "put_item_in_drawer": "Open the specified drawer and place the designated item inside.",
                "grape_picking": "Pick up the designated target object and place it on the target plate.",
            }
            r.update(task_instruction=fallback[r["task_name"]], prompt_provenance="inferred_from_task_category",
                     prompt_source={"type": "inference", "basis": r["original_webpage_description"]})
            notes.append("Inferred generic action instruction; object identities and target are not author annotations.")
        if notes:
            r["annotation_notes"] = " ".join(dict.fromkeys(notes))
        if r["prompt_provenance"] == "transferred_from_matching_task":
            r["matching_task_instruction_video_id"] = "011"
            r["matching_task_instruction_source_url"] = PAGE
        else:
            r["matching_task_instruction_video_id"] = None
            r["matching_task_instruction_source_url"] = None
        robot_slug = "franka" if r["robot_type"].startswith("Franka") else "dobot"
        r["relative_folder_path"] = "_".join([r["video_id"], robot_slug, r["task_name"], slug(r["scenario"]), r["execution_status"]])
        # Three unseen-object clips share a caption but remain distinct by stable ID.
    if len({r["relative_folder_path"] for r in records}) != 23:
        raise ValueError("Duplicate output folders")
    return records


def original_integrity(path, record):
    return (path.is_file() and path.stat().st_size == record["expected_file_size_bytes"]
            and git_hash(path) == record["repository_blob_sha1"])


def download(path, record, timeout, retries):
    if original_integrity(path, record):
        return "skipped_verified"
    part = path.with_suffix(".mp4.part")
    expected = record["expected_file_size_bytes"]
    for attempt in range(retries + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset >= expected:
                if original_integrity(part, record):
                    part.replace(path)
                    return "resumed"
                part.unlink()
                offset = 0
            headers = {"User-Agent": "TAVP-dataset-builder/1.0", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = "bytes=%d-" % offset
            req = urllib.request.Request(record["original_video_url"], headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                append = response.status == 206 and offset > 0
                if response.status == 206:
                    value = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
                    if not match or int(match[1]) != offset or int(match[3]) != expected:
                        raise ValueError("Invalid HTTP range response: " + value)
                elif response.status != 200:
                    raise ValueError("Unexpected HTTP status " + str(response.status))
                # A server ignoring Range returns 200: restart safely rather than append.
                with part.open("ab" if append else "wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            if not original_integrity(part, record):
                if part.stat().st_size >= expected:
                    part.unlink()
                raise ValueError("Downloaded bytes failed pinned size/Git blob check")
            part.replace(path)
            return "resumed" if append else "downloaded"
        except (OSError, ValueError, http.client.HTTPException, urllib.error.URLError) as exc:
            log("[%s] download attempt %d failed: %s" % (record["video_id"], attempt + 1, exc))
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))


def run(command, timeout):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode or result.stderr.strip():
        raise ValueError("%s failed: %s" % (command[0], result.stderr.decode(errors="replace")[-3000:]))
    return result.stdout


def probe(path, timeout):
    data = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], timeout))
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    fmt = data["format"]
    # Keep metadata portable when the dataset is copied to another machine/directory.
    fmt["filename"] = path.name
    fps_rational = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    fps = float(Fraction(fps_rational)) if fps_rational and fps_rational != "0/0" else None
    return {
        "resolution": {"width": int(stream["width"]), "height": int(stream["height"])},
        "fps": fps, "fps_rational": fps_rational, "nominal_fps_rational": stream.get("r_frame_rate"),
        "duration_seconds": float(fmt["duration"]) if fmt.get("duration") else None,
        "codec": stream.get("codec_name"), "pixel_format": stream.get("pix_fmt"),
        "bitrate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        "video_stream_bitrate": int(stream["bit_rate"]) if stream.get("bit_rate") else None,
        "declared_frame_count": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None,
        "file_size_bytes": path.stat().st_size, "sha256": sha256(path),
        "ffprobe": data,
    }


def decode_all(path, timeout):
    output = run(["ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode", "-nostats", "-progress", "pipe:1",
                  "-noautorotate", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "null", "-"], timeout)
    frames = [int(value) for value in re.findall(rb"(?m)^frame=(\d+)", output)]
    if not frames or frames[-1] <= 0 or b"progress=end" not in output:
        raise ValueError("Full video decode did not finish with any frames")
    return frames[-1]


def first_pixels(path, timeout):
    return run(["ffmpeg", "-v", "error", "-xerror", "-noautorotate", "-i", str(path), "-map", "0:v:0",
                "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], timeout)


def check_frame(image, video, dimensions, timeout):
    info = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(image)], timeout))["streams"][0]
    if info["codec_name"] != "png" or (info["width"], info["height"]) != (dimensions["width"], dimensions["height"]):
        raise ValueError("First frame is not a PNG at original video resolution")
    pixels = first_pixels(image, timeout)
    expected = first_pixels(video, timeout)
    if len(pixels) != dimensions["width"] * dimensions["height"] * 3 or pixels != expected:
        raise ValueError("PNG pixels do not match decoded video frame index 0")
    return hashlib.sha256(pixels).hexdigest()


def extract_frame(video, image, timeout):
    temp = image.with_name("first_frame.tmp.png")
    run(["ffmpeg", "-v", "error", "-xerror", "-y", "-noautorotate", "-i", str(video), "-map", "0:v:0",
         "-frames:v", "1", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-c:v", "png", "-threads", "1", str(temp)], timeout)
    temp.replace(image)


def source_frame_path(root, record):
    generation = root / record['relative_folder_path'] / 'generation input'
    if record['robot_type'].startswith('Dobot'):
        return generation / 'source_first_frame.png'
    review_path = generation / 'metadata/action_review.json'
    if review_path.exists():
        review = read_json(review_path)
        if review['selected_segment']['end_frame_exclusive'] < review['source_frame_count']:
            return generation / 'first_substep' / 'first_frame.png'
    return generation / 'first_frame.png'


def process(root, record, args):
    folder = root / record["relative_folder_path"] / "original"
    if not args.verify_only:
        folder.mkdir(parents=True, exist_ok=True)
    video, frame = folder / "video.mp4", source_frame_path(root, record)
    if not args.verify_only:
        frame.parent.mkdir(parents=True, exist_ok=True)
    prompt, metadata_path = folder / "task_prompt.txt", folder / "metadata/metadata.json"
    if not args.verify_only:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        if not original_integrity(video, record):
            raise ValueError("MP4 missing or differs from pinned original")
        download_status = "skipped_verified"
    else:
        download_status = download(video, record, args.timeout, args.retries)
    technical = probe(video, args.decode_timeout)
    frame_count = decode_all(video, args.decode_timeout)
    if technical["declared_frame_count"] is not None and frame_count != technical["declared_frame_count"]:
        raise ValueError("Decoded frame count differs from MP4 declared count")
    if not args.verify_only:
        try:
            check_frame(frame, video, technical["resolution"], args.decode_timeout)
        except (OSError, ValueError, KeyError, StopIteration):
            extract_frame(video, frame, args.decode_timeout)
    pixel_hash = check_frame(frame, video, technical["resolution"], args.decode_timeout)
    metadata = dict(record, **technical)
    metadata.update(
        decoded_frame_count=frame_count, first_frame_index=0,
        first_frame_sha256=sha256(frame), first_frame_rgb24_sha256=pixel_hash,
        relative_video_path=record["relative_folder_path"] + "/original/video.mp4",
        relative_first_frame_path=frame.relative_to(root).as_posix(),
        relative_task_prompt_path=record["relative_folder_path"] + "/original/task_prompt.txt",
        relative_metadata_path=record["relative_folder_path"] + "/original/metadata/metadata.json",
        download_status=download_status, validation_status="passed", validation_errors=[],
        validation={"original_git_blob_verified": True, "full_video_decode": True,
                    "png_readable": True, "png_original_dimensions": True, "frame_zero_pixels_match": True},
    )
    if args.verify_only:
        saved = read_json(metadata_path)
        # Run-specific download status is allowed to differ; all source and content fields must agree.
        for key, value in metadata.items():
            if key != "download_status" and saved.get(key) != value:
                raise ValueError("Metadata mismatch: " + key)
    else:
        atomic_text(prompt, record["task_instruction"] + "\n")
        write_json(metadata_path, metadata)
    if not prompt.read_text(encoding="utf-8").strip() or prompt.read_text(encoding="utf-8").strip() != metadata["task_instruction"]:
        raise ValueError("Task prompt is empty or differs from metadata")
    saved = read_json(metadata_path)
    if saved["prompt_provenance"] not in PROVENANCES or not saved["original_video_url"].startswith(RAW + COMMIT + "/"):
        raise ValueError("Invalid annotation provenance or original source URL")
    log("[%s] %s; full decode %d frames; PNG frame 0 verified" % (record["video_id"], download_status, frame_count))
    return metadata


MANIFEST_FIELDS = ["video_id", "robot_type", "task_name", "task_instruction", "prompt_provenance", "environment",
                   "execution_status", "task_category", "scenario", "robustness_test_type", "relative_folder_path",
                   "relative_video_path", "relative_first_frame_path", "relative_task_prompt_path", "relative_metadata_path",
                   "original_video_url", "published_video_url", "original_repository_path", "repository_commit",
                   "width", "height", "fps", "fps_rational", "duration_seconds", "codec", "bitrate", "file_size_bytes", "sha256",
                   "manual_verification_needed", "download_status", "validation_status", "validation_errors"]


def manifest_row(metadata):
    row = {key: metadata.get(key) for key in MANIFEST_FIELDS}
    row.update(width=metadata.get("resolution", {}).get("width"), height=metadata.get("resolution", {}).get("height"))
    row["manual_verification_needed"] = str(bool(row["manual_verification_needed"])).lower()
    row["validation_errors"] = json.dumps(row["validation_errors"] or [], ensure_ascii=False)
    return row


def write_manifest(root, results):
    temp = root / "manifest.csv.tmp"
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_row(r) for r in results)
    temp.replace(root / "manifest.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--csv", type=Path, action="append", help="CSV to merge by video URL; repeat for multiple CSVs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60, help="Network socket timeout in seconds")
    parser.add_argument("--decode-timeout", type=float, default=300)
    parser.add_argument("--retries", type=int, default=4, help="Retries after the first download attempt")
    parser.add_argument("--verify-only", action="store_true", help="Validate all existing artifacts without repairing them")
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 0 or args.timeout <= 0 or args.decode_timeout <= 0:
        parser.error("workers/timeouts must be positive and retries nonnegative")
    for program in ["ffmpeg", "ffprobe"]:
        if not shutil.which(program):
            parser.error(program + " is required on PATH")
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "sources").mkdir(exist_ok=True)
    bundled_sources = Path(__file__).resolve().parent.parent / "sources"
    if root != Path(__file__).resolve().parent.parent and not args.csv:
        for name in ["repository_tree.json", "repository_index.html", "source_catalog.json"]:
            source = bundled_sources / name
            target = root / "sources" / name
            if source.exists() and not target.exists():
                shutil.copy2(source, target)
        for source in sorted(bundled_sources.glob("*.csv")):
            target = root / "sources" / source.name
            if not target.exists():
                shutil.copy2(source, target)
    csv_paths = args.csv or sorted((root / "sources").glob("*.csv"))
    if args.csv:
        for path in args.csv:
            destination = root / "sources" / path.name
            if path.resolve() != destination.resolve():
                if destination.exists() and sha256(destination) != sha256(path):
                    raise ValueError("Different input CSVs have the same filename: " + path.name)
                shutil.copy2(path, destination)
        csv_paths = [root / "sources" / path.name for path in args.csv]
    records = discover(root, csv_paths, args.timeout, args.retries)
    catalog_path = root / "sources/source_catalog.json"
    if args.verify_only:
        if read_json(catalog_path) != records:
            raise ValueError("Saved source catalog differs from rediscovery")
    else:
        write_json(catalog_path, records)
    log("Discovered exactly 23 published real-world clips (8 Franka, 15 Dobot), pinned to " + COMMIT)
    results, failures = {}, []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, root, record, args): record for record in records}
        for future in concurrent.futures.as_completed(futures):
            record = futures[future]
            try:
                results[record["video_id"]] = future.result()
            except Exception as exc:
                log("[%s] FAILED: %s" % (record["video_id"], exc))
                failure = dict(record, download_status="failed_or_unverified", validation_status="failed", validation_errors=[str(exc)])
                for key, filename in [("relative_video_path", "video.mp4"), ("relative_first_frame_path", "first_frame.png"),
                                      ("relative_task_prompt_path", "task_prompt.txt"), ("relative_metadata_path", "metadata.json")]:
                    failure[key] = record["relative_folder_path"] + "/original/" + ("metadata/" if key == "relative_metadata_path" else "") + filename
                failure['relative_first_frame_path'] = source_frame_path(root, record).relative_to(root).as_posix()
                results[record["video_id"]] = failure
                failures.append({"video_id": record["video_id"], "folder": record["relative_folder_path"], "error": str(exc)})
    ordered = [results[r["video_id"]] for r in records]
    if args.verify_only:
        with (root / "manifest.csv").open(encoding="utf-8", newline="") as f:
            saved_rows = list(csv.DictReader(f))
        expected_rows = [{k: "" if v is None else str(v) for k, v in manifest_row(r).items()} for r in ordered]
        for row in expected_rows:
            saved = next((s for s in saved_rows if s["video_id"] == row["video_id"]), None)
            if saved is None or any(saved.get(k) != v for k, v in row.items() if k != "download_status"):
                failures.append({"video_id": row["video_id"], "error": "Manifest differs from validated artifact metadata"})
        if len(saved_rows) != 23 or len({r["video_id"] for r in saved_rows}) != 23:
            failures.append({"error": "Manifest does not contain exactly 23 unique entries"})
    else:
        write_manifest(root, ordered)
    folders = sorted(p.name for p in root.iterdir() if p.is_dir() and re.match(r"^\d{3}_", p.name))
    expected_folders = sorted(r["relative_folder_path"] for r in records)
    if folders != expected_folders:
        failures.append({"error": "Video subfolder set differs from expected 23", "actual": folders})
    valid = [r for r in ordered if r["validation_status"] == "passed"]
    uncertain = [{"video_id": r["video_id"], "prompt_provenance": r["prompt_provenance"],
                  "task_instruction": r["task_instruction"], "annotation_notes": r["annotation_notes"]}
                 for r in ordered if r["manual_verification_needed"]]
    summary = {
        "complete": not failures and len(valid) == 23,
        "expected_videos": 23, "validated_videos": len(valid), "video_folder_count": len(folders),
        "total_video_size_bytes": sum(r["file_size_bytes"] for r in valid),
        "robot_counts": dict(Counter(r["robot_type"] for r in valid)),
        "execution_status_counts": dict(Counter(r["execution_status"] for r in valid)),
        "prompt_provenance_counts": dict(Counter(r["prompt_provenance"] for r in valid)),
        "resolution_counts": dict(Counter("%dx%d" % (r["resolution"]["width"], r["resolution"]["height"]) for r in valid)),
        "fps_counts": dict(Counter(r["fps_rational"] for r in valid)),
        "resolution_fps_counts": dict(Counter("%dx%d @ %s fps" % (r["resolution"]["width"], r["resolution"]["height"], r["fps_rational"]) for r in valid)),
        "codec_counts": dict(Counter(r["codec"] for r in valid)),
        "manual_verification_needed_count": len(uncertain), "uncertain_annotations": uncertain,
        "failures": failures, "repository_commit": COMMIT,
        "source_csv_files": [{"file": "sources/" + p.name, "sha256": sha256(p)} for p in csv_paths],
        "validation_time_utc": datetime.now(timezone.utc).isoformat(),
        "validation_mode": "verify_only" if args.verify_only else "build_and_validate",
        "tools": {p: subprocess.check_output([p, "-version"], text=True).splitlines()[0] for p in ["ffmpeg", "ffprobe"]},
    }
    write_json(root / "validation_report.json", summary)
    log(json.dumps({k: v for k, v in summary.items() if k not in {"uncertain_annotations", "tools", "source_csv_files"}}, indent=2))
    if uncertain:
        log("Annotation review remains flagged for videos: " + ", ".join(r["video_id"] for r in uncertain))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        log("FATAL: " + str(error))
        sys.exit(1)
