"""Keep only the right RGB camera in every RoboMIND Franka episode.

Usage: ../.venv/bin/python strip_franka_cameras.py

For each franka_3rgb__*/<episode>/original/:
  * data/trajectory.hdf5 is rewritten without observations/rgb_images/camera_{left,top}
    and without observations/depth_images/*. Every other dataset and attribute is copied
    unchanged; the camera_right JPEG bytes are verified identical to the source.
  * decoded_videos/camera_{left,top}.mp4 are deleted.
  * metadata/metadata.json records the original HDF5 hash and the derivation.
Already-stripped episodes are skipped.
"""
import glob
import hashlib
import json
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROP = ("observations/rgb_images/camera_left", "observations/rgb_images/camera_top",
        "observations/depth_images")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def keep(name):
    return not any(name == d or name.startswith(d + "/") for d in DROP)


def copy(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    for name, obj in src.items():
        if not keep(obj.name.lstrip("/")):
            continue
        if isinstance(obj, h5py.Group):
            copy(obj, dst.create_group(name))
        else:
            src.copy(obj, dst, name=name)


def strip(original):
    meta_path = os.path.join(original, "metadata", "metadata.json")
    meta = json.load(open(meta_path))
    if meta["hdf5"].get("derivation"):
        print("skip", original)
        return
    h5 = os.path.join(original, "data", "trajectory.hdf5")
    tmp = h5 + ".tmp"
    with h5py.File(h5, "r") as src, h5py.File(tmp, "w") as dst:
        copy(src, dst)
        a = src["observations/rgb_images/camera_right"]
        b = dst["observations/rgb_images/camera_right"]
        assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
        kept = []
        dst.visit(lambda n: kept.append(n) if isinstance(dst[n], h5py.Dataset) else None)
    hdf5 = meta["hdf5"]
    hdf5["source_size_bytes"], hdf5["source_sha256"] = hdf5.pop("size_bytes"), hdf5.pop("sha256")
    hdf5["removed_datasets"] = [d for d in hdf5["datasets"] if not keep(d)]
    hdf5["datasets"] = {k: v for k, v in hdf5["datasets"].items() if keep(k)}
    assert sorted(hdf5["datasets"]) == sorted(kept)
    os.replace(tmp, h5)
    hdf5["size_bytes"], hdf5["sha256"] = os.path.getsize(h5), sha256(h5)
    hdf5["derivation"] = ("Rewritten from the source member keeping only the right RGB camera: "
                          "camera_left/camera_top RGB and all depth images removed; remaining "
                          "datasets and attributes copied unchanged (camera_right JPEG bytes verified).")
    for cam in ("camera_left", "camera_top"):
        p = os.path.join(original, "decoded_videos", cam + ".mp4")
        if os.path.exists(p):
            os.remove(p)
        meta["decoded_preview_videos"].pop(cam, None)
    meta["source_bytes_preserved"] = False
    meta["camera_policy"] = "Only the right RGB camera (camera_right, the generation camera) is retained."
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"{original}: {hdf5['source_size_bytes']:,} -> {hdf5['size_bytes']:,} bytes")


if __name__ == "__main__":
    for d in sorted(glob.glob(os.path.join(HERE, "franka_3rgb__*", "*", "original"))):
        strip(d)
