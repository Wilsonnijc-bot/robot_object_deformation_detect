"""Center-crop/resize each first frame to a shared aspect bucket (multiples of 32).

All three models receive the same conditioning image. Reads manifest.json written by
build_manifest.py and writes manifest_prepped.json next to it.
"""
import json
import os
import sys

from PIL import Image

# (aspect ratio, (width, height)); every side is a multiple of 32 for Wan-style VAEs.
BUCKETS = [(16 / 9, (1280, 704)), (4 / 3, (960, 704)), (1.0, (832, 832)),
           (3 / 4, (704, 960)), (9 / 16, (704, 1280))]


def bucket_for(w, h):
    ar = w / h
    return min(BUCKETS, key=lambda b: abs(b[0] - ar))[1]


def main(root):
    items = json.load(open(os.path.join(root, "manifest.json")))
    for it in items:
        tw, th = bucket_for(it["width"], it["height"])
        im = Image.open(it["first_frame"]).convert("RGB")
        scale = max(tw / im.width, th / im.height)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        left, top = (im.width - tw) // 2, (im.height - th) // 2
        im = im.crop((left, top, left + tw, top + th))
        out = os.path.join(os.path.dirname(it["first_frame"]), "cond.png")
        im.save(out)
        it.update(cond_image=out, gen_width=tw, gen_height=th)
    json.dump(items, open(os.path.join(root, "manifest_prepped.json"), "w"), indent=1)
    print(len(items), "prepped")


if __name__ == "__main__":
    main(sys.argv[1])
