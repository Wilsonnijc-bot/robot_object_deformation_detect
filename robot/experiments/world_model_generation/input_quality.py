"""Pixel-quality metrics for every world-model input first frame in a Robotvideo-style tree.

Usage: input_quality.py <robotvideo_root> <out_csv>

One input per video folder (same rule as build_manifest.py): TAVP uses generation input/first_substep/
when present, Humanoid_Everyday uses third_person/generation input/, NVIDIA_GR1_100 uses generation input/.
`generated` marks the first 15 per dataset in natural sort order (the 45 already generated).

Metrics (luminance, 0-255):
  lap_var_native   variance of the Laplacian at native size (sharpness; resolution-dependent)
  lap_var_model    same after the cover-resize + center crop the generators received (what the model saw)
  bandwidth99      radius (fraction of Nyquist) holding 99% of non-DC spectral energy; low = soft/upsampled
  noise_sigma      Immerkaer fast noise estimate
  blockiness       mean |step| on 8-px grid lines / elsewhere; ~1.0 none, >1.15 visible blocking
  mean_luma, clip_hi_pct (>=250), clip_lo_pct (<=5), rms_contrast, colorfulness (Hasler-Suesstrunk)
  border_pct       share of rows+cols at the edges that are flat black (letterbox/pillarbox)
"""
import csv
import os
import re
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

BUCKETS = [(16 / 9, (1280, 704)), (4 / 3, (960, 704)), (1.0, (832, 832)),
           (3 / 4, (704, 960)), (9 / 16, (704, 1280))]


def nat(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def inputs(root):
    out = []
    t = os.path.join(root, "TAVP")
    for n in sorted([x for x in os.listdir(t) if re.match(r"^\d{3}_", x)], key=nat):
        g = os.path.join(t, n, "generation input")
        b = os.path.join(g, "first_substep") if os.path.isdir(os.path.join(g, "first_substep")) else g
        out.append(("TAVP", n, os.path.join(b, "first_frame.png"), os.path.join(b, "prompt.txt")))
    t = os.path.join(root, "Humanoid_Everyday")
    for n in sorted([x for x in os.listdir(t) if "__" in x], key=nat):
        g = os.path.join(t, n, "third_person", "generation input")
        out.append(("Humanoid_Everyday", n, os.path.join(g, "first_frame.png"),
                    os.path.join(g, "actiongeneration_prompt.txt")))
    t = os.path.join(root, "NVIDIA_GR1_100")
    for n in sorted([x for x in os.listdir(t) if x.isdigit()], key=nat):
        g = os.path.join(t, n, "generation input")
        out.append(("NVIDIA_GR1_100", n, os.path.join(g, "first_frame.png"),
                    os.path.join(g, "actiongeneration_prompt.txt")))
    return out


def luma(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def bucket(w, h):
    return min(BUCKETS, key=lambda b: abs(b[0] - w / h))[1]


def model_view(im, tw, th):
    s = max(tw / im.width, th / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    left, top = (im.width - tw) // 2, (im.height - th) // 2
    return im.crop((left, top, left + tw, top + th))


def bandwidth99(y):
    f = np.abs(np.fft.fftshift(np.fft.fft2(y - y.mean()))) ** 2
    h, w = y.shape
    yy, xx = np.mgrid[-(h // 2):h - h // 2, -(w // 2):w - w // 2]
    r = np.sqrt((yy / (h / 2)) ** 2 + (xx / (w / 2)) ** 2)
    r, f = r.ravel(), f.ravel()
    keep = r > 0
    order = np.argsort(r[keep])
    c = np.cumsum(f[keep][order])
    return float(r[keep][order][np.searchsorted(c, 0.99 * c[-1])])


def noise_sigma(y):
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
    h, w = y.shape
    return float(np.sqrt(np.pi / 2) * np.abs(ndimage.convolve(y, k))[1:-1, 1:-1].sum() / (6 * (w - 2) * (h - 2)))


def blockiness(y):
    dx = np.abs(np.diff(y, axis=1)).mean(axis=0)
    dy = np.abs(np.diff(y, axis=0)).mean(axis=1)
    def ratio(d):
        idx = np.arange(d.size)
        on = d[(idx % 8) == 7]
        off = d[(idx % 8) != 7]
        return on.mean() / max(off.mean(), 1e-6)
    return float((ratio(dx) + ratio(dy)) / 2)


def border_pct(y):
    def run(lines):
        n = 0
        for line in lines:
            if line.mean() < 8 and line.std() < 3:
                n += 1
            else:
                break
        return n
    h, w = y.shape
    rows = run(y) + run(y[::-1])
    cols = run(y.T) + run(y.T[::-1])
    return 100.0 * (rows + cols) / (h + w)


def colorfulness(rgb):
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def main(root, out_csv):
    rows = []
    seen = {}
    for ds, name, frame, prompt in inputs(root):
        seen[ds] = seen.get(ds, 0) + 1
        im = Image.open(frame).convert("RGB")
        rgb = np.asarray(im, float)
        y = luma(rgb)
        tw, th = bucket(im.width, im.height)
        up = max(tw / im.width, th / im.height)
        ym = luma(np.asarray(model_view(im, tw, th), float))
        rows.append(dict(
            dataset=ds, name=name, index=seen[ds], generated=seen[ds] <= 15,
            width=im.width, height=im.height, megapixels=round(im.width * im.height / 1e6, 3),
            bucket=f"{tw}x{th}", upscale=round(up, 3),
            lap_var_native=round(float(ndimage.laplace(y).var()), 1),
            lap_var_model=round(float(ndimage.laplace(ym).var()), 1),
            bandwidth99=round(bandwidth99(y), 3), noise_sigma=round(noise_sigma(y), 2),
            blockiness=round(blockiness(y), 3), mean_luma=round(float(y.mean()), 1),
            clip_hi_pct=round(100 * float((y >= 250).mean()), 2), clip_lo_pct=round(100 * float((y <= 5).mean()), 2),
            rms_contrast=round(float(y.std() / 255), 3), colorfulness=round(colorfulness(rgb), 1),
            border_pct=round(border_pct(y), 2), prompt_chars=len(open(prompt).read().strip()),
            frame=os.path.relpath(frame, root)))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "inputs ->", out_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
