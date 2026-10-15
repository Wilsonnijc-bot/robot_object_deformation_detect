"""ABot-PhysWorld (Wan2.1-I2V-14B-480P fine-tune) worker with an explicit weight-load check.

Upstream inference.py loads the checkpoint with strict=False and never checks the result, so a key
mismatch would silently fall back to vanilla Wan2.1. Here every checkpoint tensor must map onto the DiT
with the right shape, and we record how many tensors actually differ from the base weights.

Usage (run from abot-physworld/inference so the vendored diffsynth imports):
  abot_worker.py --check-only --out DIR
  abot_worker.py --manifest M --out DIR [--worker I --num-workers N --limit K]
"""
import argparse
import glob
import json
import os
import time

import torch
from PIL import Image

from diffsynth import save_video
from diffsynth.models.utils import load_state_dict
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

BASE = "/scratch/z/zy992/models/Wan2.1-I2V-14B-480P"
CKPT = "/scratch/z/zy992/models/ABot-PhysWorld/abotpw_i2v_480p.safetensors"
# Upstream DEFAULT_NEGATIVE_PROMPT, verbatim.
NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
       "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
       "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
       "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
# Upstream defaults: 81 frames at 15 fps, 50 steps, cfg 5, tiled VAE.
CFG = dict(frames=81, steps=50, guidance=5.0, fps=15)
# Shared aspect bucket -> ~480p size (multiples of 16) for the 480P model.
SIZE = {(1280, 704): (864, 480), (960, 704): (656, 480), (704, 1280): (480, 864)}


def load():
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda", model_configs=[
        ModelConfig(path=sorted(glob.glob(f"{BASE}/diffusion_pytorch_model-*.safetensors"))),
        ModelConfig(path=f"{BASE}/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path=f"{BASE}/Wan2.1_VAE.pth"),
        ModelConfig(path=f"{BASE}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ], tokenizer_config=ModelConfig(path=f"{BASE}/google/umt5-xxl"))  # default fetches from ModelScope
    sd = load_state_dict(CKPT)
    dit = pipe.dit.state_dict()
    missing = sorted(set(dit) - set(sd))
    unexpected = sorted(set(sd) - set(dit))
    shape_bad = [k for k in sd if k in dit and tuple(sd[k].shape) != tuple(dit[k].shape)]
    changed, total, diffs, unchanged = 0, 0, [], []
    for k, v in sd.items():
        if k in dit and k not in shape_bad:
            base = dit[k]
            d = (v.to(base.device, base.dtype).float() - base.float()).abs().mean().item()
            total += 1
            if d > 0:
                changed += 1
            else:
                unchanged.append(k)
            diffs.append(d)
    # RMSNorm gains sit near 1.0 where bf16 resolution (~0.008) swallows small updates, so they may
    # legitimately equal the base; every other tensor must have been fine-tuned.
    unchanged_non_norm = [k for k in unchanged if not (".norm" in k and k.endswith(".weight"))]
    res = pipe.dit.load_state_dict(sd, strict=False)
    report = dict(checkpoint=CKPT, ckpt_tensors=len(sd), dit_tensors=len(dit), missing=len(missing),
                  unexpected=len(unexpected), shape_mismatch=len(shape_bad), tensors_differing_from_base=changed,
                  tensors_compared=total, mean_abs_diff=sum(diffs) / max(len(diffs), 1),
                  unchanged_non_norm=len(unchanged_non_norm), unchanged_non_norm_examples=unchanged_non_norm[:5],
                  missing_examples=missing[:5], unexpected_examples=unexpected[:5],
                  load_missing=len(res.missing_keys), load_unexpected=len(res.unexpected_keys))
    del sd
    torch.cuda.empty_cache()
    report["weights_ok"] = (report["missing"] == 0 and report["unexpected"] == 0 and report["shape_mismatch"] == 0
                            and total == report["ckpt_tensors"] and not unchanged_non_norm)
    return pipe, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    todo = []
    if a.manifest:
        items = json.load(open(a.manifest))[: a.limit]
        todo = [i for i in items[a.worker::a.num_workers] if not os.path.exists(os.path.join(a.out, i["id"] + ".json"))]
        print(f"[abot w{a.worker}] {len(todo)} to do", flush=True)
        if not todo and not a.check_only:
            return
    t0 = time.time()
    pipe, report = load()
    load_s = time.time() - t0
    json.dump(report, open(os.path.join(a.out, f"weight_check_w{a.worker}.json"), "w"), indent=1)
    print("[abot] weight check:", json.dumps(report), flush=True)
    if not report["weights_ok"]:
        raise SystemExit("ABot checkpoint did not load cleanly onto the Wan2.1 DiT; refusing to generate.")
    if a.check_only:
        return
    for it in todo:
        w, h = SIZE[(it["gen_width"], it["gen_height"])]
        img = Image.open(it["cond_image"]).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        video = pipe(prompt=it["prompt"], negative_prompt=NEG, input_image=img, height=h, width=w,
                     num_frames=CFG["frames"], num_inference_steps=CFG["steps"], cfg_scale=CFG["guidance"],
                     seed=a.seed, tiled=True)
        gen_s = time.time() - t1
        save_video(video, os.path.join(a.out, it["id"] + ".mp4"), fps=CFG["fps"], quality=5)
        meta = dict(id=it["id"], model="abot_physworld", seed=a.seed, **CFG, width=w, height=h,
                    source_frame=it["source_frame"], source_prompt=it["source_prompt"], prompt=it["prompt"],
                    gen_seconds=round(gen_s, 1), load_seconds=round(load_s, 1), worker=a.worker,
                    num_workers=a.num_workers, peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        json.dump(meta, open(os.path.join(a.out, it["id"] + ".json"), "w"), indent=1)
        print(f"[abot w{a.worker}] {it['id']} {gen_s:.0f}s peak {meta['peak_mem_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
