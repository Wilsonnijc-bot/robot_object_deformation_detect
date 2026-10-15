"""One GPU worker: load a model once, generate its strided share of the manifest.

Usage: worker.py --model {cosmos25,cosmos3} --manifest M --out DIR --worker I --num-workers N
Skips items whose output json already exists, so relaunching never repeats finished work.
"""
import argparse
import json
import os
import sys
import time

import torch

NEG = ("The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
       "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
       "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky "
       "movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
       "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
       "Overall, the video is of poor quality.")
MODELS = "/scratch/z/zy992/models"

# Settings recorded verbatim into every output json.
CFG = {
    "cosmos25": dict(frames=93, steps=35, guidance=7.0, fps=16),
    "cosmos3": dict(frames=93, steps=35, guidance=6.0, fps=24),
}


class NoSafetyChecker(torch.nn.Module):
    """Stand-in for CosmosSafetyChecker: the guardrail models are not installed on ERIS."""

    dtype = torch.bfloat16
    device = torch.device("cuda")

    def check_text_safety(self, prompt):
        return True


def load_cosmos25():
    from diffusers import Cosmos2_5_PredictBasePipeline
    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        f"{MODELS}/Cosmos-Predict2.5-2B-diffusers", torch_dtype=torch.bfloat16, safety_checker=NoSafetyChecker())
    pipe = pipe.to("cuda")

    def run(it, c, gen):
        from PIL import Image
        out = pipe(image=Image.open(it["cond_image"]), prompt=it["prompt"], negative_prompt=NEG,
                   height=it["gen_height"], width=it["gen_width"], num_frames=c["frames"],
                   num_inference_steps=c["steps"], guidance_scale=c["guidance"], generator=gen)
        return out.frames[0]
    return run


def load_cosmos3():
    from diffusers import Cosmos3OmniPipeline, UniPCMultistepScheduler
    pipe = Cosmos3OmniPipeline.from_pretrained(f"{MODELS}/Cosmos3-Nano", torch_dtype=torch.bfloat16,
                                               device_map="cuda", enable_safety_checker=False)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)

    def run(it, c, gen):
        from PIL import Image
        out = pipe(prompt=it["prompt"], negative_prompt=NEG, image=Image.open(it["cond_image"]),
                   num_frames=c["frames"], height=it["gen_height"], width=it["gen_width"], fps=float(c["fps"]),
                   num_inference_steps=c["steps"], guidance_scale=c["guidance"], generator=gen,
                   enable_safety_check=False)
        return out.video
    return run


LOADERS = {"cosmos25": load_cosmos25, "cosmos3": load_cosmos3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LOADERS))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    items = json.load(open(a.manifest))[: a.limit]
    mine = items[a.worker::a.num_workers]
    os.makedirs(a.out, exist_ok=True)
    todo = [i for i in mine if not os.path.exists(os.path.join(a.out, i["id"] + ".json"))]
    print(f"[{a.model} w{a.worker}] {len(todo)}/{len(mine)} to do", flush=True)
    if not todo:
        return
    from diffusers.utils import export_to_video
    t0 = time.time()
    run = LOADERS[a.model]()
    load_s = time.time() - t0
    c = CFG[a.model]
    for it in todo:
        torch.cuda.reset_peak_memory_stats()
        gen = torch.Generator(device="cuda").manual_seed(a.seed)
        t1 = time.time()
        frames = run(it, c, gen)
        gen_s = time.time() - t1
        mp4 = os.path.join(a.out, it["id"] + ".mp4")
        export_to_video(frames, mp4, fps=c["fps"])
        meta = dict(id=it["id"], model=a.model, seed=a.seed, **c, width=it["gen_width"], height=it["gen_height"],
                    source_frame=it["source_frame"], source_prompt=it["source_prompt"], prompt=it["prompt"],
                    gen_seconds=round(gen_s, 1), load_seconds=round(load_s, 1), worker=a.worker,
                    num_workers=a.num_workers, peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        json.dump(meta, open(os.path.join(a.out, it["id"] + ".json"), "w"), indent=1)  # written last = completion marker
        print(f"[{a.model} w{a.worker}] {it['id']} {gen_s:.0f}s peak {meta['peak_mem_gib']} GiB", flush=True)


if __name__ == "__main__":
    main()
