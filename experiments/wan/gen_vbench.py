#!/usr/bin/env python3
"""
Generate the VBench prompt suite with Wan2.2-TI2V-5B, dense or INTRA (+ the distilled adapter).

VBench standard mode matches files by name: for every prompt it looks for exactly
'<prompt_en>-<i>.mp4', i = 0..4, verbatim (spaces kept; human_action even parses the label back out
of the file name). It takes the file extension from the first entry of os.listdir(videos_dir), so
the output directory must contain nothing but .mp4 files; logs and the manifest go to --meta_dir.

The seed depends only on (prompt index, sample index), so the dense and INTRA arms start from
identical noise for every clip. Clips are written to a temporary file and renamed into place, and
existing clips are skipped, so runs can be killed and resumed.

INTRA here uses the reference attention (intra_wan.sparse_self_attention), as for the paper's
VBench numbers; latency is measured separately with bench_speed.py.
"""
import argparse, json, os, time

import torch

from intra_wan import (add_pattern_args, config_from_args, install_intra_processors, set_intra_grid,
                       grid_from_pixels, add_dora_adapter)


def load_prompts(path):
    """Unique prompt_en in first-appearance order (VBench_full_info.json has 946 entries but 944
    distinct prompts; a repeated prompt's 5 files serve both entries)."""
    seen, out = set(), []
    for e in json.load(open(path)):
        if e["prompt_en"] not in seen:
            seen.add(e["prompt_en"])
            out.append(e["prompt_en"])
    return out


def build_pipe(a, dev):
    from diffusers import WanPipeline
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor
    pipe = WanPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    tf = pipe.transformer
    for b in tf.blocks:
        b.attn1.set_processor(WanAttnProcessor())
    if a.mode == "dense":
        return pipe

    cfg = config_from_args(a)
    install_intra_processors(tf, cfg)                         # blocks 0 and last stay dense
    if a.lora:
        add_dora_adapter(tf, a.lora_rank, bool(a.ffn_lora))
        sd = torch.load(a.lora, map_location="cpu")
        _, unexpected = tf.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f"[gen] adapter r={a.lora_rank}: {loaded}/{len(sd)} tensors matched", flush=True)
        if loaded == 0:   # strict=False would otherwise silently keep a random adapter
            raise RuntimeError(f"{a.lora} matched no parameters (wrong rank or targets?)")
    grid = grid_from_pixels(a.frames, a.height, a.width)
    set_intra_grid(tf, grid)
    print(f"[gen] INTRA {cfg} grid {grid}", flush=True)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "intra"], required=True)
    ap.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers", help="HF id or local dir")
    ap.add_argument("--prompts", required=True, help="VBench_full_info.json from the VBench repo")
    ap.add_argument("--out", required=True, help="mp4 directory (will contain only .mp4 files)")
    ap.add_argument("--meta_dir", default="", help="manifest / temp dir, default <out>_meta")
    ap.add_argument("--lora", default="", help="adapter from train.py (intra mode)")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--ffn_lora", type=int, default=1)
    add_pattern_args(ap)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--samples", type=int, default=5, help="VBench standard mode expects 5")
    ap.add_argument("--seed_base", type=int, default=20260728)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts only")
    a = ap.parse_args()

    dev = "cuda"
    prompts = load_prompts(a.prompts)
    if a.limit:
        prompts = prompts[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    meta_dir = a.meta_dir or (a.out.rstrip("/") + "_meta")
    os.makedirs(meta_dir, exist_ok=True)

    jobs = [(pi, si) for pi in range(len(prompts)) for si in range(a.samples)]
    jobs = [j for n, j in enumerate(jobs) if n % a.nshard == a.shard]
    todo = []
    for pi, si in jobs:
        fp = os.path.join(a.out, f"{prompts[pi]}-{si}.mp4")
        if not (os.path.exists(fp) and os.path.getsize(fp) > 0):
            todo.append((pi, si, fp))
    print(f"[gen] shard {a.shard}/{a.nshard}: {len(jobs)} clips assigned, {len(todo)} to generate", flush=True)
    if not todo:
        return

    pipe = build_pipe(a, dev)
    zd = pipe.vae.config.z_dim
    vmean = torch.tensor(pipe.vae.config.latents_mean, device=dev).view(1, zd, 1, 1, 1)
    vstd = torch.tensor(pipe.vae.config.latents_std, device=dev).view(1, zd, 1, 1, 1)

    from diffusers.utils import export_to_video
    from PIL import Image

    t0 = time.time()
    manifest = open(os.path.join(meta_dir, f"manifest_shard{a.shard}.jsonl"), "a")
    for n, (pi, si, fp) in enumerate(todo):
        seed = a.seed_base + pi * a.samples + si
        g = torch.Generator(device=dev).manual_seed(seed)
        with torch.no_grad():
            out = pipe(prompt=prompts[pi], height=a.height, width=a.width, num_frames=a.frames,
                       num_inference_steps=a.steps, guidance_scale=a.guidance,
                       output_type="latent", generator=g)
            lat = (out.frames if hasattr(out, "frames") else out[0]).float()
            z = (lat.to(dev, torch.float32) * vstd + vmean).to(pipe.vae.dtype)
            vid = pipe.vae.decode(z, return_dict=False)[0]           # (1, 3, T, H, W) in [-1, 1]
        v = ((vid[0].permute(1, 0, 2, 3).float() + 1.0) / 2.0).clamp(0, 1)
        frames = [Image.fromarray((f.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")) for f in v]
        tmp = os.path.join(meta_dir, f".tmp_shard{a.shard}.mp4")
        export_to_video(frames, tmp, fps=a.fps)
        os.replace(tmp, fp)
        manifest.write(json.dumps({"prompt": prompts[pi], "pi": pi, "si": si, "seed": seed,
                                   "file": os.path.basename(fp)}) + "\n")
        manifest.flush()
        if (n + 1) % 10 == 0 or n == 0:
            el = time.time() - t0
            print(f"[gen] {n + 1}/{len(todo)}  {el / (n + 1):.1f}s/clip  "
                  f"eta {(len(todo) - n - 1) * el / (n + 1) / 60:.0f}m", flush=True)
    manifest.close()


if __name__ == "__main__":
    main()
