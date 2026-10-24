#!/usr/bin/env python3
"""
Teacher data for self-distillation: dense Wan2.2-TI2V-5B generations on templated motion prompts.

Writes into --out:
  prompts.json        the prompt list (600 by default; disjoint from VBench by construction)
  uncond_embeds.pt    UMT5 embedding of the empty negative prompt (the unconditional CFG branch)
  clip_{gid:05d}.pt   for each prompt index gid in [gid_start, gid_end): the final denoised latent
                      x0 (output_type="latent", no VAE round trip), the conditional UMT5 embedding
                      and the prompt. Seed = 1000 + gid, so any sharding reproduces the same clips.

Launch: torchrun --standalone --nproc_per_node 8 selfgen.py --out $OUT_DIR/teacher_clips --gid_end 500
Existing clips are skipped, so an interrupted run can be resumed.
"""
import argparse, itertools, json, os, random, time, traceback

import torch

SUBJECTS = [
    "a red sports car", "a galloping horse", "a soaring eagle", "a running cheetah",
    "a surfer", "a downhill skier", "a hummingbird", "a school of fish", "a paper airplane",
    "a bouncing basketball", "a spinning top", "a waving flag", "a steam locomotive",
    "a sailboat", "a hot-air balloon", "a skateboarder", "a dolphin", "a race drone",
    "a herd of running deer", "a fire truck", "a windmill", "a kite", "a motorcyclist",
    "a rolling boulder", "a flock of birds", "a speedboat", "a cyclist", "a ballerina",
]
ACTIONS = [
    "racing forward", "leaping over an obstacle", "diving steeply then pulling up",
    "weaving left and right", "accelerating into the distance", "spinning rapidly",
    "splashing through water", "banking into a sharp turn", "tumbling down a slope",
    "gliding across the frame", "bursting into motion", "chasing after something",
]
SETTINGS = [
    "on a coastal highway at sunset", "across a snowy mountain pass", "through a dense forest",
    "over a calm ocean", "in a neon-lit city at night", "across a golden desert dune",
    "under a dramatic stormy sky", "along a rain-slicked street", "over rolling green hills",
    "through a canyon at dawn", "across an open savanna", "in a crowded stadium",
]
CAMERA = [
    "tracking shot", "sweeping aerial drone shot", "low-angle follow cam",
    "slow-motion close-up", "wide cinematic pan", "dynamic handheld shot",
]


def make_prompts(n=600, seed=1234):
    """Deterministic (subject x action x setting) prompts with a random camera move, so every clip
    has motion. A larger n keeps the first entries unchanged."""
    rng = random.Random(seed)
    combos = list(itertools.product(SUBJECTS, ACTIONS, SETTINGS))
    rng.shuffle(combos)
    prompts, seen = [], set()
    for s, act, setg in combos:
        cam = rng.choice(CAMERA)
        p = f"{cam} of {s} {act} {setg}, highly detailed, realistic motion, 4k"
        if p in seen:
            continue
        seen.add(p)
        prompts.append(p)
        if len(prompts) >= n:
            break
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers", help="HF id or local dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="", help="JSON list of prompts; default: make_prompts()")
    ap.add_argument("--n_prompts", type=int, default=600)
    ap.add_argument("--prompt_seed", type=int, default=1234)
    ap.add_argument("--gid_start", type=int, default=0)
    ap.add_argument("--gid_end", type=int, default=500, help="0 = all prompts")
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=5.0)
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    os.makedirs(a.out, exist_ok=True)

    prompts = json.load(open(a.prompts)) if a.prompts else make_prompts(a.n_prompts, a.prompt_seed)
    if rank == 0:
        json.dump(prompts, open(os.path.join(a.out, "prompts.json"), "w"), indent=1)

    from diffusers import WanPipeline
    pipe = WanPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)
    pipe.set_progress_bar_config(disable=True)

    uncond_path = os.path.join(a.out, "uncond_embeds.pt")
    if rank == 0 and not os.path.exists(uncond_path):
        # WanPipeline uses "" as the negative prompt when none is given, so this is exactly the
        # unconditional branch evaluated at guidance_scale > 1.
        with torch.no_grad():
            emb = pipe.encode_prompt(prompt=[""], device=dev, num_videos_per_prompt=1,
                                     do_classifier_free_guidance=False)[0]
        torch.save({"uncond_embeds": emb.to(torch.bfloat16).cpu(), "negative_prompt": ""}, uncond_path)

    end = a.gid_end or len(prompts)
    mine = list(enumerate(prompts))[a.gid_start:end][rank::world]
    if rank == 0:
        print(f"[selfgen] {len(prompts)} prompts, gids {a.gid_start}..{end - 1}, {len(mine)} on rank 0, "
              f"{a.frames}f {a.height}x{a.width}", flush=True)

    done, t0 = 0, time.time()
    for gid, prompt in mine:
        outp = os.path.join(a.out, f"clip_{gid:05d}.pt")
        if os.path.exists(outp):
            done += 1
            continue
        try:
            gen = torch.Generator(device=dev).manual_seed(1000 + gid)
            with torch.no_grad():
                prompt_embeds = pipe.encode_prompt(prompt=[prompt], device=dev, num_videos_per_prompt=1,
                                                   do_classifier_free_guidance=False)[0]
                out = pipe(prompt=prompt, height=a.height, width=a.width, num_frames=a.frames,
                           num_inference_steps=a.steps, guidance_scale=a.guidance,
                           output_type="latent", generator=gen)
                latent = out.frames if hasattr(out, "frames") else out[0]
            torch.save({"latent": latent.detach().to(torch.bfloat16).cpu(),
                        "prompt_embeds": prompt_embeds.detach().to(torch.bfloat16).cpu(),
                        "prompt": prompt, "frames": a.frames, "height": a.height, "width": a.width}, outp)
            done += 1
            if rank == 0 and done % 5 == 0:
                print(f"[selfgen r0] {done}/{len(mine)}  {(time.time() - t0) / done:.1f}s/clip", flush=True)
        except Exception as e:
            print(f"[selfgen r{rank}] FAIL clip {gid}: {e}", flush=True)
            traceback.print_exc()
    print(f"[selfgen r{rank}] DONE {done}/{len(mine)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
