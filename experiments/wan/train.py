#!/usr/bin/env python3
"""
DoRA self-distillation of INTRA attention into Wan2.2-TI2V-5B.

Teacher: frozen dense Wan. Student: the same frozen weights + DoRA adapters + INTRA attn1 (blocks
0 and 29 stay dense). Each step draws a cached teacher clip x0 (selfgen.py), a flow-matching noise
level sigma (logit-normal u mapped through flow_shift = 5) and noise, forms
x_t = (1 - sigma) x0 + sigma noise, and minimises MSE(student(x_t), teacher(x_t)) on identical
inputs. With probability --p_uncond the step uses the empty-prompt embedding instead of the clip's
prompt, so the unconditional branch that classifier-free guidance also evaluates is trained too.
Only the adapters train (Prodigy, lr 1, grad-norm clip 1, bf16, gradient checkpointing, DDP).

Defaults are the recipe of the checkpoint evaluated in the paper (rank 32, 20000 steps, 8 GPUs x
batch 1). Launch: torchrun --standalone --nproc_per_node 8 train.py --cache $OUT_DIR/teacher_clips --out ...
"""
import argparse, glob, json, os, random, time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from intra_wan import (add_pattern_args, config_from_args, install_intra_processors, set_intra_grid,
                       add_dora_adapter)


def log(msg):
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_adapter(model, path):
    """Only the DoRA tensors (lora_A/B and magnitude vectors)."""
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}, path)


def sample_sigma(bsz, device, flow_shift=5.0):
    u = torch.sigmoid(torch.randn(bsz, device=device))      # logit-normal(0, 1)
    return flow_shift * u / (1.0 + (flow_shift - 1.0) * u)


class ClipCache(torch.utils.data.Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "clip_*.pt")))
        assert self.files, f"no clips in {cache_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu")
        lat, emb = d["latent"], d["prompt_embeds"]
        return (lat.squeeze(0) if lat.dim() == 5 else lat), (emb.squeeze(0) if emb.dim() == 3 else emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers", help="HF id or local dir")
    ap.add_argument("--cache", required=True, help="selfgen.py output directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--ffn_lora", type=int, default=1, help="also adapt the FFN projections")
    ap.add_argument("--flow_shift", type=float, default=5.0)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--p_uncond", type=float, default=0.5, help="P(step trains the unconditional branch)")
    ap.add_argument("--uncond", default="", help="default: <cache>/uncond_embeds.pt")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=100)
    add_pattern_args(ap)
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    os.makedirs(a.out, exist_ok=True)
    if rank == 0:
        json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=2)

    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel, WanAttnProcessor
    cfg = config_from_args(a)
    log(f"pattern {cfg}")

    teacher = WanTransformer3DModel.from_pretrained(a.model, subfolder="transformer",
                                                    torch_dtype=torch.bfloat16).to(dev).eval()
    teacher.requires_grad_(False)
    for b in teacher.blocks:
        b.attn1.set_processor(WanAttnProcessor())
    student = WanTransformer3DModel.from_pretrained(a.model, subfolder="transformer",
                                                    torch_dtype=torch.bfloat16).to(dev)
    student.requires_grad_(False)
    install_intra_processors(student, cfg)                   # blocks 0 and last stay dense
    targets = add_dora_adapter(student, a.lora_rank, bool(a.ffn_lora))
    if a.grad_ckpt:
        student.enable_gradient_checkpointing()
    trainable = [p for p in student.parameters() if p.requires_grad]
    log(f"trainable {sum(p.numel() for p in trainable) / 1e6:.2f}M params, DoRA targets {targets}")

    uncond = None
    if a.p_uncond > 0:
        d = torch.load(a.uncond or os.path.join(a.cache, "uncond_embeds.pt"), map_location="cpu")
        uncond = d["uncond_embeds"]
        uncond = (uncond.squeeze(0) if uncond.dim() == 3 else uncond).to(dev, torch.bfloat16).unsqueeze(0)

    ds = ClipCache(a.cache)
    sampler = torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    dl = torch.utils.data.DataLoader(ds, batch_size=1, sampler=sampler, shuffle=(sampler is None),
                                     num_workers=2, pin_memory=True, drop_last=True)
    log(f"dataset {len(ds)} clips")
    p_t, p_h, p_w = student.config.patch_size

    model = student
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            student, device_ids=[local], output_device=local, find_unused_parameters=False)

    from prodigyopt import Prodigy
    opt = Prodigy(trainable, lr=1.0, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8,
                  decouple=True, use_bias_correction=True, safeguard_warmup=True)

    step, micro, running, epoch, t0 = 0, 0, 0.0, 0, time.time()
    model.train()
    if sampler is not None:
        sampler.set_epoch(0)
    data_iter = iter(dl)
    while step < a.steps:
        try:
            latent, ehs = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(dl)
            latent, ehs = next(data_iter)
        x0 = latent.to(dev, torch.bfloat16)              # (1, C, T, H, W)
        ehs = ehs.to(dev, torch.bfloat16)                # (1, L, dim)
        _, _, Tl, Hl, Wl = x0.shape
        grid = (Tl // p_t, Hl // p_h, Wl // p_w)
        set_intra_grid(student, grid)

        with torch.no_grad():
            sig = sample_sigma(1, dev, a.flow_shift)
            sigma = sig.view(1, 1, 1, 1, 1).to(torch.bfloat16)
            timestep = (sig * 1000.0).to(torch.float32)
            x_t = (1 - sigma) * x0 + sigma * torch.randn_like(x0)

        cond_in = uncond if (uncond is not None and random.random() < a.p_uncond) else ehs
        with torch.no_grad():
            teacher_out = teacher(x_t, timestep, cond_in, return_dict=False)[0]
        student_out = model(x_t, timestep, cond_in, return_dict=False)[0]
        loss = F.mse_loss(student_out.float(), teacher_out.float())
        (loss / a.grad_accum).backward()
        running += loss.item()
        micro += 1

        if micro % a.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0:
                log(f"step {step}/{a.steps} loss {running / (a.log_every * a.grad_accum):.5f} "
                    f"{(time.time() - t0) / step:.1f}s/step grid{grid}")
                running = 0.0
            if step % a.save_every == 0 and rank == 0:
                save_adapter(student, os.path.join(a.out, f"lora_step{step}.pt"))

    if rank == 0:
        save_adapter(student, os.path.join(a.out, "lora_final.pt"))
        log(f"done: {a.steps} steps in {(time.time() - t0) / 60:.1f} min -> {a.out}/lora_final.pt")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
