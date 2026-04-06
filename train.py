"""
train.py — Training loop for FaceDiT.

Usage:
    python train.py --data_path /workspace/celebvhq_latents.pt --output_dir /workspace/checkpoints

Features:
    - bf16 mixed precision
    - torch.compile
    - EMA weights
    - Learning rate warmup
    - Gradient clipping
    - Checkpointing with resume support
    - Periodic sample generation
"""

import argparse
import os
import time
import json

import torch
from torch.utils.data import DataLoader

from dit import FaceDiT
from dataset import FaceVideoDataset
from flow_matching import train_step, sample


def train(config):
    device = "cuda"

    # ── Model ──
    model = FaceDiT(
        hidden_dim=config["hidden_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params ({total_params/1e6:.1f}M)")

    model = torch.compile(model)

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.99),
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / config["warmup_steps"]),
    )

    # ── Data ──
    train_dataset = FaceVideoDataset(config["data_path"], split="train")
    val_dataset = FaceVideoDataset(config["data_path"], split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    print(f"Train: {len(train_dataset)} clips, {len(train_loader)} batches/epoch")
    print(f"Val:   {len(val_dataset)} clips")

    # ── EMA ──
    ema_weights = {}
    for name, param in model.named_parameters():
        ema_weights[name] = param.data.clone()

    # ── Resume ──
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    ckpt_path = os.path.join(config["output_dir"], "latest.pt")

    if os.path.exists(ckpt_path) and config["resume"]:
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        ema_weights = ckpt["ema_weights"]
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    os.makedirs(config["output_dir"], exist_ok=True)

    # Save config
    with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Training ──
    for epoch in range(start_epoch, config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch_idx, (latents, emotion_idx) in enumerate(train_loader):
            latents = latents.to(device)

            loss = train_step(model, latents, optimizer)

            scheduler.step()

            # EMA update
            with torch.no_grad():
                for name, param in model.named_parameters():
                    ema_weights[name].lerp_(param.data, 1.0 - config["ema_decay"])

            epoch_loss += loss
            num_batches += 1
            global_step += 1

            # Log
            if global_step % config["log_every"] == 0:
                avg_loss = epoch_loss / num_batches
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                steps_per_sec = num_batches / elapsed
                print(
                    f"Epoch {epoch} | Step {global_step} | "
                    f"Loss {loss:.4f} | Avg {avg_loss:.4f} | "
                    f"LR {lr:.2e} | {steps_per_sec:.1f} steps/s"
                )

        epoch_loss /= max(num_batches, 1)
        epoch_time = time.time() - t0
        print(f"Epoch {epoch} done | Loss: {epoch_loss:.4f} | Time: {epoch_time:.0f}s")

        # ── Validation ──
        if (epoch + 1) % config["val_every"] == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for latents, emotion_idx in val_loader:
                    latents = latents.to(device)
                    x1 = latents
                    x0 = torch.randn_like(x1)
                    t = torch.rand(x1.shape[0], device=device)
                    t_expand = t[:, None, None, None, None]
                    xt = (1 - t_expand) * x0 + t_expand * x1
                    v = x1 - x0
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        v_pred = model(xt, t)
                        loss = torch.nn.functional.mse_loss(v_pred, v)
                    val_loss += loss.item()
                    val_batches += 1

            val_loss /= max(val_batches, 1)
            print(f"  Val loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {name: ema_weights[name].clone() for name in ema_weights},
                    os.path.join(config["output_dir"], "best_ema.pt"),
                )
                print(f"  New best! Saved best_ema.pt")

            model.train()

        # ── Checkpoint ──
        if (epoch + 1) % config["save_every"] == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ema_weights": ema_weights,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint at epoch {epoch}")

        # ── Generate samples ──
        if (epoch + 1) % config["sample_every"] == 0:
            model.eval()

            # Swap in EMA weights
            orig_weights = {}
            for name, param in model.named_parameters():
                orig_weights[name] = param.data.clone()
                param.data.copy_(ema_weights[name])

            samples = sample(
                model,
                shape=(4, 16, 4, 32, 32),
                num_steps=50,
                device=device,
            )
            torch.save(
                samples,
                os.path.join(config["output_dir"], f"samples_epoch{epoch:04d}.pt"),
            )
            print(f"  Saved samples at epoch {epoch}")

            # Restore training weights
            for name, param in model.named_parameters():
                param.data.copy_(orig_weights[name])

            model.train()

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--sample_every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = vars(args)
    train(config)