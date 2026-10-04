"""Train ArcFace on CASIA-WebFace with an identity-disjoint train/val/test split.

  python train.py                      # train (validates every val_interval epochs)
  python train.py --batch_size 128     # smaller GPU
  python evaluate.py                   # final test-set evaluation of best checkpoint
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from datasets.casia import FaceDataset, train_transform
from datasets.split import get_pairs, make_split
from datasets.verification import evaluate
from models.arcface import ArcFaceHead
from models.iresnet import build_backbone


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=cfg.batch_size)
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--train_dir", default=cfg.train_dir)
    ap.add_argument("--resume", default=cfg.resume)
    args = ap.parse_args()
    cfg.batch_size, cfg.epochs, cfg.train_dir, cfg.resume = args.batch_size, args.epochs, args.train_dir, args.resume

    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data: identity-disjoint split ---------------------------------
    split = make_split(cfg)
    val_pairs = get_pairs(cfg, split, "val")
    train_set = FaceDataset(cfg.train_dir, split["train"], train_transform())
    print(f"identities  train={len(split['train'])}  val={len(split['val'])}  test={len(split['test'])}")
    print(f"train images={len(train_set)}  val pairs={len(val_pairs)}")
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, pin_memory=True)

    # ---- model ---------------------------------------------------------
    backbone = build_backbone(cfg.backbone, cfg.embedding_size, cfg.dropout).to(device)
    head = ArcFaceHead(cfg.embedding_size, len(train_set.classes), cfg.s, cfg.m, cfg.easy_margin).to(device)
    opt = torch.optim.SGD([{"params": backbone.parameters()}, {"params": head.parameters()}],
                          lr=cfg.scaled_lr(), momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, cfg.lr_milestones, cfg.lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")
    ce = nn.CrossEntropyLoss()

    start_epoch, best_val = 0, 0.0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device)
        backbone.load_state_dict(ck["backbone"]); head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best_val = ck["epoch"] + 1, ck.get("best_val", 0.0)

    # ---- train ---------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        backbone.train(); head.train()
        run_loss = run_acc = 0.0
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=cfg.use_amp and device.type == "cuda"):
                emb = backbone(x)
            logits, plain = head(emb, y)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), cfg.grad_clip_norm)
            scaler.step(opt); scaler.update()
            run_loss += loss.item(); run_acc += (plain.argmax(1) == y).float().mean().item()
            if step % cfg.log_interval == 0:
                print(f"ep {epoch+1}/{cfg.epochs} step {step}/{len(loader)} "
                      f"loss {run_loss/step:.4f} train-acc {run_acc/step:.4f} lr {opt.param_groups[0]['lr']:.5f}")
        sched.step()

        # ---- validation on UNSEEN identities ---------------------------
        if (epoch + 1) % cfg.val_interval == 0:
            r = evaluate(backbone, cfg.train_dir, val_pairs, device, num_workers=cfg.num_workers)
            print(f"[VAL] epoch {epoch+1}: verification acc {r['acc']:.4f} (thr {r['threshold']:.3f})")
            if r["acc"] > best_val:
                best_val = r["acc"]
                torch.save({"backbone": backbone.state_dict(), "threshold": r["threshold"],
                            "epoch": epoch, "val_acc": best_val},
                           os.path.join(cfg.checkpoint_dir, "best.pt"))
        if (epoch + 1) % cfg.save_interval == 0:
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch, "best_val": best_val},
                       os.path.join(cfg.checkpoint_dir, "last.pt"))

    print(f"Done. Best val acc {best_val:.4f}. Run `python evaluate.py` for the final TEST score.")


if __name__ == "__main__":
    main()
