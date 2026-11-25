import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common_args import add_common_args
from srepunet import build_model
from srepunet.dataset import get_loaders
from srepunet.losses import kl_loss, orth_loss, seg_loss, spoke_loss
from srepunet.lr_scheduler import LinearWarmupCosineAnnealingLR
from srepunet.metrics import dice, spoke_angle


def parse_args():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--logdir", required=True)
    p.add_argument("--max_epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=3, help="learning-rate warm-up")
    p.add_argument("--patience", type=int, default=30, help="early stopping on validation Dice")
    p.add_argument("--training_ratio", type=float, default=1.0)
    p.add_argument("--srep_weight", type=float, default=0.1)
    p.add_argument("--srep_warmup_epochs", type=int, default=20)
    p.add_argument("--kld_weight", type=float, default=1e-4)
    p.add_argument("--spoke_weight", type=float, default=0.1, help="0 disables L_spoke")
    p.add_argument("--orth_weight", type=float, default=0.1, help="0 disables L_orth")
    return p.parse_args()


def srep_weight(epoch, a):
    if epoch < a.srep_warmup_epochs:
        return 0.0
    return min(a.srep_weight, a.srep_weight * (epoch - a.srep_warmup_epochs + 1) / max(1, a.srep_warmup_epochs))


def train_epoch(model, loader, opt, scaler, epoch, a):
    model.train()
    sw = srep_weight(epoch, a)
    amp = not a.no_amp
    tot, n = {}, 0
    for batch in loader:
        img, label = batch["image"].cuda(non_blocking=True), batch["label"].cuda(non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            if a.seg_only:
                seg_logits = model(img)
            else:
                seg_logits, srep = model(img, use_skeleton=sw > 0)
            parts = {"seg": seg_loss(seg_logits, label)}
            loss = parts["seg"]
            if not a.seg_only and sw > 0:
                target = batch["srep"].cuda(non_blocking=True)
                head = model.srep_model
                parts["mse"] = F.mse_loss(srep, target)
                parts["kl"] = kl_loss(head.mu, head.log_var)
                loss = loss + sw * (parts["mse"] + a.kld_weight * parts["kl"])
                if a.spoke_weight > 0:
                    parts["spoke"] = spoke_loss(srep, target)
                    loss = loss + sw * a.spoke_weight * parts["spoke"]
                if a.orth_weight > 0:
                    parts["orth"] = orth_loss(srep, batch["normals_volume"].cuda(non_blocking=True), (a.roi,) * 3)
                    loss = loss + sw * a.orth_weight * parts["orth"]
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        B = img.shape[0]
        n += B
        for k, v in dict(loss=loss, **parts).items():
            tot[k] = tot.get(k, 0.0) + v.item() * B
    return {k: v / n for k, v in tot.items()}, sw


@torch.no_grad()
def validate(model, loader, a):
    model.eval()
    dices, angles = [], []
    for batch in loader:
        img = batch["image"].cuda(non_blocking=True)
        with torch.autocast("cuda", enabled=not a.no_amp):
            out = model(img)
        seg_logits, srep = (out, None) if a.seg_only else out
        pred = (torch.softmax(seg_logits, 1)[0, 1] > 0.5).cpu().numpy()
        dices.append(dice(pred, batch["label"][0, 0].numpy()))
        if srep is not None:
            angles.append(spoke_angle(srep[0].float().cpu().numpy(), batch["srep"][0].numpy()))
    return float(np.mean(dices)), (float(np.mean(angles)) if angles else None)


def main():
    a = parse_args()
    if not a.seg_only and a.srep_dir is None:
        raise SystemExit("--srep_dir is required unless --seg_only")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.makedirs(a.logdir, exist_ok=True)

    train_loader, val_loader = get_loaders(a)
    model = build_model(a).cuda()
    print(f"parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=a.warmup_epochs, max_epochs=a.max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=not a.no_amp)

    def save(name, epoch):
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "args": vars(a)}, os.path.join(a.logdir, name))

    best_dice, best_angle, stale = -1.0, float("inf"), 0
    log = open(os.path.join(a.logdir, "log.csv"), "w", newline="")
    writer = None
    for epoch in range(a.max_epochs):
        t0 = time.time()
        losses, sw = train_epoch(model, train_loader, opt, scaler, epoch, a)
        vd, va = validate(model, val_loader, a)
        row = dict(epoch=epoch, srep_weight=sw, lr=opt.param_groups[0]["lr"], val_dice=vd, val_angle=va, **losses)
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(row) + ["mse", "kl", "spoke", "orth"], extrasaction="ignore")
            writer.writeheader()
        writer.writerow(row)
        log.flush()
        print(f"epoch {epoch}  " + "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
              + f"  sw={sw:.3f}  val_dice={vd:.4f}" + (f"  val_angle={va:.4f}" if va is not None else "")
              + f"  {time.time() - t0:.0f}s", flush=True)
        if vd > best_dice:
            best_dice, stale = vd, 0
            save("model_best_dice.pt", epoch)
        else:
            stale += 1
        if va is not None and va < best_angle:
            best_angle = va
            save("model_best_angle.pt", epoch)
        save("model_last.pt", epoch)
        if stale >= a.patience:
            print(f"early stop: no Dice improvement for {a.patience} epochs")
            break
        sched.step()
    log.close()


if __name__ == "__main__":
    main()
