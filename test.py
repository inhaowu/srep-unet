import argparse
import csv
import os

import numpy as np
import torch

from common_args import add_common_args
from srepunet import build_model
from srepunet.dataset import NPZSrepDataset, read_split
from srepunet import metrics as M


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--mesh_dir", default=None, help="directory of <case>_SPHARM.vtk for surface metrics")
    a = p.parse_args()

    model = build_model(a).cuda()
    ck = torch.load(a.checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded {a.checkpoint} (epoch {ck.get('epoch')})")

    ids = read_split(a.splits_dir, a.fold, a.split)
    ds = NPZSrepDataset(ids, a.data_dir, a.srep_dir, a.srep_suffix, (a.roi,) * 3)
    scale = np.array([a.roi] * 3, dtype=np.float64)
    os.makedirs(os.path.join(a.out, "srep_pred"), exist_ok=True)
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            cid = b["case_id"]
            with torch.autocast("cuda", enabled=not a.no_amp):
                out = model(b["image"][None].cuda())
            seg_logits = out if a.seg_only else out[0]
            pred = torch.argmax(seg_logits, 1)[0].cpu().numpy().astype(np.uint8)
            lab = b["label"][0].numpy().astype(np.uint8)
            r = dict(case=cid, dice=M.dice(pred, lab), hd95=M.hd95(pred, lab))
            if not a.seg_only:
                Xn = out[1][0].float().cpu().numpy().astype(np.float64)
                Gn = b["srep"].numpy().astype(np.float64)
                X, G = Xn * scale, Gn * scale
                np.save(os.path.join(a.out, "srep_pred", f"{cid}_srep_pred.npy"), X.astype(np.float32))
                r.update(point_err=M.point_error(X, G), rmse_norm=float(np.sqrt(np.mean((Xn - Gn) ** 2))),
                         angle=M.spoke_angle(X, G), medialness=M.medialness(X), medialness_ref=M.medialness(G))
                if a.mesh_dir:
                    mesh = os.path.join(a.mesh_dir, f"{cid}_SPHARM.vtk")
                    if not os.path.isfile(mesh):
                        raise FileNotFoundError(mesh)
                    surf = M.read_vtk(mesh)
                    r["ortho"], r["tip_surface"] = M.surface_metrics(X, surf)
                    r["ortho_ref"], r["tip_surface_ref"] = M.surface_metrics(G, surf)
            rows.append(r)
            print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)

    keys = list(rows[0])
    mean = {"case": "MEAN", **{k: float(np.nanmean([r[k] for r in rows])) for k in keys[1:]}}
    std = {"case": "STD", **{k: float(np.nanstd([r[k] for r in rows])) for k in keys[1:]}}
    with open(os.path.join(a.out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([mean, std] + rows)
    print("MEAN  " + "  ".join(f"{k}={mean[k]:.4f}" for k in keys[1:]))


if __name__ == "__main__":
    main()
