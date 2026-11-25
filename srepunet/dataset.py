import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NPZSrepDataset(Dataset):
    def __init__(self, case_ids, npz_dir, srep_dir=None, srep_suffix="_srep.npy", size=(64, 64, 64)):
        self.case_ids = list(case_ids)
        self.npz_dir = npz_dir
        self.srep_dir = srep_dir
        self.srep_suffix = srep_suffix
        self.size = size

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        cid = self.case_ids[idx]
        z = np.load(os.path.join(self.npz_dir, f"{cid}.npz"))
        img = z["image"].astype(np.float32)
        if img.std() > 0:
            img = (img - img.mean()) / img.std()
        out = {"case_id": cid,
               "image": torch.from_numpy(img[None]),
               "label": torch.from_numpy(z["label"].astype(np.float32)[None]),
               "normals_volume": torch.from_numpy(z["normals_volume"].astype(np.float32)),
               "sdf_volume": torch.from_numpy(z["sdf_volume"].astype(np.float32))}
        if self.srep_dir is not None:
            X = np.load(os.path.join(self.srep_dir, f"{cid}{self.srep_suffix}")).astype(np.float32)
            D, H, W = self.size
            out["srep"] = torch.from_numpy(X / np.array([W, H, D], dtype=np.float32))
        return out


def read_split(splits_dir, fold, split):
    with open(os.path.join(splits_dir, f"fold{fold}", f"{split}.txt")) as f:
        return [s.strip() for s in f if s.strip()]


def get_loaders(args):
    size = (args.roi, args.roi, args.roi)
    kw = dict(npz_dir=args.data_dir, srep_dir=args.srep_dir, srep_suffix=args.srep_suffix, size=size)
    train_ids = read_split(args.splits_dir, args.fold, "train")
    if args.training_ratio < 1.0:
        train_ids = random.Random(42).sample(train_ids, max(1, int(len(train_ids) * args.training_ratio)))
    train_ds = NPZSrepDataset(train_ids, **kw)
    val_ds = NPZSrepDataset(read_split(args.splits_dir, args.fold, "valid"), **kw)
    print(f"train {len(train_ds)} | valid {len(val_ds)}")

    g = torch.Generator()
    g.manual_seed(args.seed)

    def seed_worker(worker_id):
        s = (args.seed + worker_id) % 2 ** 32
        np.random.seed(s)
        random.seed(s)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True,
                            worker_init_fn=seed_worker)
    return train_loader, val_loader
