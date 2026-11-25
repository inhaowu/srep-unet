import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage

STRUCTURES = {"anterior": (1, (2, 0, 1)), "posterior": (2, (0, 1, 2))}


def sdf_and_normals(mask):
    sdf = (ndimage.distance_transform_edt(~mask) - ndimage.distance_transform_edt(mask)).astype(np.float32)
    gz, gy, gx = np.gradient(sdf)
    n = np.stack([gx, gy, gz], 0)
    norm = np.linalg.norm(n, axis=0, keepdims=True)
    n = n / np.maximum(norm, 1e-8)
    n[:, norm[0] < 1e-8] = 0.0
    return sdf, n.astype(np.float16)


def process(args):
    case, msd_dir, out_dir, size = args
    img = np.asarray(nib.load(os.path.join(msd_dir, "imagesTr", f"{case}.nii.gz")).dataobj, dtype=np.float32)
    lab = np.asarray(nib.load(os.path.join(msd_dir, "labelsTr", f"{case}.nii.gz")).dataobj).astype(np.int16)
    if any(s > size for s in img.shape):
        raise ValueError(f"{case}: shape {img.shape} larger than {size}")
    off = [(size - s) // 2 for s in img.shape]
    sl = tuple(slice(o, o + s) for o, s in zip(off, img.shape))
    I = np.zeros((size,) * 3, np.float32)
    I[sl] = img
    for name, (value, axes) in STRUCTURES.items():
        m = np.zeros((size,) * 3, bool)
        m[sl] = lab == value
        m = np.ascontiguousarray(m.transpose(axes))
        sdf, normals = sdf_and_normals(m)
        np.savez_compressed(os.path.join(out_dir, "npz", f"{case}_{name}.npz"),
                            image=np.ascontiguousarray(I.transpose(axes)), label=m.astype(np.uint8),
                            sdf_volume=sdf, normals_volume=normals)
    return case, off


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msd_dir", required=True, help="extracted Task04_Hippocampus directory")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, "npz"), exist_ok=True)
    cases = sorted(f[:-7] for f in os.listdir(os.path.join(a.msd_dir, "imagesTr"))
                   if f.endswith(".nii.gz") and not f.startswith("."))
    with ProcessPoolExecutor(a.workers) as ex:
        offsets = dict(ex.map(process, [(c, a.msd_dir, a.out_dir, a.size) for c in cases]))
    json.dump(offsets, open(os.path.join(a.out_dir, "padding_offsets.json"), "w"), indent=1)
    print(f"{len(cases)} images -> {2 * len(cases)} cases in {a.out_dir}/npz")


if __name__ == "__main__":
    main()
