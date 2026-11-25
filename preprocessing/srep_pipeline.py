import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
PAD = 8


def log(*a):
    print(*a, flush=True)


def load_label(path, value):
    if path.endswith(".npz"):
        with np.load(path) as z:
            lab = z["label"]
    else:
        import nibabel as nib
        lab = np.asarray(nib.load(path).dataobj)
    lab = np.squeeze(lab)
    return (lab == value) if value is not None else (lab > 0.5)


def clean(mask):
    lab, n = ndimage.label(mask)
    if n > 1:
        mask = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndimage.binary_fill_holes(mask)


def write_nrrd(path, mask, origin):
    Z, Y, X = mask.shape
    hdr = ("NRRD0004\ntype: uint8\ndimension: 3\nspace: left-posterior-superior\n"
           f"sizes: {X} {Y} {Z}\nspace directions: (1,0,0) (0,1,0) (0,0,1)\nkinds: domain domain domain\n"
           f"endian: little\nencoding: raw\nspace origin: ({origin[0]},{origin[1]},{origin[2]})\n\n")
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())


def prep_one(args):
    case, src, out, value = args
    dst = f"{out}/nrrd/{case}.nrrd"
    if os.path.exists(dst):
        return case, "exists"
    m = clean(load_label(src, value))
    if m.sum() < 50:
        return case, f"too small ({int(m.sum())} voxels)"
    write_nrrd(dst, np.pad(m, PAD), (-PAD, -PAD, -PAD))
    return case, "ok"


def run(cmd, logf, env=None, cwd=None):
    with open(logf, "a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd).returncode


def spharm_one(args):
    case, out, salt, cli, iters, subdiv, degree = args
    dst = f"{out}/mesh/{case}_SPHARM.vtk"
    if os.path.exists(dst):
        return case, "exists"
    w = f"{out}/spharm_work/{case}"
    os.makedirs(w, exist_ok=True)
    lg = f"{w}/log.txt"
    L = [f"{salt}/SlicerSALT", "--launch"]
    steps = [
        L + [f"{cli}/SegPostProcessCLP", f"{out}/nrrd/{case}.nrrd", f"{w}/{case}_pp.nrrd", "--label", "1"],
        L + [f"{cli}/GenParaMeshCLP", f"{w}/{case}_pp.nrrd", f"{w}/{case}_para.vtk", f"{w}/{case}_surf.vtk",
             "--iter", str(iters), "--label", "1"],
        L + [f"{cli}/ParaToSPHARMMeshCLP", f"{w}/{case}_para.vtk", f"{w}/{case}_surf.vtk", f"{w}/{case}_",
             "--subdivLevel", str(subdiv), "--spharmDegree", str(degree)],
    ]
    for c in steps:
        rc = run(c, lg)
        if rc != 0:
            return case, f"failed ({os.path.basename(c[2])}, rc {rc}; see {lg})"
    cand = [f for f in os.listdir(w) if f.endswith("SPHARM.vtk") and "ellalign" not in f and "procalign" not in f]
    if not cand:
        return case, "no SPHARM output"
    shutil.copy(f"{w}/{cand[0]}", dst)
    return case, "ok"


def read_mesh(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    r = vtk.vtkPolyDataReader()
    r.SetFileName(path)
    r.Update()
    pd = r.GetOutput()
    tri = vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return pd, vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64), tri


def write_mesh_points(pd, pts, path):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    out = vtk.vtkPolyData()
    out.DeepCopy(pd)
    vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(np.ascontiguousarray(pts, dtype=np.float32), deep=True))
    out.SetPoints(vp)
    w = vtk.vtkPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(out)
    w.Write()


def mesh_inside(pd, shape):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    imp = vtk.vtkImplicitPolyDataDistance()
    imp.SetInput(pd)
    Z, Y, X = shape
    s = vtk.vtkSampleFunction()
    s.SetImplicitFunction(imp)
    s.SetModelBounds(0, X - 1, 0, Y - 1, 0, Z - 1)
    s.SetSampleDimensions(X, Y, Z)
    s.ComputeNormalsOff()
    s.Update()
    return vtk_to_numpy(s.GetOutput().GetPointData().GetScalars()).reshape(Z, Y, X) < 0


def qc_one(args):
    case, src, out, value = args
    mp = f"{out}/mesh/{case}_SPHARM.vtk"
    if not os.path.exists(mp):
        return case, dict(status="no mesh")
    raw = f"{out}/mesh/{case}_SPHARM_raw.vtk"
    if not os.path.exists(raw):
        shutil.copy(mp, raw)
    pd, pts, tri = read_mesh(raw)
    pts = pts * np.array([-1.0, -1.0, 1.0])
    write_mesh_points(pd, pts, mp)
    pd, _, _ = read_mesh(mp)
    lab = clean(load_label(src, value))

    def dice_at(d):
        write_mesh_points(pd, pts + d, mp)
        ins = mesh_inside(read_mesh(mp)[0], lab.shape)
        return float(2 * (ins & lab).sum() / (ins.sum() + lab.sum() + 1e-9)), ins


    _, ins = dice_at(np.zeros(3))
    d0 = np.argwhere(lab).mean(0)[::-1] - np.argwhere(ins).mean(0)[::-1] if ins.any() else np.zeros(3)
    base = np.round(d0 * 2) / 2
    best = (-1.0, np.zeros(3))
    for off in np.stack(np.meshgrid(*[np.array([-0.5, 0, 0.5])] * 3, indexing="ij"), -1).reshape(-1, 3):
        dice, _ = dice_at(base + off)
        if dice > best[0]:
            best = (dice, base + off)
    dice, d = best
    write_mesh_points(pd, pts + d, mp)
    V, F = len(pts), len(tri)
    E = len({tuple(sorted(e)) for t in tri for e in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2]))})
    return case, dict(status="ok" if dice >= 0.95 and V - E + F == 2 else "check", dice=dice, shift=d.tolist(),
                      euler=V - E + F, n_points=V)


def json_to_207(path):
    d = json.load(open(path))
    skel, up, dn, cr, seen = [], [], [], [], set()
    for col in d["EllipticalSRep"]["Skeleton"]:
        for layer in col:
            sp = tuple(layer["UpSpoke"]["SkeletalPoint"]["Value"])
            if sp not in seen:
                seen.add(sp)
                p = np.array(sp, dtype=float)
                skel.append(p)
                up.append(p + np.array(layer["UpSpoke"]["Direction"]["Value"], dtype=float))
                dn.append(p + np.array(layer["DownSpoke"]["Direction"]["Value"], dtype=float))
            if "CrestSpoke" in layer:
                cs = layer["CrestSpoke"]
                cr.append(np.array(cs["SkeletalPoint"]["Value"], dtype=float) + np.array(cs["Direction"]["Value"], dtype=float))
    assert len(skel) == 61 and len(cr) == 24, (len(skel), len(cr))
    return np.concatenate([skel, up, dn, cr]).astype(np.float64)


def evo_one(args):
    case, out, gpu, salt, deformetrica = args
    dst = f"{out}/srep_evo/{case}_srep_evo.npy"
    if os.path.exists(dst):
        return case, "exists"
    inp = f"{out}/evo_in/{case}"
    os.makedirs(inp, exist_ok=True)
    if not os.path.exists(f"{inp}/{case}_SPHARM.vtk"):
        shutil.copy(f"{out}/mesh/{case}_SPHARM.vtk", f"{inp}/{case}_SPHARM.vtk")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SLICER_LIBS_DIR"] = _salt_libs(salt)
    env["HOME"] = f"{out}/evo_home/{case}"
    os.makedirs(env["HOME"], exist_ok=True)
    rc = run([f"{salt}/bin/PythonSlicer", os.path.join(HERE, "evo_fit.py"), "--input-dir", inp,
              "--output-base-dir", f"{out}/evo_runs", "--deformetrica-cmd", deformetrica, "--merge-crest",
              "--skip-existing", "--workers", "1"], f"{inp}/log.txt", env=env, cwd=salt)
    run_dir = f"{out}/evo_runs/{case}"
    js = f"{run_dir}/final_srep_merged.srep.json"
    js = js if os.path.exists(js) else f"{run_dir}/final_srep.srep.json"
    if not os.path.exists(js):
        return case, f"failed (rc {rc}; see {inp}/log.txt)"
    np.save(dst, json_to_207(js).astype(np.float32))
    return case, "ok"


def sync_stage(out, classes):
    sys.path.insert(0, HERE)
    from srep_sync import apply_elems, synchronise
    os.makedirs(f"{out}/srep", exist_ok=True)
    rep = {}
    for cls, cases in classes.items():
        cs = [c for c in cases if os.path.exists(f"{out}/srep_evo/{c}_srep_evo.npy")]
        if len(cs) < 3:
            log(f"sync {cls}: {len(cs)} s-reps, need at least 3 -- skipped")
            continue
        Xs = np.stack([np.load(f"{out}/srep_evo/{c}_srep_evo.npy").astype(np.float64) for c in cs])
        g = synchronise(Xs)
        for c, x in zip(cs, apply_elems(Xs, g)):
            np.save(f"{out}/srep/{c}_srep.npy", x.astype(np.float32))
        rep[cls] = dict(n=len(cs), elements={e: g.count(e) for e in sorted(set(g))})
        log(f"sync {cls}: {rep[cls]}")
    json.dump(rep, open(f"{out}/sync.json", "w"), indent=1)


def _salt_libs(salt):
    for sub in ("lib", "share"):
        for d in sorted(os.listdir(os.path.join(salt, sub))) if os.path.isdir(os.path.join(salt, sub)) else []:
            p = os.path.join(salt, sub, d, "qt-scripted-modules", "Resources", "Libraries")
            if d.startswith("SlicerSALT-") and os.path.isdir(p):
                return p
    raise FileNotFoundError(f"SlicerSALT scripted libraries not found under {salt}")


def _salt_cli(salt):
    for d in sorted(os.listdir(os.path.join(salt, "lib"))):
        p = os.path.join(salt, "lib", d, "cli-modules")
        if d.startswith("SlicerSALT-") and os.path.isdir(p):
            return p
    raise FileNotFoundError(f"SlicerSALT cli-modules not found under {salt}/lib")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prep", "spharm", "qc", "evo", "sync", "all"])
    ap.add_argument("--labels", required=True, help="directory of <case>.npz (key 'label') or <case>.nii[.gz]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slicersalt", default=os.environ.get("SLICERSALT_DIR"), help="SlicerSALT install directory")
    ap.add_argument("--deformetrica", default="deformetrica", help="Deformetrica executable")
    ap.add_argument("--cases", default="", help="comma-separated subset (default: every label file)")
    ap.add_argument("--label-value", type=float, default=None, help="label value of the structure (default: > 0.5)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--evo-workers", type=int, default=4, help="concurrent evolutionary fits")
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids for the evolutionary fits")
    ap.add_argument("--spharm-iter", type=int, default=1000)
    ap.add_argument("--subdiv", type=int, default=10)
    ap.add_argument("--degree", type=int, default=15)
    a = ap.parse_args()
    a.out, a.labels = os.path.abspath(a.out), os.path.abspath(a.labels)
    files = {}
    for f in sorted(os.listdir(a.labels)):
        for ext in (".npz", ".nii.gz", ".nii"):
            if f.endswith(ext) and not f.startswith("."):
                files[f[:-len(ext)]] = f"{a.labels}/{f}"
    cases = [c for c in (a.cases.split(",") if a.cases else files) if c in files]
    for d in ("nrrd", "mesh", "srep_evo"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    classes = {}
    for c in cases:
        cls = next((s for s in ("anterior", "posterior") if c.endswith("_" + s)), "all")
        classes.setdefault(cls, []).append(c)
    stages = ["prep", "spharm", "qc", "evo", "sync"] if a.stage == "all" else [a.stage]
    if any(s in stages for s in ("spharm", "evo")) and not a.slicersalt:
        raise SystemExit("--slicersalt (or env SLICERSALT_DIR) is required for the spharm and evo stages")
    for st in stages:
        log(f"== {st} ({len(cases)} cases)")
        if st == "prep":
            with cf.ProcessPoolExecutor(a.workers) as ex:
                for c, s in ex.map(prep_one, [(c, files[c], a.out, a.label_value) for c in cases]):
                    s != "exists" and log(f"prep {c}: {s}")
        elif st == "spharm":
            cli = _salt_cli(a.slicersalt)
            with cf.ThreadPoolExecutor(a.workers) as ex:
                for c, s in ex.map(spharm_one, [(c, a.out, a.slicersalt, cli, a.spharm_iter, a.subdiv, a.degree)
                                                for c in cases]):
                    s != "exists" and log(f"spharm {c}: {s}")
        elif st == "qc":
            qp = f"{a.out}/qc.json"
            qc = json.load(open(qp)) if os.path.exists(qp) else {}
            todo = [c for c in cases if c not in qc]
            with cf.ProcessPoolExecutor(a.workers) as ex:
                for c, r in ex.map(qc_one, [(c, files[c], a.out, a.label_value) for c in todo]):
                    qc[c] = r
                    log(f"qc {c}: {r}")
            json.dump(qc, open(qp, "w"), indent=1)
            bad = [c for c in cases if qc.get(c, {}).get("status") != "ok"]
            bad and log(f"qc: {len(bad)} cases need a look: {bad[:20]}")
        elif st == "evo":
            gpus = a.gpus.split(",")
            with cf.ThreadPoolExecutor(a.evo_workers) as ex:
                for c, s in ex.map(evo_one, [(c, a.out, gpus[i % len(gpus)], a.slicersalt, a.deformetrica)
                                             for i, c in enumerate(cases)]):
                    s != "exists" and log(f"evo {c}: {s}")
        elif st == "sync":
            sync_stage(a.out, classes)


if __name__ == "__main__":
    main()
