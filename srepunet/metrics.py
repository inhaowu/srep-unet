import numpy as np

from .srep_head import CREST_SKEL_IDX


def dice(pred, target):
    p, t = pred > 0, target > 0
    return float(2.0 * (p & t).sum() / (p.sum() + t.sum() + 1e-5))


def hd95(pred, target, spacing=(1.0, 1.0, 1.0)):
    try:
        from medpy.metric.binary import hd95 as _hd95
    except ImportError:
        return float("nan")
    if not (pred > 0).any() or not (target > 0).any():
        return float("nan")
    return float(_hd95(pred, target, voxelspacing=spacing))


def _spokes(X):
    hub = X[:61]
    return np.concatenate([X[61:122] - hub, X[122:183] - hub, X[183:207] - hub[CREST_SKEL_IDX]])


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def spoke_angle(pred, gt):
    return float(np.arccos(np.clip((_unit(_spokes(pred)) * _unit(_spokes(gt))).sum(1), -1, 1)).mean())


def medialness(X):
    up = np.linalg.norm(X[61:122] - X[:61], axis=1)
    dn = np.linalg.norm(X[122:183] - X[:61], axis=1)
    ok = up > 1e-8
    return float(np.mean(dn[ok] / up[ok]))


def point_error(pred, gt):
    return float(np.linalg.norm(pred - gt, axis=1).mean())


def read_vtk(path):
    import vtk
    r = vtk.vtkPolyDataReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def surface_metrics(X, surface):
    import vtk
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(surface)
    nf.AutoOrientNormalsOn()
    nf.Update()
    normals = nf.GetOutput().GetPointData().GetArray("Normals")
    loc = vtk.vtkKdTreePointLocator()
    loc.SetDataSet(surface)
    loc.BuildLocator()
    dist = vtk.vtkImplicitPolyDataDistance()
    dist.SetInput(surface)
    tips, dirs = X[61:207], _unit(_spokes(X))
    ang, d = [], []
    for tip, u in zip(tips, dirs):
        n = _unit(np.array(normals.GetTuple(loc.FindClosestPoint(tip))))
        a = np.arccos(np.clip(u @ n, -1, 1))
        ang.append(min(a, np.pi - a))
        d.append(abs(dist.EvaluateFunction(tip)))
    return float(np.mean(ang)), float(np.mean(d))
