import numpy as np


M72 = np.array([[3 * j, 3 * j + 1, 3 * j + 2] for j in range(13)] +
               [[3 * (24 - j), 39 + 2 * (j - 13), 40 + 2 * (j - 13)] for j in range(13, 24)])


def _hub_perm(line_map):
    p = -np.ones(61, dtype=int)
    for j in range(24):
        for s in range(3):
            p[M72[j, s]] = M72[line_map(j), s]
    return p


def _srep_perm(line_map, swap):
    h = _hub_perm(line_map)
    top, bot = 61 + h, 122 + h
    if swap:
        top, bot = bot, top
    return np.concatenate([h, top, bot, 183 + np.array([line_map(k) for k in range(24)])])


ELEMS = {"id": _srep_perm(lambda j: j, 0), "g3": _srep_perm(lambda j: (j + 12) % 24, 0),
         "g1+swap": _srep_perm(lambda j: (24 - j) % 24, 1), "g2+swap": _srep_perm(lambda j: (12 - j) % 24, 1)}
CHI = {"id": (1, 1), "g1+swap": (-1, 1), "g2+swap": (1, -1), "g3": (-1, -1)}
FROM_CHI = {v: k for k, v in CHI.items()}


def _procrustes_residual(A, B):
    A0, B0 = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(np.einsum("npi,npj->nij", A0, B0))
    d = np.sign(np.linalg.det(U @ Vt))
    S[:, 2] *= d
    U[:, :, 2] *= d[:, None]
    s = S.sum(1) / (A0 ** 2).sum((1, 2))
    return np.linalg.norm(s[:, None, None] * A0 @ (U @ Vt) - B0, axis=2).mean(1)


def synchronise(Xs):
    Xs = np.asarray(Xs, dtype=np.float64)
    N = len(Xs)
    names = list(ELEMS)
    res = np.zeros((len(names), N, N))
    ii, jj = np.triu_indices(N, 1)
    for k, n in enumerate(names):
        Xg = Xs[:, ELEMS[n]]
        r = np.zeros(len(ii))
        for s in range(0, len(ii), 20000):
            r[s:s + 20000] = _procrustes_residual(Xg[ii[s:s + 20000]], Xs[jj[s:s + 20000]])
        res[k, ii, jj] = r
        res[k, jj, ii] = r
    best = res.argmin(0)
    srt = np.sort(res, 0)
    w = (srt[1] - srt[0]) / np.maximum(srt[0], 1e-9)
    np.fill_diagonal(w, 0)
    chis = []
    for c in (0, 1):
        H = np.vectorize(lambda b: CHI[names[b]][c])(best) * w
        np.fill_diagonal(H, w.sum(1).max())
        v = np.linalg.eigh(H)[1][:, -1]
        chis.append(np.where(v >= 0, 1, -1))
    g = [FROM_CHI[(int(a), int(b))] for a, b in zip(*chis)]
    cnt = {n: g.count(n) for n in names}
    ga, gb = CHI[max(cnt, key=cnt.get)]
    return [FROM_CHI[(CHI[x][0] * ga, CHI[x][1] * gb)] for x in g]


def apply_elems(Xs, g):
    return np.stack([X[ELEMS[e]] for X, e in zip(Xs, g)])
