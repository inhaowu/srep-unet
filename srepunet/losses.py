import torch
import torch.nn.functional as F

from .srep_head import CREST_SKEL_IDX


def seg_loss(logits, label, smooth=1e-5):
    ce = F.cross_entropy(logits, label[:, 0].long())
    probs = torch.softmax(logits, dim=1)
    target = torch.cat([1.0 - label.float(), label.float()], dim=1)
    inter = (probs * target).sum(dim=(2, 3, 4))
    denom = probs.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    return ce + 1.0 - ((2.0 * inter + smooth) / (denom + smooth)).mean()


def kl_loss(mu, log_var):
    return -0.5 * torch.mean(torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim=1))


def spoke_vectors(X):
    hub = X[:, :61]
    return X[:, 61:122] - hub, X[:, 122:183] - hub, X[:, 183:207] - hub[:, CREST_SKEL_IDX]


def spoke_loss(pred, target):
    loss = 0.0
    for p, t in zip(spoke_vectors(pred), spoke_vectors(target)):
        loss = loss + (1.0 - (F.normalize(p, dim=2, eps=1e-8) * F.normalize(t, dim=2, eps=1e-8)).sum(2)).mean()
    return loss


def _to_grid(coords, size):
    D, H, W = size
    scale = coords.new_tensor([W / max(W - 1, 1), H / max(H - 1, 1), D / max(D - 1, 1)])
    return torch.clamp(coords * scale * 2.0 - 1.0, -1.0, 1.0)


def orth_loss(pred, normals_volume, size=(64, 64, 64)):
    D, H, W = size
    c = torch.clamp(pred, 0.0, 1.0)
    phys = c * c.new_tensor([W, H, D])
    dirs = F.normalize(torch.cat(spoke_vectors(phys), dim=1), dim=2, eps=1e-8)
    grid = _to_grid(c[:, 61:207], size)[:, :, None, None, :]
    n = F.grid_sample(normals_volume, grid, mode="bilinear", padding_mode="border", align_corners=True)
    n = F.normalize(n[..., 0, 0].permute(0, 2, 1), dim=2, eps=1e-8)
    return (1.0 - (dirs * n).sum(2).clamp(-1.0, 1.0)).mean()
