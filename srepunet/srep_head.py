import importlib.util
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.cheb_conv import ChebConv

CREST_SKEL_IDX = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60]


def edge_index_from_adjacency(A):
    A = A.tocoo()
    return torch.as_tensor(np.vstack([A.row, A.col]), dtype=torch.long)


class residualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels, track_running_stats=False),
            )
        else:
            self.skip = None
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x):
        out = self.block(x)
        out += x if self.skip is None else self.skip(x)
        return F.relu(out)


class Encoder(nn.Module):
    def __init__(self, latents=64, size=(64, 64, 64)):
        super().__init__()
        self.residual1 = residualBlock3D(1, 8)
        self.residual2 = residualBlock3D(8, 16)
        self.residual3 = residualBlock3D(16, 32)
        self.residual4 = residualBlock3D(32, 64)
        self.residual5 = residualBlock3D(64, 128)
        self.residual6 = residualBlock3D(128, 128)
        self.maxpool = nn.MaxPool3d(2, stride=2)
        n = 128 * (size[0] // 32) * (size[1] // 32) * (size[2] // 32)
        self.mu = nn.Linear(n, latents)
        self.sigma = nn.Linear(n, latents)

    def forward(self, x):
        for blk in (self.residual1, self.residual2, self.residual3, self.residual4, self.residual5):
            x = self.maxpool(blk(x))
        x = self.residual6(x).flatten(1)
        return self.mu(x), self.sigma(x)


class GConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=6):
        super().__init__()
        self.gConv = ChebConv(in_channels, out_channels, K)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        return F.relu(self.norm(self.gConv(x, edge_index)))


def load_skeletonizer(module_path, beta=0.33, tau=1.0, num_iter=5, simple_point_detection="Boolean"):
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            f"skeletonize.py not found at {module_path}. Clone "
            "https://github.com/martinmenten/skeletonization-for-gradient-based-optimization and pass "
            "--skeletonize_module_path <clone>/skeletonize.py")
    spec = importlib.util.spec_from_file_location("skeletonize_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Skeletonize(probabilistic=True, beta=beta, tau=tau,
                           simple_point_detection=simple_point_detection, num_iter=num_iter)


class SrepHead(nn.Module):
    def __init__(self, adjacency_path, skeletonize_module_path, size=(64, 64, 64), latents=64,
                 use_skeleton_condition=True, deterministic_skeleton_eval=True, attn_heads=4):
        super().__init__()
        A = sp.load_npz(adjacency_path)
        self.register_buffer("edge_index", edge_index_from_adjacency(A), persistent=False)
        self.n_nodes = A.shape[0]
        self.latents = latents
        self.filters = [3, 16, 32, 32, 64, 64]
        self.K = 6
        C = self.filters[5]

        self.encoder = Encoder(latents=latents, size=size)
        self.linear_decoder = nn.Linear(latents, C * self.n_nodes)
        nn.init.normal_(self.linear_decoder.weight, 0, 0.1)


        self.use_skeleton_condition = use_skeleton_condition
        self.deterministic_skeleton_eval = deterministic_skeleton_eval
        if use_skeleton_condition:
            self.skeletonizer = load_skeletonizer(skeletonize_module_path)
            self.skel_tokenizer = nn.Sequential(
                nn.Conv3d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv3d(32, C, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            )
            self.skel_proj = nn.Linear(C, C)
            self.skel_out_ln = nn.LayerNorm(C)
            self.skel_query_pos = nn.Parameter(torch.randn(self.n_nodes, 3) * 0.1)
            self.skel_q_pos_mlp = nn.Sequential(nn.Linear(3, C), nn.ReLU(inplace=True), nn.Linear(C, C))
            self.skel_kv_pos_mlp = nn.Sequential(nn.Linear(3, C), nn.ReLU(inplace=True), nn.Linear(C, C))
            self.skel_wk = nn.Linear(C, C)
            self.skel_wv = nn.Linear(C, C)
            self.skel_ln_q = nn.LayerNorm(C)
            self.skel_ln_kv = nn.LayerNorm(C)
            self.skel_cross_attn = nn.MultiheadAttention(embed_dim=C, num_heads=attn_heads, batch_first=True)
            self.skel_attn_out_proj = nn.Linear(C, C)
            self.skel_attn_out_ln = nn.LayerNorm(C)


        f = self.filters
        self.GC1 = GConv(f[5], f[4], self.K)
        self.GC2 = GConv(f[4], f[3], self.K)
        self.GC3 = GConv(f[3], f[2], self.K)
        self.GC4 = GConv(f[2], f[1], self.K)
        self.GC5 = GConv(f[1], f[1], self.K)
        self.GCout = ChebConv(f[1], f[0], self.K, bias=False)

        self.mu = self.log_var = None

    @staticmethod
    def _unit_grid(d, h, w, device, dtype):
        z = torch.linspace(0.0, 1.0, d, device=device, dtype=dtype)
        y = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1).reshape(1, d * h * w, 3)

    def _skeleton(self, prob):
        with torch.no_grad():
            if self.deterministic_skeleton_eval and not self.training:
                beta0 = self.skeletonizer.beta
                self.skeletonizer.beta = 0.0
                skel = self.skeletonizer(prob.float())
                self.skeletonizer.beta = beta0
            else:
                skel = self.skeletonizer(prob.float())
        return skel

    def forward(self, prob, use_skeleton=True):
        self.mu, self.log_var = self.encoder(prob)
        z = self.mu if not self.training else self.mu + torch.randn_like(self.mu) * torch.exp(0.5 * self.log_var)
        h = F.relu(self.linear_decoder(z)).reshape(z.shape[0], self.n_nodes, self.filters[5])

        if self.use_skeleton_condition and use_skeleton:
            feat = self.skel_tokenizer(self._skeleton(prob).to(h.dtype))
            B, _, d, hh, w = feat.shape
            g = self.skel_out_ln(self.skel_proj(F.adaptive_avg_pool3d(feat, 1).flatten(1)))
            h = h + g.unsqueeze(1).expand(-1, self.n_nodes, -1)

            tokens = feat.flatten(2).transpose(1, 2)
            kv = self.skel_ln_kv(tokens + self.skel_kv_pos_mlp(self._unit_grid(d, hh, w, tokens.device, tokens.dtype)).to(tokens.dtype))
            q = self.skel_ln_q(self.skel_q_pos_mlp(torch.sigmoid(self.skel_query_pos).to(h.dtype)).unsqueeze(0).expand(B, -1, -1))
            att, _ = self.skel_cross_attn(q, self.skel_wk(kv), self.skel_wv(kv), need_weights=False)
            h = h + self.skel_attn_out_ln(self.skel_attn_out_proj(att))

        e = self.edge_index
        for layer in (self.GC1, self.GC2, self.GC3, self.GC4, self.GC5):
            h = layer(h, e)
        return self.GCout(h, e)


class SegSrepNet(nn.Module):
    def __init__(self, seg_model, srep_model):
        super().__init__()
        self.seg_model = seg_model
        self.srep_model = srep_model

    def forward(self, x, use_skeleton=True):
        seg_logits = self.seg_model(x)
        prob = torch.softmax(seg_logits, dim=1)[:, 1:2]
        return seg_logits, self.srep_model(prob, use_skeleton=use_skeleton)
