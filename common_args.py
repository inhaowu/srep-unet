import os

HERE = os.path.dirname(os.path.abspath(__file__))


def add_common_args(p):
    p.add_argument("--data_dir", required=True, help="directory of <case>.npz from preprocessing/prepare_msd.py")
    p.add_argument("--srep_dir", default=None, help="directory of <case><srep_suffix> reference s-reps")
    p.add_argument("--srep_suffix", default="_srep.npy")
    p.add_argument("--splits_dir", default=os.path.join(HERE, "data", "splits"))
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--roi", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--fChannel", type=int, default=32, help="U-Net base channels")
    p.add_argument("--adjacency_path", default=os.path.join(HERE, "adjacency.npz"))
    p.add_argument("--skeletonize_module_path",
                   default=os.path.join(HERE, "third_party", "skeletonization-for-gradient-based-optimization",
                                        "skeletonize.py"))
    p.add_argument("--no_skeleton_prior", action="store_true", help="ablation: no skeletal prior (SP)")
    p.add_argument("--seg_only", action="store_true", help="segmentation-only U-Net")
    return p
