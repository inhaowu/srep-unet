from .srep_head import SegSrepNet, SrepHead
from .unet3d import UNet3D


def build_model(args):
    seg = UNet3D(in_channels=1, out_channels=2, fChannel=args.fChannel)
    if args.seg_only:
        return seg
    head = SrepHead(adjacency_path=args.adjacency_path, skeletonize_module_path=args.skeletonize_module_path,
                    size=(args.roi,) * 3, use_skeleton_condition=not args.no_skeleton_prior,
                    deterministic_skeleton_eval=True)
    return SegSrepNet(seg, head)
