# Joint Segmentation and Graph-Based Skeletal Representation Learning with Geometric Priors

[![Paper - MICCAI 2026 Oral](https://img.shields.io/badge/Paper-MICCAI%202026%20Oral-0969DA?logo=readthedocs&logoColor=white)](https://papers.miccai.org/miccai-2026/0521-Paper4240.html)
[![Slides - PowerPoint](https://img.shields.io/badge/Slides-PowerPoint-B7472A?logo=slides&logoColor=white)](./s-rep-unet.pptx?raw=true)

Official implementation of our MICCAI 2026 paper. Given a 3D image, the network jointly predicts the segmentation
and the skeletal representation (s-rep) of the structure: a 3D U-Net produces the foreground probability map, and
an s-rep head (variational encoder, skeletal prior, graph-convolutional decoder) maps it to the 207 points of the
s-rep (61 skeletal points and the tips of 146 spokes).

## Installation

```bash
conda create -n srepunet python=3.9 -y && conda activate srepunet
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

git clone https://github.com/martinmenten/skeletonization-for-gradient-based-optimization \
    third_party/skeletonization-for-gradient-based-optimization
```

Generating the reference s-reps additionally requires
[SlicerSALT](https://salt.slicer.org) (6.0.0, Linux) and [Deformetrica](https://www.deformetrica.org)
(installed by `requirements.txt`).

## Data

Download **Task04_Hippocampus** from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) and extract it.

### 1. Preprocessing

```bash
python preprocessing/prepare_msd.py --msd_dir /path/to/Task04_Hippocampus --out_dir data/msd
```

Each training image is zero-padded to 64³. MSD label 1 (anterior hippocampus) and label 2 (posterior hippocampus)
give two cases per image, `<id>_anterior` and `<id>_posterior`, stored in `data/msd/npz/` (520 cases).

### 2. Reference s-reps

```bash
python preprocessing/srep_pipeline.py all --labels data/msd/npz --out data/msd/srep_work \
    --slicersalt /path/to/SlicerSALT-6.0.0-linux-amd64 --deformetrica $(which deformetrica) \
    --gpus 0,1,2,3 --evo-workers 8
```

Stages: `prep` (label to NRRD) → `spharm` (SPHARM-PDM surface mesh) → `qc` (mesh-to-label alignment) → `evo`
(evolutionary s-rep fitting) → `sync` (consistent s-rep grid labelling). Each stage can also be run on its own
(`python preprocessing/srep_pipeline.py <stage> ...`), and finished cases are skipped when the pipeline is resumed.
Outputs:

```
data/msd/srep_work/srep/<case>_srep.npy       reference s-reps (207 x 3, voxel coordinates)
data/msd/srep_work/mesh/<case>_SPHARM.vtk     surface meshes
```

### 3. Splits

The five-fold splits are in `data/splits/fold{0..4}/{train,valid,test}.txt`.

## Training

```bash
python train.py --data_dir data/msd/npz --srep_dir data/msd/srep_work/srep --fold 0 --logdir runs/msd_fold0
```

Default settings: AdamW (lr 1e-4, weight decay 1e-5), 80 epochs, batch size 4, s-rep loss weight 0.1,
L_spoke weight 0.1, L_orth weight 0.1, KL weight 1e-4. Options for ablations: `--no_skeleton_prior`,
`--spoke_weight 0`, `--orth_weight 0`, `--seg_only`.

Checkpoints saved in the log directory: `model_best_angle.pt`, `model_best_dice.pt`, `model_last.pt`.

## Evaluation

```bash
python test.py --data_dir data/msd/npz --srep_dir data/msd/srep_work/srep --fold 0 \
    --checkpoint runs/msd_fold0/model_best_angle.pt --out runs/msd_fold0/test \
    --mesh_dir data/msd/srep_work/mesh
```

Outputs `metrics.csv` (Dice, HD95, RMSE, spoke angle, medialness, orthogonality, tip-to-surface distance) and the
predicted s-reps in `srep_pred/`.

All five folds: `bash scripts/run_msd_5fold.sh [GPU]`.

## Citation

```bibtex
@inproceedings{wu2026joint,
  title={Joint Segmentation and Graph-Based Skeletal Representation Learning with Geometric Priors},
  author={Wu, Yinhao and Li, Haiqing and Ma, Hehuan and Huang, Junzhou},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={284--294},
  year={2026},
  organization={Springer}
}
```

## Acknowledgements

This repository builds on the following open-source projects and datasets. We thank their authors for making them
available.

- [HybridGNet](https://github.com/ngaggion/HybridGNet) (Gaggion et al., IEEE TMI 2022): graph-convolutional decoder
  and residual encoder blocks.
- Fitting Skeletal Models via Graph-Based Learning (Gaggion, Ferrante, Paniagua, Vicory; ISBI 2024): s-rep graph
  adjacency (`adjacency.npz`).
- [A Skeletonization Algorithm for Gradient-Based Optimization](https://github.com/martinmenten/skeletonization-for-gradient-based-optimization)
  (Menten et al., ICCV 2023): skeletonization module.
- [MONAI research-contributions](https://github.com/Project-MONAI/research-contributions): learning-rate scheduler.
- [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet): 3D U-Net building blocks.
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric): Chebyshev graph convolutions.
- [SlicerSALT](https://salt.slicer.org) (Vicory et al., 2018): SPHARM-PDM (Styner et al., 2006) and evolutionary
  s-rep fitting (Pizer et al., JMIV 2025).
- [Deformetrica](https://www.deformetrica.org) (Bône et al., 2018).
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (Antonelli et al., Nature Communications 2022).

## License

GPL-3.0. `srepunet/lr_scheduler.py` is from MONAI and is licensed under Apache-2.0.
