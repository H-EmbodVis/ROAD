# ROAD: Reciprocal-Objective Alignment of Discriminative Semantics for 3D Shape Generation

<p align="center">
  Xiao Luo<sup>1</sup> &nbsp;&nbsp;
  Mingyang Du<sup>1</sup> &nbsp;&nbsp;
  Xin Zhou<sup>1</sup> &nbsp;&nbsp;
  Tianrui Feng<sup>1</sup><br>
  Xiwu Chen<sup>2</sup> &nbsp;&nbsp;
  Xiaofan Li<sup>3</sup> &nbsp;&nbsp;
  Jiangning Zhang<sup>3</sup> &nbsp;&nbsp;
  Dingkang Liang<sup>1,*</sup>
</p>

<p align="center">
  <sup>1</sup>Huazhong University of Science and Technology, China<br>
  <sup>2</sup>Megvii, China &nbsp;&nbsp;
  <sup>3</sup>Zhejiang University, China<br>
  <code>{dkliang, tianruifeng, xzhou03}@hust.edu.cn</code><br>
  <sup>*</sup>Corresponding author
</p>

<p align="center">
  <img src="assets/road3d.png" alt="ROAD 3D generation overview" width="100%">
</p>

This repository contains the training and evaluation code for **ROAD**, a
reciprocal-objective alignment approach that transfers discriminative semantic
representations to 3D shape generation through global feature alignment and
token-level matching. It provides the configurations needed to reproduce:

- `configs/baseline.yaml`: rectified-flow training without representation
  alignment.
- `configs/uni3d_repa.yaml`: ROAD training with a frozen Uni3D teacher, global
  feature alignment, and token-level Hungarian matching.

Model weights and full training datasets are not included.

## Qualitative results

The release includes 34 selected qualitative examples. Each example contains
the conditioning image, textured GLB, and all selected high-resolution renders.
The following examples are a fixed random subset of the complete gallery.

<table>
  <tr>
    <td align="center"><img src="assets/showcase/007/preview_thumb.jpg" width="220"><br><sub>007</sub></td>
    <td align="center"><img src="assets/showcase/008/preview_thumb.jpg" width="220"><br><sub>008</sub></td>
    <td align="center"><img src="assets/showcase/020/preview_thumb.jpg" width="220"><br><sub>020</sub></td>
    <td align="center"><img src="assets/showcase/021/preview_thumb.jpg" width="220"><br><sub>021</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/showcase/028/preview_thumb.jpg" width="220"><br><sub>028</sub></td>
    <td align="center"><img src="assets/showcase/029/preview_thumb.jpg" width="220"><br><sub>029</sub></td>
    <td align="center"><img src="assets/showcase/031/preview_thumb.jpg" width="220"><br><sub>031</sub></td>
    <td align="center"><img src="assets/showcase/034/preview_thumb.jpg" width="220"><br><sub>034</sub></td>
  </tr>
</table>

[Browse all 34 examples, compare their conditions, and download the textured GLBs](assets/showcase/README.md).

## 1. Repository structure

```text
.
├── assets/
│   ├── road3d.png
│   └── showcase/
├── configs/
│   ├── baseline.yaml
│   ├── uni3d_repa.yaml
│   └── uni3d_g.json
├── data/
│   ├── 3d_data/
│   └── demo_3d_data/
├── evaluation/
│   ├── evaluate_uni3d.py
│   ├── evaluate_ulip.py
│   ├── common.py
│   ├── ulip_model.py
│   └── uni3d/                # Uni3D-I evaluation subset
├── pretrained/
├── scripts/
│   ├── train_baseline.sh
│   ├── train_uni3d_repa.sh
│   └── evaluate.sh
├── training/
│   └── uni3d/                # frozen Uni3D teacher used during training
├── step1x3d_geometry/
├── tools/create_demo_asset.py
├── environment.yml
├── requirements.txt
├── requirements-eval.txt
└── train.py
```

Run all commands from the repository root.

## 2. Training environment

The training environment uses Python 3.10, PyTorch 2.5.1, and CUDA 11.8.

```bash
conda env create -f environment.yml
conda activate step1x
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

`torch-cluster` and `pointnet2-ops` contain compiled extensions and must match
the installed PyTorch and CUDA versions. Reinstall them after changing PyTorch
or CUDA.

If FlashAttention is unavailable, append
`system.shape_model.use_flash=false` to a training command or set `use_flash`
to `false` in both YAML files.

## 3. Evaluation environment

Evaluation can run in a separate environment with the following tested core
versions:

| Component | Training environment | Evaluation environment |
|---|---:|---:|
| Python | 3.10 | 3.8 |
| PyTorch | 2.5.1+cu118 | 2.1.0+cu118 |
| torchvision | 0.20.1+cu118 | 0.16.0+cu118 |
| timm | 0.9.16 | 1.0.15 |
| pointnet2-ops | 3.0.0 | 3.0.0 |
| open-clip-torch | not required for training | 2.32.0 |
| ftfy | not required for training | 6.2.3 |

If an existing `driveuni3d` environment provides the evaluation column,
activate it directly:

```bash
conda activate driveuni3d
```

To run evaluation in the `step1x` environment instead, install only the
additional packages:

```bash
conda activate step1x
pip install -r requirements-eval.txt
```

Do not replace the training environment's PyTorch, torchvision, or timm merely
to add evaluation. `open-clip-torch` accepts the already installed timm
version for inference.

On its first evaluation run, `pointnet2-ops` may compile a local CUDA extension.
The launcher places this build below the selected output directory. A working
CUDA toolkit, C++ compiler, and Ninja are required for that one-time build.

## 4. Pretrained weights

Prepare the following files:

```text
pretrained/
├── vae/
│   └── diffusion_pytorch_model.safetensors
├── visual_encoder/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── uni3d/
│   └── model.pt
└── evaluation/
    ├── uni3d_openclip.bin
    ├── ulip_openclip.bin
    └── ulip2_pointbert.pt
```

The baseline requires the Step1X-3D VAE and DINOv2 visual encoder. Uni3D-REPA
and Uni3D-I additionally use `pretrained/uni3d/model.pt`. Uni3D-I uses the
EVA02-E-14-plus OpenCLIP checkpoint, while ULIP-I uses the ViT-bigG-14 OpenCLIP
checkpoint and the ULIP-2 PointBERT checkpoint.

Download weights from the corresponding official releases and review their
licenses before use or redistribution. See `pretrained/README.md` for the same
layout.

## 5. Training dataset

### Included synthetic demo

The repository includes one small synthetic cube at `data/demo_3d_data`. It
contains a point surface, sharp-edge samples, 12 matching rendered views, and
train/validation/test manifests. It is intended for checking the complete data
and training path, not for model quality or benchmarking.

The committed files can be regenerated deterministically:

```bash
python tools/create_demo_asset.py
```

The demo asset is released under CC0 1.0. Its layout and contents are described
in `data/demo_3d_data/README.md`.

### Full training data

The default dataset root is `data/3d_data`:

```text
data/3d_data/
├── train.json
├── val.json
├── test.json
├── surfaces/
│   └── <uid>.npz
└── 3d_images/
    └── <uid>/
        ├── 012.png
        ├── 013.png
        └── ...
```

Each split file is a JSON list of UID strings:

```json
[
  "example_uid_0001",
  "example_uid_0002"
]
```

Each `surfaces/<uid>.npz` file must contain:

- `surface`: an `N x 6` floating-point array of XYZ coordinates and normals.
- `sharp_surface`: an `N x 6` floating-point array of sharp-surface samples.

Images may be RGB or RGBA PNG files. Training randomly selects one view from
indices 12–23, and validation uses view 13 by default.

To use another dataset location, pass an OmegaConf override:

```bash
DATA_ROOT=/absolute/path/to/dataset
GPU_IDS=0 bash scripts/train_baseline.sh data.root_dir="${DATA_ROOT}"
```

## 6. One-step training check

Run one optimizer step before a full training run. These commands load the
checkpoints and dataset, perform forward and backward propagation, update the
optimizer, and write logs.

Baseline:

```bash
GPU_IDS=0 bash scripts/train_baseline.sh \
  exp_root_dir=outputs/smoke_baseline \
  use_timestamp=false tag=smoke \
  data.root_dir=data/demo_3d_data \
  data.batch_size=1 data.num_workers=0 data.n_samples=4096 \
  trainer.max_steps=1 trainer.limit_val_batches=0 \
  trainer.num_sanity_val_steps=0 trainer.log_every_n_steps=1 \
  system.skip_validation=true
```

Uni3D-REPA, including token-level GPU Hungarian matching:

```bash
GPU_IDS=1 bash scripts/train_uni3d_repa.sh \
  exp_root_dir=outputs/smoke_repa \
  use_timestamp=false tag=smoke \
  data.root_dir=data/demo_3d_data \
  data.batch_size=1 data.num_workers=0 data.n_samples=4096 \
  data.align_points=4096 \
  trainer.max_steps=1 trainer.limit_val_batches=0 \
  trainer.num_sanity_val_steps=0 trainer.log_every_n_steps=1 \
  system.skip_validation=true system.alignment_start_epoch=0
```

`system.alignment_start_epoch=0` is only used to exercise token matching in
this check. Full Uni3D-REPA training uses the configured value of 3.

A successful check ends with:

```text
Trainer.fit stopped: max_steps=1 reached
```

## 7. Baseline training

Single GPU:

```bash
GPU_IDS=0 bash scripts/train_baseline.sh
```

Multiple GPUs on one node:

```bash
GPU_IDS=0,1,2,3 bash scripts/train_baseline.sh
```

The default configuration uses batch size 16 per process, 32,768 surface
points, BF16 mixed precision, DeepSpeed ZeRO stage 2, and 150 epochs.

## 8. Uni3D-REPA training

Single GPU:

```bash
GPU_IDS=0 bash scripts/train_uni3d_repa.sh
```

Multiple GPUs on one node:

```bash
GPU_IDS=0,1,2,3 bash scripts/train_uni3d_repa.sh
```

The default configuration uses batch size 16 per process, 32,768 surface
points, 10,000 Uni3D alignment points, global alignment weight 0.5,
token-matching weight 0.1, token matching from epoch 3, BF16 mixed precision,
DeepSpeed ZeRO stage 2, and 600 epochs.

The frozen training teacher is loaded from `training/uni3d/` with
`configs/uni3d_g.json`. Point groups are selected with farthest-point sampling.

The GPU matcher compiles a local CUDA extension the first time it is imported.
To use the slower SciPy implementation, append:

```bash
system.matcher=cpu
```

Any OmegaConf option can be appended to a launcher command:

```bash
GPU_IDS=0,1 bash scripts/train_uni3d_repa.sh \
  data.batch_size=4 \
  data.num_workers=4 \
  system.matcher=cpu \
  trainer.max_epochs=10
```

`GPU_IDS` always contains physical GPU IDs. The launchers export
`CUDA_VISIBLE_DEVICES` before importing PyTorch.

## 9. Resume training

Pass a Lightning or DeepSpeed checkpoint through `resume`:

```bash
GPU_IDS=0,1 bash scripts/train_uni3d_repa.sh \
  resume=outputs/uni3d_repa/step1x3d/<run_name>/checkpoints/last.ckpt
```

Use the same number of processes and a compatible model configuration when
resuming a distributed checkpoint.

## 10. Evaluation data

Prepare one manifest and two roots with matching relative UIDs:

```text
evaluation_data/
├── manifest.json
├── images/
│   └── <uid>/
│       └── 013.png
└── meshes/
    └── <uid>/
        └── 013.glb
```

Nested UIDs are supported. For example:

```json
[
  "category_a/example_uid_0001",
  "category_b/example_uid_0002"
]
```

corresponds to
`images/category_a/example_uid_0001/013.png` and
`meshes/category_a/example_uid_0001/013.glb`.

Images are composited over a white background. Mesh geometry is loaded with
Trimesh, sampled to 8,192 vertices with farthest-point sampling, and assigned
constant unit RGB features, matching the original metric implementation.

## 11. Two-GPU evaluation

Activate the evaluation environment and launch both metrics. Uni3D-I uses the
first GPU and ULIP-I uses the second:

```bash
conda activate driveuni3d
GPU_IDS=0,1 bash scripts/evaluate.sh \
  evaluation_data/manifest.json \
  evaluation_data/images \
  evaluation_data/meshes \
  outputs/evaluation
```

The command writes:

```text
outputs/evaluation/
├── uni3d_i.json
├── ulip_i.json
└── torch_extensions/
```

Each JSON file contains the mean score, sample counts, missing inputs, failures,
and per-sample scores. To preserve the source evaluation behavior, a missing
GLB contributes zero to Uni3D-I and is skipped by ULIP-I. Missing images and
GLB processing errors are reported explicitly.

Run one metric directly when only one GPU is available:

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.evaluate_uni3d \
  --manifest evaluation_data/manifest.json \
  --image-root evaluation_data/images \
  --glb-root evaluation_data/meshes \
  --openclip-checkpoint pretrained/evaluation/uni3d_openclip.bin \
  --output outputs/evaluation/uni3d_i.json
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluation.evaluate_ulip \
  --manifest evaluation_data/manifest.json \
  --image-root evaluation_data/images \
  --glb-root evaluation_data/meshes \
  --openclip-checkpoint pretrained/evaluation/ulip_openclip.bin \
  --ulip-checkpoint pretrained/evaluation/ulip2_pointbert.pt \
  --output outputs/evaluation/ulip_i.json
```

Use `--view`, `--npoints`, or `--missing-glb` only when intentionally changing
the default protocol. `--uni3d-checkpoint` can override the checkpoint path in
`evaluation/uni3d/config.json`.

## 12. Outputs and logs

Training runs write below `exp_root_dir`:

```text
<exp_root_dir>/
└── step1x3d/
    └── <run_name>/
        ├── checkpoints/
        ├── configs/
        ├── csv/
        ├── tensorboard/
        └── artifacts/
```

Inspect logs with:

```bash
tensorboard --logdir outputs
```

## 13. License and attribution

The Step1X-3D-derived source is distributed under Apache License 2.0. The
training and evaluation Uni3D subsets under `training/uni3d/` and
`evaluation/uni3d/` retain the upstream MIT license.
The minimal ULIP-2 PointBERT evaluation implementation retains the upstream
BSD 3-Clause license.
See `LICENSE`, `NOTICE`, `MODIFICATIONS.md`, `training/uni3d/LICENSE`,
`evaluation/uni3d/LICENSE`, and `evaluation/LICENSE-ULIP`.

Model weights and datasets are distributed separately and may use different
licenses.

Please cite and acknowledge the original
[Step1X-3D](https://github.com/stepfun-ai/Step1X-3D),
[Uni3D](https://github.com/baaivision/Uni3D), and
[ULIP](https://github.com/salesforce/ULIP) projects when using this code.
