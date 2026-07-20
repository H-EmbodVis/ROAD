# Modifications from the research workspace

This public release keeps the behavior of the baseline and Uni3D-REPA losses
while removing unrelated experimental branches. The main changes are:

- Replaced machine-specific paths with relative, configurable paths.
- Renamed the two public configurations because the experiments do not enable
  LoRA adapters.
- Removed VGGT, ReCon, Utonia, DINO alignment, SRA, token-cache, optimal-
  transport variants, texture generation, web demos, and cluster utilities.
- Reduced the rectified-flow system to diffusion loss, Uni3D global cosine
  alignment, and Uni3D token-level Hungarian matching.
- Reduced Uni3D to two purpose-specific point-encoder subsets: the frozen REPA
  teacher under `training/uni3d` and Uni3D-I inference under
  `evaluation/uni3d`. Both use farthest-point sampling, and training no longer
  imports evaluation code.
- Replaced recursive bad-sample retries with an explicit data-loading error.
- Corrected horizontal image flipping to operate on the width dimension.
- Removed private debug visualization, checkpoint, and validation paths.
- Made checkpoint, data, output, and pretrained-model locations configurable.
- Replaced machine-specific evaluation launchers with portable Uni3D-I and
  ULIP-I command-line entry points and a two-GPU launcher.
- Reduced ULIP evaluation to the checkpoint-compatible PointBERT inference
  modules used by ULIP-I.
- Added a deterministic, CC0 synthetic cube asset for exercising the complete
  baseline and Uni3D-REPA training data paths.
- Added 34 numbered qualitative examples with conditioning images, textured
  GLBs, selected renders, a gallery, and a machine-readable manifest.
- Added the ROAD paper title, authors, affiliations, overview image, and a
  fixed random subset of qualitative results to the repository front page.

The original experiment hyperparameters, alignment weights, alignment start
epoch, token count, model widths, and training epoch counts are preserved in
`configs/baseline.yaml` and `configs/uni3d_repa.yaml`.
