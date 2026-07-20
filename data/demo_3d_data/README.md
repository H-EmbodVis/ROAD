# Synthetic training demo

This directory contains one synthetic cube sample for checking the complete
training data path without downloading a dataset.

```text
demo_3d_data/
├── train.json
├── val.json
├── test.json
├── surfaces/demo/cube.npz
└── 3d_images/demo/cube/012.png ... 023.png
```

The NPZ file contains `surface` and `sharp_surface` arrays in the same `N x 6`
XYZ-plus-normal format expected by the training loader. The PNG files are 12
matching synthetic views with transparent backgrounds.

Regenerate the asset from the repository root with:

```bash
python tools/create_demo_asset.py
```

The generated cube geometry and renders are dedicated to the public domain
under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
