# Dataset layout

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

Each split file is a JSON list of UID strings. A UID may contain `/`; the same
relative UID is used below both `surfaces/` and `3d_images/`.

Each NPZ file contains:

- `surface`: an `N x 6` float array containing XYZ and normals.
- `sharp_surface`: an `N x 6` float array containing sharp-surface samples.

Images should be RGB or RGBA PNG files. Training randomly selects views 12–23;
validation uses view 13 by default. These values can be changed in the YAML
configuration.
