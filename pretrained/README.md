# Pretrained weights

Place pretrained files in the following layout:

```text
pretrained/
├── vae/
│   └── diffusion_pytorch_model.safetensors
├── visual_encoder/
│   ├── config.json
│   └── ... Hugging Face DINOv2 model files
├── uni3d/
│   └── model.pt
└── evaluation/
    ├── uni3d_openclip.bin
    ├── ulip_openclip.bin
    └── ulip2_pointbert.pt
```

The source archive intentionally does not contain model weights. Download the
Step1X-3D VAE, DINOv2 visual encoder, Uni3D, EVA02-E-14-plus OpenCLIP,
ViT-bigG-14 OpenCLIP, and ULIP-2 PointBERT checkpoints from their corresponding
official releases and comply with their respective licenses.
