import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import open_clip
import torch

from evaluation.common import evaluate_point_encoder, extract_image_features, read_manifest
from evaluation.uni3d.models.uni3d import create_uni3d


def load_uni3d_config(path):
    defaults = {
        "pc_model": "eva_giant_patch14_560",
        "pretrained_pc": "",
        "drop_path_rate": 0.0,
        "patch_dropout": 0.0,
        "pc_feat_dim": 1408,
        "embed_dim": 1024,
        "num_group": 512,
        "group_size": 64,
        "pc_encoder_dim": 512,
        "ckpt_path": "pretrained/uni3d/model.pt",
    }
    with Path(path).open("r", encoding="utf-8") as handle:
        defaults.update(json.load(handle))
    return SimpleNamespace(**defaults)


def point_encoder_state_dict(checkpoint):
    state = checkpoint.get("module", checkpoint.get("state_dict", checkpoint))
    cleaned = {}
    for name, value in state.items():
        if name == "logit_scale":
            continue
        if name.startswith("module."):
            name = name[len("module.") :]
        if name.startswith("point_encoder."):
            name = name[len("point_encoder.") :]
        cleaned[name] = value
    return cleaned


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Uni3D image-mesh similarity")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--glb-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--uni3d-config", default="evaluation/uni3d/config.json")
    parser.add_argument("--uni3d-checkpoint", default=None)
    parser.add_argument("--openclip-checkpoint", required=True)
    parser.add_argument("--openclip-model", default="EVA02-E-14-plus")
    parser.add_argument(
        "--view",
        default="eval",
        help="Filename stem shared by <uid>/<view>.png and <uid>/<view>.glb",
    )
    parser.add_argument("--npoints", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--missing-glb", choices=("zero", "skip"), default="zero")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    entries = read_manifest(args.manifest)

    image_model, _, preprocess = open_clip.create_model_and_transforms(
        args.openclip_model, pretrained=args.openclip_checkpoint
    )
    image_model = image_model.to(device).eval()
    image_features, image_missing = extract_image_features(
        entries, args.image_root, args.view, image_model, preprocess, device
    )
    del image_model, preprocess
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    config = load_uni3d_config(args.uni3d_config)
    if args.uni3d_checkpoint is not None:
        config.ckpt_path = args.uni3d_checkpoint
    model = create_uni3d(config)
    checkpoint = torch.load(config.ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(point_encoder_state_dict(checkpoint), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Uni3D checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model = model.to(device).eval()

    def encode(points):
        xyz = points[..., :3].contiguous()
        colors = points[..., 3:].contiguous()
        return model(xyz, colors)[:, 0]

    evaluate_point_encoder(
        metric="Uni3D-I",
        entries=entries,
        image_features=image_features,
        image_missing=image_missing,
        glb_root=args.glb_root,
        view=args.view,
        npoints=args.npoints,
        device=device,
        encode_points=encode,
        missing_glb=args.missing_glb,
        output=args.output,
    )


if __name__ == "__main__":
    main()
