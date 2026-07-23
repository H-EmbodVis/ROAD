import argparse
import gc

import open_clip
import torch

from evaluation.common import evaluate_point_encoder, extract_image_features, read_manifest
from evaluation.ulip_model import load_ulip_point_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Compute ULIP image-mesh similarity")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--glb-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ulip-checkpoint", required=True)
    parser.add_argument("--openclip-checkpoint", required=True)
    parser.add_argument("--openclip-model", default="ViT-bigG-14")
    parser.add_argument(
        "--view",
        default="eval",
        help="Filename stem shared by <uid>/<view>.png and <uid>/<view>.glb",
    )
    parser.add_argument("--npoints", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--missing-glb", choices=("zero", "skip"), default="skip")
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

    model = load_ulip_point_encoder(args.ulip_checkpoint).to(device).eval()
    evaluate_point_encoder(
        metric="ULIP-I",
        entries=entries,
        image_features=image_features,
        image_missing=image_missing,
        glb_root=args.glb_root,
        view=args.view,
        npoints=args.npoints,
        device=device,
        encode_points=model,
        missing_glb=args.missing_glb,
        output=args.output,
    )


if __name__ == "__main__":
    main()
