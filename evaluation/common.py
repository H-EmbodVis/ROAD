import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from pointnet2_ops import pointnet2_utils
from tqdm import tqdm


def read_manifest(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
        raise ValueError("The evaluation manifest must be a JSON list of UID strings")
    return entries


def sample_path(root, uid, view, suffix):
    relative = Path(uid)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid UID in manifest: {uid!r}")
    return Path(root) / relative / f"{view}.{suffix}"


def load_rgb(path):
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def extract_image_features(entries, image_root, view, model, preprocess, device):
    features = {}
    missing = []
    for uid in tqdm(entries, desc="Images"):
        path = sample_path(image_root, uid, view, "png")
        if not path.is_file():
            missing.append(uid)
            continue
        image = preprocess(load_rgb(path)).unsqueeze(0).to(device)
        with torch.inference_mode():
            feature = model.encode_image(image)
        features[uid] = feature.detach().float().cpu()
    return features, missing


def load_vertices(path):
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Trimesh):
        arrays = [np.asarray(loaded.vertices)]
    elif isinstance(loaded, trimesh.Scene):
        arrays = [
            np.asarray(geometry.vertices)
            for geometry in loaded.geometry.values()
            if hasattr(geometry, "vertices") and len(geometry.vertices)
        ]
    else:
        arrays = []
    if not arrays:
        raise ValueError(f"No mesh vertices found in {path}")
    vertices = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(vertices).all():
        raise ValueError(f"Non-finite mesh vertices found in {path}")
    return vertices


def farthest_point_sample(vertices, count, device):
    points = torch.from_numpy(vertices).to(device=device, dtype=torch.float32)
    if points.shape[0] < count:
        repeats = (count + points.shape[0] - 1) // points.shape[0]
        points = points.repeat(repeats, 1)
    points = points.unsqueeze(0).contiguous()
    indices = pointnet2_utils.furthest_point_sample(points, count)
    return pointnet2_utils.gather_operation(
        points.transpose(1, 2).contiguous(), indices
    ).transpose(1, 2).contiguous()


def cosine_score(point_feature, image_feature):
    point_feature = F.normalize(point_feature.float(), dim=-1)
    image_feature = F.normalize(image_feature.to(point_feature.device).float(), dim=-1)
    return F.cosine_similarity(point_feature, image_feature).mean().item()


def evaluate_point_encoder(
    *,
    metric,
    entries,
    image_features,
    image_missing,
    glb_root,
    view,
    npoints,
    device,
    encode_points,
    missing_glb,
    output,
):
    scores = {}
    missing_meshes = []
    failed = {}
    zero_count = 0

    for uid in tqdm(entries, desc=metric):
        if uid not in image_features:
            continue
        path = sample_path(glb_root, uid, view, "glb")
        if not path.is_file():
            missing_meshes.append(uid)
            if missing_glb == "zero":
                scores[uid] = 0.0
                zero_count += 1
            continue
        try:
            xyz = farthest_point_sample(load_vertices(path), npoints, device)
            points = torch.cat([xyz, torch.ones_like(xyz)], dim=-1)
            with torch.inference_mode():
                point_feature = encode_points(points)
            scores[uid] = cosine_score(point_feature, image_features[uid])
        except Exception as error:
            failed[uid] = f"{type(error).__name__}: {error}"

    average = sum(scores.values()) / len(scores) if scores else 0.0
    result = {
        "metric": metric,
        "average": average,
        "scored_samples": len(scores),
        "successful_meshes": len(scores) - zero_count,
        "missing_images": image_missing,
        "missing_meshes": missing_meshes,
        "failed_samples": failed,
        "missing_mesh_policy": missing_glb,
        "scores": scores,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"{metric}: {average:.8f} ({len(scores)} scored samples)")
    print(f"Saved: {output}")
    if failed and len(scores) == zero_count:
        raise RuntimeError(f"All mesh evaluations failed; see {output}")
    return result
