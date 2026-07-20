"""Generate the small synthetic cube asset shipped with this repository."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


UID = "demo/cube"
VIEWS = range(12, 24)


def cube_surfaces(rng, points_per_face=2048):
    samples = []
    for axis in range(3):
        other_axes = [index for index in range(3) if index != axis]
        for sign in (-1.0, 1.0):
            xyz = rng.uniform(-0.5, 0.5, size=(points_per_face, 3)).astype(np.float32)
            xyz[:, axis] = sign * 0.5
            normals = np.zeros_like(xyz)
            normals[:, axis] = sign
            samples.append(np.concatenate([xyz, normals], axis=1))
    return np.concatenate(samples, axis=0)


def cube_edges(rng, points_per_edge=512):
    samples = []
    for varying_axis in range(3):
        fixed_axes = [index for index in range(3) if index != varying_axis]
        for first_sign in (-1.0, 1.0):
            for second_sign in (-1.0, 1.0):
                xyz = np.zeros((points_per_edge, 3), dtype=np.float32)
                xyz[:, varying_axis] = rng.uniform(-0.5, 0.5, size=points_per_edge)
                xyz[:, fixed_axes[0]] = first_sign * 0.5
                xyz[:, fixed_axes[1]] = second_sign * 0.5
                normals = np.zeros_like(xyz)
                normals[:, fixed_axes[0]] = first_sign
                normals[:, fixed_axes[1]] = second_sign
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                samples.append(np.concatenate([xyz, normals], axis=1))
    return np.concatenate(samples, axis=0)


def rotation_y(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float32,
    )


def rotation_x(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float32,
    )


def render_view(view_index, path):
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (3, 2, 6, 7),
        (0, 3, 7, 4),
        (1, 5, 6, 2),
    ]

    angle = 2.0 * np.pi * (view_index - min(VIEWS)) / len(VIEWS)
    rotation = rotation_x(np.deg2rad(-22.0)) @ rotation_y(angle)
    transformed = vertices @ rotation.T
    distance = 3.0
    scale = 630.0
    projected = np.empty((len(vertices), 2), dtype=np.float32)
    projected[:, 0] = 512.0 + transformed[:, 0] * scale / (distance - transformed[:, 2])
    projected[:, 1] = 512.0 - transformed[:, 1] * scale / (distance - transformed[:, 2])

    canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    light = np.array([-0.35, 0.55, 0.76], dtype=np.float32)
    light /= np.linalg.norm(light)
    face_order = sorted(faces, key=lambda face: transformed[list(face), 2].mean())
    for face in face_order:
        face_vertices = transformed[list(face)]
        normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
        normal /= max(np.linalg.norm(normal), 1.0e-8)
        brightness = 0.48 + 0.42 * abs(float(np.dot(normal, light)))
        base = np.array([70, 145, 210], dtype=np.float32)
        color = tuple(np.clip(base * brightness, 0, 255).astype(np.uint8)) + (255,)
        polygon = [tuple(projected[index]) for index in face]
        draw.polygon(polygon, fill=color, outline=(24, 47, 70, 255), width=5)

    canvas = canvas.resize((512, 512), Image.Resampling.LANCZOS)
    canvas.save(path, optimize=True)


def generate(output):
    output = Path(output)
    surface_path = output / "surfaces" / f"{UID}.npz"
    image_dir = output / "3d_images" / UID
    surface_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260718)
    np.savez_compressed(
        surface_path,
        surface=cube_surfaces(rng),
        sharp_surface=cube_edges(rng),
    )
    for view_index in VIEWS:
        render_view(view_index, image_dir / f"{view_index:03d}.png")
    for split in ("train", "val", "test"):
        with (output / f"{split}.json").open("w", encoding="utf-8") as handle:
            json.dump([UID], handle, indent=2)
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo_3d_data")
    generate(parser.parse_args().output)


if __name__ == "__main__":
    main()
