import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm


def _ensure_repo_root_on_path() -> None:
    # Make "trellis" importable when running this script directly.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_path()

from trellis.representations.mesh.cube2mesh import MeshExtractResult  # noqa: E402
from trellis.utils.render_utils import (  # noqa: E402
    render_frames,
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)


def _normalize_mesh_in_place(
    vertices: np.ndarray,
    r: float,
    fov_deg: float,
    near: float,
    margin: float,
) -> np.ndarray:
    if vertices.size == 0:
        return vertices

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    vertices = vertices - center[None, :]

    radius = float(np.linalg.norm(vertices, axis=1).max(initial=0.0))
    if radius <= 0.0:
        return vertices

    half_fov = math.radians(float(fov_deg)) * 0.5
    max_radius_by_fov = float(r) * math.tan(half_fov)
    max_radius_by_near = float(r) - float(near)
    target_radius = max(1e-6, min(max_radius_by_fov, max_radius_by_near) * float(margin))
    scale = target_radius / radius
    return vertices * scale


def _build_fixed_cameras(pitch_deg: float, yaws_deg: List[float], r: float, fov_deg: float):
    pitch_rad = math.radians(float(pitch_deg))
    yaws_rad = [math.radians(float(y)) for y in yaws_deg]
    pitchs_rad = [pitch_rad for _ in yaws_rad]
    return yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws_rad, pitchs_rad, r, fov_deg)


def _save_normal_images(
    normals: List[np.ndarray],
    model_name: str,
    yaws_deg: List[float],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for idx, (img, yaw) in enumerate(zip(normals, yaws_deg)):
        out_name = f"{model_name}_yaw{int(yaw)}_view{idx:02d}.png"
        out_path = os.path.join(output_dir, out_name)
        Image.fromarray(img).save(out_path)


def _pick_obj_file(model_dir: str) -> Optional[str]:
    d = Path(model_dir)
    if not d.is_dir():
        return None

    preferred = [
        "model.obj",
        "models/model.obj",
        "model_normalized.obj",
        "normalized.obj",
        "mesh.obj",
    ]
    for rel in preferred:
        p = d / rel
        if p.is_file():
            return str(p)

    candidates = sorted(d.glob("*.obj"))
    if candidates:
        return str(candidates[0])

    candidates = sorted(d.rglob("*.obj"))
    if candidates:
        return str(candidates[0])

    return None


def _load_obj_as_trimesh(obj_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported trimesh object type: {type(mesh)}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces.")

    # Keep consistent with your current glb script (no forced triangulation here).
    mesh.remove_unreferenced_vertices()
    return mesh


def render_shapenet_fixed_mesh_dataset(
    json_path: str,
    fixed_mesh_root: str,
    output_dir: str,
    resolution: int = 512,
    pitch_deg: float = 30.0,
    yaws_deg: List[float] = None,
    r: float = 2.0,
    fov_deg: float = 40.0,
    near: float = 1.0,
    far: float = 100.0,
    ssaa: int = 4,
    margin: float = 0.95,
) -> None:
    if yaws_deg is None:
        yaws_deg = [0.0, 90.0, 180.0, 270.0]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for trellis MeshRenderer, but torch.cuda.is_available() is False.")

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("JSON root must be a list.")

    names: List[str] = []
    for item in dataset:
        if isinstance(item, dict) and "name" in item:
            names.append(str(item["name"]))

    if not names:
        raise ValueError("No 'name' items found in JSON.")

    extrinsics, intrinsics = _build_fixed_cameras(pitch_deg=pitch_deg, yaws_deg=yaws_deg, r=r, fov_deg=fov_deg)

    for name in tqdm(names, desc="Rendering normals (fixed_mesh obj)"):
        model_dir = os.path.join(fixed_mesh_root, name)
        obj_path = _pick_obj_file(model_dir)
        if obj_path is None:
            print(f"[WARN] Missing obj for name={name} under {model_dir}")
            continue

        try:
            mesh = _load_obj_as_trimesh(obj_path)
        except Exception as e:
            print(f"[WARN] Failed to load obj for name={name} ({obj_path}): {e}")
            continue

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        vertices = _normalize_mesh_in_place(
            vertices=vertices,
            r=r,
            fov_deg=fov_deg,
            near=near,
            margin=margin,
        )

        mesh_cuda = MeshExtractResult(
            vertices=torch.from_numpy(vertices).cuda(),
            faces=torch.from_numpy(faces).cuda(),
            vertex_attrs=None,
            res=64,
        )

        rets = render_frames(
            sample=mesh_cuda,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            options={
                "resolution": int(resolution),
                "near": float(near),
                "far": float(far),
                "ssaa": int(ssaa),
            },
            verbose=False,
        )

        normals = rets.get("normal", None)
        if not normals:
            print(f"[WARN] No normals rendered for name={name} ({obj_path})")
            continue

        _save_normal_images(normals=normals, model_name=name, yaws_deg=yaws_deg, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 4-view normal maps for ShapeNet fixed_mesh .obj dataset.")
    parser.add_argument(
        "--json_path",
        type=str,
        default="datasets/shapenet/mllm_shapenet_test_airplane_image.json",
    )
    parser.add_argument(
        "--fixed_mesh_root",
        type=str,
        default="datasets/shapenet/fixed_mesh/02691156",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/shapenet/reference_normals",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--pitch", type=float, default=30.0)
    parser.add_argument("--r", type=float, default=2.0)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument("--near", type=float, default=1.0)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--ssaa", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.95)
    args = parser.parse_args()

    render_shapenet_fixed_mesh_dataset(
        json_path=args.json_path,
        fixed_mesh_root=args.fixed_mesh_root,
        output_dir=args.output_dir,
        resolution=args.resolution,
        pitch_deg=args.pitch,
        yaws_deg=[0.0, 90.0, 180.0, 270.0],
        r=args.r,
        fov_deg=args.fov,
        near=args.near,
        far=args.far,
        ssaa=args.ssaa,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()


