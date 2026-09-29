import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

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


def _rotation_matrix_xyz_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Build rotation matrix from Euler angles (degrees) in X->Y->Z order.
    """
    rx = math.radians(float(rx))
    ry = math.radians(float(ry))
    rz = math.radians(float(rz))

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def _rotate_vertices(vertices: np.ndarray, rot_x: float, rot_y: float, rot_z: float) -> np.ndarray:
    if vertices.size == 0:
        return vertices
    if float(rot_x) == 0.0 and float(rot_y) == 0.0 and float(rot_z) == 0.0:
        return vertices
    rmat = _rotation_matrix_xyz_deg(rot_x, rot_y, rot_z)
    return vertices @ rmat.T


def _load_glb_as_trimesh(glb_path: str) -> trimesh.Trimesh:
    obj = trimesh.load(glb_path, force=None)
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("No Trimesh geometries found in scene.")
        mesh = trimesh.util.concatenate(geoms)
    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj
    else:
        raise TypeError(f"Unsupported trimesh object type: {type(obj)}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces.")

    if not mesh.is_watertight:
        # Not required for normal rendering; keep as-is.
        pass


    mesh.remove_unreferenced_vertices()
    return mesh


def _normalize_mesh_in_place(
    vertices: np.ndarray,
    r: float,
    fov_deg: float,
    near: float,
    margin: float,
    rot_x: float = 0.0,
    rot_y: float = 0.0,
    rot_z: float = 0.0,
) -> np.ndarray:
    """
    Center mesh at origin and scale it to fit into camera frustum.
    """
    if vertices.size == 0:
        return vertices

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    vertices = vertices - center[None, :]

    vertices = _rotate_vertices(vertices, rot_x=rot_x, rot_y=rot_y, rot_z=rot_z)

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
    model_stem: str,
    yaws_deg: List[float],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for idx, (img, yaw) in enumerate(zip(normals, yaws_deg)):
        out_name = f"{model_stem}_yaw{int(yaw)}_view{idx:02d}.png"
        out_path = os.path.join(output_dir, out_name)
        Image.fromarray(img).save(out_path)


def render_directory(
    input_dir: str,
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
    rot_x: float = 0.0,
    rot_y: float = 0.0,
    rot_z: float = 0.0,
) -> None:
    if yaws_deg is None:
        yaws_deg = [0.0, 90.0, 180.0, 270.0]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for trellis MeshRenderer, but torch.cuda.is_available() is False.")

    input_path = Path(input_dir)
    glb_files = sorted([p for p in input_path.rglob("*.glb") if p.is_file()])
    if not glb_files:
        raise FileNotFoundError(f"No .glb files found under: {input_dir}")

    extrinsics, intrinsics = _build_fixed_cameras(pitch_deg=pitch_deg, yaws_deg=yaws_deg, r=r, fov_deg=fov_deg)

    for glb_path in tqdm(glb_files, desc="Rendering normals"):
        model_stem = glb_path.stem
        try:
            mesh = _load_glb_as_trimesh(str(glb_path))
        except Exception as e:
            # Skip broken meshes but keep the batch running.
            # Keep output clean: only print one line per failure.
            print(f"[WARN] Failed to load {glb_path}: {e}")
            continue

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        vertices = _normalize_mesh_in_place(
            vertices=vertices,
            r=r,
            fov_deg=fov_deg,
            near=near,
            margin=margin,
            rot_x=rot_x,
            rot_y=rot_y,
            rot_z=rot_z,
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
            print(f"[WARN] No normals rendered for {glb_path}")
            continue

        _save_normal_images(normals=normals, model_stem=model_stem, yaws_deg=yaws_deg, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 4-view normal maps for all .glb files under a directory.")
    parser.add_argument("--input_dir", type=str, default="outputs/shapenet/octllm/meshes", help="Directory containing .glb files (recursively).")
    parser.add_argument("--output_dir", type=str, default="outputs/shapenet/octllm/normals", help="Directory to save rendered normal images.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--pitch", type=float, default=30.0)
    parser.add_argument("--r", type=float, default=2.0)
    parser.add_argument("--fov", type=float, default=40.0)
    parser.add_argument("--near", type=float, default=1.0)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--ssaa", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.95, help="Safety margin to fit mesh into camera frustum.")
    parser.add_argument("--rot_x", type=float, default=0.0, help="Rotate mesh around X axis (degrees).")
    parser.add_argument("--rot_y", type=float, default=0.0, help="Rotate mesh around Y axis (degrees).")
    parser.add_argument("--rot_z", type=float, default=0.0, help="Rotate mesh around Z axis (degrees).")
    args = parser.parse_args()

    render_directory(
        input_dir=args.input_dir,
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
        rot_x=args.rot_x,
        rot_y=args.rot_y,
        rot_z=args.rot_z,
    )


if __name__ == "__main__":
    main()


