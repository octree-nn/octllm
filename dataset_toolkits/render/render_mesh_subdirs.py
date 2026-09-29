import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from subprocess import run

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset_toolkits.render.utils import sphere_hammersley_sequence


BLENDER_VERSION = "4.0.0"
BLENDER_LINK = (
    f"https://download.blender.org/release/Blender{BLENDER_VERSION[:3]}/"
    f"blender-{BLENDER_VERSION}-linux-x64.tar.xz"
)
BLENDER_INSTALLATION_PATH = os.path.expanduser("~/")
BLENDER_PATH = f"{BLENDER_INSTALLATION_PATH}/blender-{BLENDER_VERSION}-linux-x64/blender"
SUPPORTED_MESH_EXTENSIONS = {
    # ".abc",
    # ".blend",
    # ".dae",
    ".fbx",
    ".glb",
    ".gltf",
    # ".obj",
    ".ply",
    # ".stl",
    # ".usd",
    # ".usda",
}


def _install_blender(binary=None):
    global BLENDER_PATH
    candidate = binary or os.environ.get("BLENDER_PATH") or shutil.which("blender") or BLENDER_PATH
    candidate = os.path.expanduser(candidate)
    if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
        raise FileNotFoundError("Install Blender 4.0 and set --blender-path or BLENDER_PATH. See README.md#installation.")
    BLENDER_PATH = os.path.abspath(candidate)
    return BLENDER_PATH


def _build_views(num_views):
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)

    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (100,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [1.5 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    return [
        {"yaw": y, "pitch": p, "radius": r, "fov": f}
        for y, p, r, f in zip(yaws, pitchs, radius, fov)
    ]


def _is_mesh_file(filename):
    return os.path.splitext(filename)[1].lower() in SUPPORTED_MESH_EXTENSIONS


def _get_immediate_subdirs(root_dir):
    subdirs = []
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        if entry.name.endswith("rendered"):
            continue
        subdirs.append(entry.path)
    return sorted(subdirs)


def _build_output_prefix(mesh_path, source_dir, layout="relative"):
    relative_path = os.path.relpath(mesh_path, source_dir)
    relative_stem, ext = os.path.splitext(relative_path)
    if layout == "octllm":
        parts = relative_stem.split(os.sep)
        if len(parts) != 2 or parts[0] != parts[1]:
            raise ValueError(f"Expected <asset_id>/<asset_id>.glb for OctLLM layout: {relative_path}")
        relative_stem = parts[0]
    prefix = relative_stem.replace(os.sep, "__")
    ext_name = ext.lstrip(".").lower()
    if ext_name:
        prefix = f"{prefix}__{ext_name}"
    return prefix


def _collect_meshes(source_dir):
    mesh_files = []
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if not d.endswith("rendered")]
        for filename in files:
            if not _is_mesh_file(filename):
                continue
            mesh_files.append(os.path.join(root, filename))
    return sorted(mesh_files)


def _render_single_mesh(task, opt):
    mesh_path, source_dir, output_dir = task
    prefix = _build_output_prefix(mesh_path, source_dir, opt.layout)
    marker_file = os.path.join(output_dir, f"{prefix}__transforms.json")

    if os.path.exists(marker_file) and not opt.overwrite:
        return

    os.makedirs(output_dir, exist_ok=True)
    views = _build_views(opt.num_views)
    render_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "blender_script",
        "render.py",
    )
    job_dir = tempfile.mkdtemp(prefix="render_mesh_subdirs_")
    copied_mesh_path = os.path.join(job_dir, f"{uuid.uuid4()}_{os.path.basename(mesh_path)}")
    temp_output_dir = os.path.join(job_dir, "render_output")
    os.makedirs(temp_output_dir, exist_ok=True)

    try:
        shutil.copy2(mesh_path, copied_mesh_path)

        args = [
            opt.blender_path,
            "-b",
            "-P",
            render_script,
            "--",
            "--views",
            json.dumps(views),
            "--object",
            copied_mesh_path,
            "--output_folder",
            temp_output_dir,
            "--resolution",
            str(opt.resolution),
            "--engine",
            opt.engine,
        ]
        if copied_mesh_path.endswith(".blend"):
            args.insert(1, copied_mesh_path)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
        if opt.engine == "BLENDER_EEVEE":
            env["__NV_PRIME_RENDER_OFFLOAD"] = str(opt.gpu_id)
            env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

        result = run(args, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Rendering failed for {mesh_path}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return

        for filename in sorted(os.listdir(temp_output_dir)):
            src_path = os.path.join(temp_output_dir, filename)
            if not os.path.isfile(src_path):
                continue
            dst_path = os.path.join(output_dir, f"{prefix}__{filename}")
            shutil.move(src_path, dst_path)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory whose immediate subdirectories will be rendered.",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=24,
        help="Number of views to render for each mesh.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Render resolution.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE",
        help="Blender rendering engine.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument(
        "--layout", choices=("relative", "octllm"), default="relative",
        help="Use octllm for batch_generate_toys4k.py outputs and Toys4K metric filenames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerender meshes even if the prefixed transforms file already exists.",
    )

    parser.add_argument("--blender-path", help="Path to the Blender 4.0 executable.")
    opt = parser.parse_args()
    root_dir = os.path.abspath(opt.root_dir)

    if not os.path.isdir(root_dir):
        raise ValueError(f"Root directory does not exist: {root_dir}")

    subdirs = _get_immediate_subdirs(root_dir)
    if not subdirs:
        print("No subdirectories found to process.")
        return

    tasks = []
    for source_dir in subdirs:
        output_dir = os.path.join(root_dir, f"{os.path.basename(source_dir)}rendered")
        mesh_files = _collect_meshes(source_dir)
        print(
            f"Found {len(mesh_files)} mesh files under {source_dir}. "
            f"Output will be saved to {output_dir}.",
            flush=True,
        )
        for mesh_path in mesh_files:
            tasks.append((mesh_path, source_dir, output_dir))

    if not tasks:
        print("No mesh files found to render.")
        return

    total_tasks = len(tasks)
    start = total_tasks * opt.rank // opt.world_size
    end = total_tasks * (opt.rank + 1) // opt.world_size
    my_tasks = tasks[start:end]

    print(
        f"Total meshes: {total_tasks}. "
        f"Rank {opt.rank}/{opt.world_size} processing {len(my_tasks)} meshes.",
        flush=True,
    )

    opt.blender_path = _install_blender(opt.blender_path)

    with ProcessPoolExecutor(max_workers=opt.max_workers) as executor:
        render_fn = partial(_render_single_mesh, opt=opt)
        list(tqdm(executor.map(render_fn, my_tasks), total=len(my_tasks), desc="Rendering"))


if __name__ == "__main__":
    main()
