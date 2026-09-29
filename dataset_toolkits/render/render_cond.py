import os
import tempfile
import json
import copy
import sys
import argparse
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from subprocess import DEVNULL, run

import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

# Add parent directory to path to allow imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset_toolkits.render.utils import sphere_hammersley_sequence

BLENDER_VERSION = '4.0.0'
BLENDER_LINK = f'https://download.blender.org/release/Blender{BLENDER_VERSION[:3]}/blender-{BLENDER_VERSION}-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = os.path.expanduser('~/')
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-{BLENDER_VERSION}-linux-x64/blender'

def _install_blender(binary=None):
    global BLENDER_PATH
    candidate = binary or os.environ.get("BLENDER_PATH") or shutil.which("blender") or BLENDER_PATH
    candidate = os.path.expanduser(candidate)
    if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
        raise FileNotFoundError("Install Blender 4.0 and set --blender-path or BLENDER_PATH. See README.md#installation.")
    BLENDER_PATH = os.path.abspath(candidate)
    return BLENDER_PATH


def _render_cond(file_info, opt):
    file_path, relative_dir = file_info
    num_views = opt.num_views
    output_dir = opt.output_dir
    
    # Output folder mirrors the relative directory structure
    output_folder = os.path.join(output_dir, relative_dir)
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if already rendered
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return
        
    # Build camera {yaw, pitch, radius, fov}
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
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    # Create a temporary copy of the object file with a unique name to avoid collisions
    temp_dir = tempfile.mkdtemp(prefix="octllm_render_cond_")
    os.makedirs(temp_dir, exist_ok=True)
    unique_filename = f"{uuid.uuid4()}_{os.path.basename(file_path)}"
    copy_temp = os.path.join(temp_dir, unique_filename)
    shutil.copy2(file_path, copy_temp)
    
    engine = "BLENDER_EEVEE"
    render_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blender_script', 'render.py')
    
    args = [
        opt.blender_path, '-b', '-P', render_script,
        '--',
        '--views', json.dumps(views),
        '--object', copy_temp,
        '--output_folder', output_folder,
        '--resolution', '1024',
        "--engine", engine,
    ]
    if copy_temp.endswith('.blend'):
        args.insert(1, copy_temp)
    
    env = os.environ.copy()
    # Use GPU 0 by default. In a real distributed setting, this might need better management.
    gpu_id = "0"
    env['CUDA_VISIBLE_DEVICES'] = gpu_id
    if engine == "BLENDER_EEVEE":
        env['__NV_PRIME_RENDER_OFFLOAD'] = gpu_id
        env['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    
    try:
        result = run(args,
            env=env,
            capture_output=True,
            text=True
            )
        if result.returncode != 0:
            print(f"Rendering failed for {file_path}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Rendering exception for {file_path}: {e}")
    
    if os.path.exists(copy_temp):
        try:
            os.remove(copy_temp)
            os.rmdir(temp_dir)
        except OSError:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='datasets/shapenet/fixed_mesh',
                        help='Input directory containing obj files')
    parser.add_argument('--output_dir', type=str, default='datasets/shapenet/renders_cond_fixed',
                        help='Directory to save the renders')
    parser.add_argument('--num_views', type=int, default=24,
                        help='Number of views to render')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--blender-path', help='Path to the Blender 4.0 executable.')
    
    args = parser.parse_args()
    opt = edict(vars(args))
    
    print(f"Scanning {opt.input_dir} for .obj files...", flush=True)
    files_to_process = []
    input_dir_abs = os.path.abspath(opt.input_dir)
    
    for root, dirs, files in os.walk(input_dir_abs):
        for file in files:
            if file.endswith('.obj') or file.endswith('.blend'): # Support blend files too as per original script logic
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, input_dir_abs)
                rel_dir = os.path.dirname(rel_path)
                folder_name = os.path.splitext(os.path.basename(file))[0]
                rel_dir = os.path.join(rel_dir, folder_name)
                files_to_process.append((full_path, rel_dir))
    
    files_to_process.sort()
    
    total_files = len(files_to_process)
    if total_files == 0:
        print("No files found to process.")
        return

    start = total_files * opt.rank // opt.world_size
    end = total_files * (opt.rank + 1) // opt.world_size
    my_files = files_to_process[start:end]
    
    print(f"Total files: {total_files}. Rank {opt.rank}/{opt.world_size} processing {len(my_files)} files.", flush=True)
    
    opt.blender_path = _install_blender(opt.blender_path)
    
    # Process using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=opt.max_workers) as executor:
        func = partial(_render_cond, opt=opt)
        # Use tqdm to show progress
        list(tqdm(executor.map(func, my_files), total=len(my_files), desc="Rendering"))

if __name__ == '__main__':
    main()
