import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional


_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".gif",
}


def _ensure_repo_root_on_path() -> None:
    # Keep the same style as other scripts in evaluation/.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_path()


def _iter_image_files(root: Path, recursive: bool = True) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Invalid image directory: {root}")
    it = root.rglob("*") if recursive else root.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            yield p


def _collect_images(root: Path, recursive: bool = True) -> List[Path]:
    files = sorted(_iter_image_files(root, recursive=recursive))
    if not files:
        raise ValueError(f"No images found under: {root}")
    return files


def _materialize_dir(
    src_images: List[Path],
    dst_dir: Path,
    *,
    use_symlink: bool = True,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(src_images):
        ext = src.suffix.lower() or ".png"
        dst = dst_dir / f"{idx:08d}{ext}"
        if dst.exists():
            continue
        if use_symlink:
            os.symlink(src.as_posix(), dst.as_posix())
        else:
            dst.write_bytes(src.read_bytes())


def compute_fid_kid(
    dir1: str,
    dir2: str,
    *,
    recursive: bool = True,
    device: Optional[str] = None,
    mode: str = "clean",
    use_symlink: bool = True,
) -> Dict[str, float]:
    try:
        from cleanfid import fid as cleanfid_fid  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: cleanfid. Please install it in your environment, e.g. `pip install clean-fid`."
        ) from e

    try:
        import torch
    except ModuleNotFoundError:
        torch = None  # type: ignore

    if device is None:
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    root1 = Path(dir1).expanduser().resolve()
    root2 = Path(dir2).expanduser().resolve()
    imgs1 = _collect_images(root1, recursive=recursive)
    imgs2 = _collect_images(root2, recursive=recursive)

    with tempfile.TemporaryDirectory(prefix="cleanfid_dir1_") as tmp1, tempfile.TemporaryDirectory(
        prefix="cleanfid_dir2_"
    ) as tmp2:
        tmp_dir1 = Path(tmp1)
        tmp_dir2 = Path(tmp2)
        _materialize_dir(imgs1, tmp_dir1, use_symlink=use_symlink)
        _materialize_dir(imgs2, tmp_dir2, use_symlink=use_symlink)

        fid_value = cleanfid_fid.compute_fid(
            fdir1=tmp_dir1.as_posix(),
            fdir2=tmp_dir2.as_posix(),
            mode=mode,
            device=device,
        )

        kid_value = cleanfid_fid.compute_kid(
            fdir1=tmp_dir1.as_posix(),
            fdir2=tmp_dir2.as_posix(),
            mode=mode,
            device=device,
        )

    return {"fid": fid_value, "kid": kid_value}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute image similarity metrics (Inception-V3) between two directories: FID and KID (via cleanfid)."
    )
    parser.add_argument("--dir1", type=str, default="outputs/shapenet/reference_normals", help="First image directory.")
    parser.add_argument("--dir2", type=str, default="outputs/shapenet/octllm/normals", help="Second image directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search images under directories.")
    parser.add_argument("--device", type=str, default=None, help="Device passed to cleanfid, e.g. cpu/cuda/cuda:0.")
    parser.add_argument("--mode", type=str, default="clean", help="cleanfid mode, e.g. clean/legacy.")
    parser.add_argument("--no_symlink", action="store_true", help="Copy images instead of symlinking.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output JSON path.")

    args = parser.parse_args()

    metrics = compute_fid_kid(
        dir1=args.dir1,
        dir2=args.dir2,
        recursive=bool(args.recursive),
        device=args.device,
        mode=str(args.mode),
        use_symlink=not bool(args.no_symlink),
    )

    payload: Dict[str, float] = {
        "dir1": str(Path(args.dir1).expanduser().resolve()),
        "dir2": str(Path(args.dir2).expanduser().resolve()),
        "recursive": bool(args.recursive),
        "device": args.device,
        "mode": str(args.mode),
        "fid": metrics["fid"],
        "kid": metrics["kid"],
    }

    print(json.dumps(payload, ensure_ascii=False))

    if args.output_json:
        out_path = Path(args.output_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


