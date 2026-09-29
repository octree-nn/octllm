import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import linalg
from tqdm.auto import tqdm


_SELECTED_VIEWS = ("014", "015", "016", "017")


def _ensure_repo_root_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_path()


def _load_eval_asset_ids(split_json: Path, max_assets: Optional[int] = None) -> List[str]:
    items = json.loads(split_json.read_text(encoding="utf-8"))
    asset_ids = sorted({item["asset_id"] for item in items})
    if max_assets is not None:
        asset_ids = asset_ids[:max_assets]
    return asset_ids


def _collect_gt_view_paths(renders_cond_dir: Path, asset_id: str, selected_views: Sequence[str]) -> Optional[List[Path]]:
    paths = [renders_cond_dir / asset_id / f"{view}.png" for view in selected_views]
    if all(path.is_file() for path in paths):
        return paths
    return None


def _collect_generated_view_paths(rendered_dir: Path, asset_id: str, selected_views: Sequence[str]) -> Optional[List[Path]]:
    paths = [rendered_dir / f"{asset_id}__glb__{view}.png" for view in selected_views]
    if all(path.is_file() for path in paths):
        return paths
    return None


def collect_eval_images(
    split_json: Path,
    renders_cond_dir: Path,
    generated_render_dir: Path,
    selected_views: Sequence[str],
    max_assets: Optional[int] = None,
) -> Tuple[List[Path], List[Path], Dict[str, object]]:
    asset_ids = _load_eval_asset_ids(split_json, max_assets=max_assets)
    gt_images: List[Path] = []
    generated_images: List[Path] = []
    gt_asset_ids: List[str] = []
    generated_asset_ids: List[str] = []
    missing_gt: List[str] = []
    missing_generated: List[str] = []

    for asset_id in asset_ids:
        gt_paths = _collect_gt_view_paths(renders_cond_dir, asset_id, selected_views)
        if gt_paths is None:
            missing_gt.append(asset_id)
        else:
            gt_asset_ids.append(asset_id)
            gt_images.extend(gt_paths)

        generated_paths = _collect_generated_view_paths(generated_render_dir, asset_id, selected_views)
        if generated_paths is None:
            missing_generated.append(asset_id)
        else:
            generated_asset_ids.append(asset_id)
            generated_images.extend(generated_paths)

    summary: Dict[str, object] = {
        "requested_asset_count": len(asset_ids),
        "gt_asset_count": len(gt_asset_ids),
        "generated_asset_count": len(generated_asset_ids),
        "selected_views": list(selected_views),
        "missing_gt_count": len(missing_gt),
        "missing_generated_count": len(missing_generated),
        "missing_gt_examples": missing_gt[:10],
        "missing_generated_examples": missing_generated[:10],
    }
    return gt_images, generated_images, summary


def _materialize_dir(src_images: Sequence[Path], dst_dir: Path, use_symlink: bool = True) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(src_images):
        ext = src.suffix.lower() or ".png"
        dst = dst_dir / f"{idx:08d}{ext}"
        if use_symlink:
            os.symlink(src.as_posix(), dst.as_posix())
        else:
            dst.write_bytes(src.read_bytes())


def compute_inception_fid_kid(
    gt_images: Sequence[Path],
    generated_images: Sequence[Path],
    device: str,
    mode: str,
    use_symlink: bool,
) -> Dict[str, float]:
    from cleanfid import fid as cleanfid_fid  # type: ignore

    with tempfile.TemporaryDirectory(prefix="toys4k_gt_") as gt_tmp, tempfile.TemporaryDirectory(
        prefix="toys4k_gen_"
    ) as gen_tmp:
        gt_dir = Path(gt_tmp)
        gen_dir = Path(gen_tmp)
        _materialize_dir(gt_images, gt_dir, use_symlink=use_symlink)
        _materialize_dir(generated_images, gen_dir, use_symlink=use_symlink)

        fid_value = cleanfid_fid.compute_fid(
            fdir1=gt_dir.as_posix(),
            fdir2=gen_dir.as_posix(),
            mode=mode,
            device=device,
        )
        kid_value = cleanfid_fid.compute_kid(
            fdir1=gt_dir.as_posix(),
            fdir2=gen_dir.as_posix(),
            mode=mode,
            device=device,
        )

    return {
        "fid_inception": float(fid_value),
        "kid_inception": float(kid_value),
    }


class DinoFeatureExtractor:
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        local_repo_dir: Optional[str],
        local_weights_path: Optional[str],
        image_size: int,
    ):
        import torch
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.torch = torch
        self.device = device
        self.model_mode = "transformers"
        # Match the paper's ViT-L/14 register-token preprocessing in both backends.
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        repo_dir = Path(local_repo_dir).expanduser() if local_repo_dir else None
        weights_path = Path(local_weights_path).expanduser() if local_weights_path else None
        if repo_dir is not None or weights_path is not None:
            if repo_dir is None or weights_path is None:
                raise ValueError("Set both --dino_repo_dir and --dino_weights_path for the local torch hub backend.")
            if not repo_dir.is_dir():
                raise FileNotFoundError(repo_dir)
            if not weights_path.is_file():
                raise FileNotFoundError(weights_path)
            self.model = torch.hub.load(
                repo_or_dir=repo_dir.as_posix(),
                model="dinov2_vitl14_reg",
                source="local",
                pretrained=True,
                weights=weights_path.as_posix(),
            ).to(device)
            self.model.eval()
            self.model_mode = "torchhub"
            return

        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.model.eval()

    def extract(self, image_paths: Sequence[Path], batch_size: int, progress_desc: str) -> np.ndarray:
        features: List[np.ndarray] = []
        batch_starts = range(0, len(image_paths), batch_size)
        for start in tqdm(batch_starts, desc=progress_desc):
            batch_paths = image_paths[start : start + batch_size]
            batch_images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    batch_images.append(image.convert("RGB"))

            with self.torch.no_grad():
                inputs = self.torch.stack([self.transform(image) for image in batch_images]).to(self.device)
                if self.model_mode == "torchhub":
                    outputs = self.model.forward_features(inputs)
                    batch_features = outputs["x_norm_clstoken"]
                else:
                    outputs = self.model(pixel_values=inputs)
                    batch_features = outputs.last_hidden_state[:, 0]

            features.append(batch_features.detach().cpu().numpy().astype(np.float64, copy=False))
        return np.concatenate(features, axis=0)


def _compute_fid_from_features(features1: np.ndarray, features2: np.ndarray) -> float:
    mu1 = np.mean(features1, axis=0)
    mu2 = np.mean(features2, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    sigma2 = np.cov(features2, rowvar=False)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_value = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(np.real(fid_value))


def _polynomial_mmd(x: np.ndarray, y: np.ndarray) -> float:
    dim = x.shape[1]
    k_xx = ((x @ x.T) / dim + 1.0) ** 3
    k_yy = ((y @ y.T) / dim + 1.0) ** 3
    k_xy = ((x @ y.T) / dim + 1.0) ** 3

    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    m = x.shape[0]
    n = y.shape[0]
    value = k_xx.sum() / (m * (m - 1)) + k_yy.sum() / (n * (n - 1)) - 2.0 * k_xy.mean()
    return float(value)


def _compute_kid_from_features(
    features1: np.ndarray,
    features2: np.ndarray,
    subset_size: int,
    num_subsets: int,
    seed: int,
    progress_desc: str,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    subset_size = min(subset_size, len(features1), len(features2))
    values: List[float] = []

    for _ in tqdm(range(num_subsets), desc=progress_desc):
        idx1 = rng.choice(len(features1), size=subset_size, replace=False)
        idx2 = rng.choice(len(features2), size=subset_size, replace=False)
        values.append(_polynomial_mmd(features1[idx1], features2[idx2]))

    values_array = np.asarray(values, dtype=np.float64)
    kid_mean = float(values_array.mean())
    kid_std = float(values_array.std(ddof=1)) if len(values_array) > 1 else 0.0
    return kid_mean, kid_std


def compute_dino_fid_kid(
    gt_images: Sequence[Path],
    generated_images: Sequence[Path],
    model_name_or_path: str,
    device: str,
    batch_size: int,
    kid_subset_size: int,
    kid_num_subsets: int,
    seed: int,
    local_repo_dir: Optional[str],
    local_weights_path: Optional[str],
    image_size: int,
    progress_prefix: str,
) -> Dict[str, float]:
    extractor = DinoFeatureExtractor(
        model_name_or_path=model_name_or_path,
        device=device,
        local_repo_dir=local_repo_dir,
        local_weights_path=local_weights_path,
        image_size=image_size,
    )
    gt_features = extractor.extract(gt_images, batch_size=batch_size, progress_desc=f"DINO features [{progress_prefix}] GT")
    generated_features = extractor.extract(
        generated_images,
        batch_size=batch_size,
        progress_desc=f"DINO features [{progress_prefix}] Generated",
    )

    fid_value = _compute_fid_from_features(gt_features, generated_features)
    kid_mean, kid_std = _compute_kid_from_features(
        gt_features,
        generated_features,
        subset_size=kid_subset_size,
        num_subsets=kid_num_subsets,
        seed=seed,
        progress_desc=f"DINO KID subsets [{progress_prefix}]",
    )
    return {
        "fid_dino": fid_value,
        "kid_dino": kid_mean,
        "kid_dino_std": kid_std,
        "dino_feature_dim": int(gt_features.shape[1]),
        "dino_backend": extractor.model_mode,
    }


def _resolve_condition_dirs(method_dir: Path, requested: Optional[Sequence[str]]) -> List[Path]:
    if requested:
        return [method_dir / name for name in requested]
    return sorted(path for path in method_dir.iterdir() if path.is_dir() and path.name.endswith("rendered"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Toys4k FID/KID on selected views using Inception-V3 and DINOv2 features."
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default="datasets/toys4k/test.json",
        help="Path to the Toys4k evaluation split JSON.",
    )
    parser.add_argument(
        "--renders_cond_dir",
        type=str,
        default="datasets/toys4k/renders_cond",
        help="Directory with ground-truth render folders.",
    )
    parser.add_argument(
        "--method_dir",
        type=str,
        default="outputs/toys4k",
        help="Method directory containing rendered generation results.",
    )
    parser.add_argument(
        "--condition_dirs",
        nargs="*",
        default=None,
        help="Optional rendered subdirectories to evaluate, e.g. image_conditionrendered text_conditionrendered.",
    )
    parser.add_argument(
        "--selected_views",
        nargs="+",
        default=list(_SELECTED_VIEWS),
        help="View indices to evaluate.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device for feature extraction, e.g. cpu/cuda/cuda:0.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DINO feature extraction.")
    parser.add_argument(
        "--dino_model_name_or_path",
        type=str,
        default="facebook/dinov2-with-registers-large",
        help="Hugging Face DINOv2 repository ID or local Transformers model directory.",
    )
    parser.add_argument(
        "--dino_repo_dir",
        type=str,
        default=None,
        help="Optional local DINOv2 torch hub source directory; requires --dino_weights_path.",
    )
    parser.add_argument(
        "--dino_weights_path",
        type=str,
        default=None,
        help="Optional local DINOv2 torch hub weights; requires --dino_repo_dir.",
    )
    parser.add_argument("--dino_image_size", type=int, default=518, help="Input size for DINO preprocessing.")
    parser.add_argument("--kid_subset_size", type=int, default=1000, help="Subset size used for DINO KID estimation.")
    parser.add_argument("--kid_num_subsets", type=int, default=100, help="Number of subsets used for DINO KID estimation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for DINO KID estimation.")
    parser.add_argument("--mode", type=str, default="clean", help="cleanfid mode for Inception metrics.")
    parser.add_argument("--no_symlink", action="store_true", help="Copy temporary files instead of symlinking them.")
    parser.add_argument("--max_assets", type=int, default=None, help="Optional cap for quick debugging runs.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save metrics as JSON.")
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    split_json = Path(args.split_json).expanduser().resolve()
    renders_cond_dir = Path(args.renders_cond_dir).expanduser().resolve()
    method_dir = Path(args.method_dir).expanduser().resolve()
    condition_dirs = _resolve_condition_dirs(method_dir, args.condition_dirs)

    results: Dict[str, object] = {
        "split_json": split_json.as_posix(),
        "renders_cond_dir": renders_cond_dir.as_posix(),
        "method_dir": method_dir.as_posix(),
        "device": device,
        "selected_views": list(args.selected_views),
        "results": {},
    }

    for condition_dir in condition_dirs:
        gt_images, generated_images, summary = collect_eval_images(
            split_json=split_json,
            renders_cond_dir=renders_cond_dir,
            generated_render_dir=condition_dir,
            selected_views=args.selected_views,
            max_assets=args.max_assets,
        )
        if not gt_images:
            raise ValueError(f"No ground-truth evaluation images found for: {condition_dir}")
        if not generated_images:
            raise ValueError(f"No generated evaluation images found for: {condition_dir}")

        inception_metrics = compute_inception_fid_kid(
            gt_images=gt_images,
            generated_images=generated_images,
            device=device,
            mode=args.mode,
            use_symlink=not args.no_symlink,
        )
        dino_metrics = compute_dino_fid_kid(
            gt_images=gt_images,
            generated_images=generated_images,
            model_name_or_path=args.dino_model_name_or_path,
            device=device,
            batch_size=args.batch_size,
            kid_subset_size=args.kid_subset_size,
            kid_num_subsets=args.kid_num_subsets,
            seed=args.seed,
            local_repo_dir=args.dino_repo_dir,
            local_weights_path=args.dino_weights_path,
            image_size=args.dino_image_size,
            progress_prefix=condition_dir.name,
        )

        condition_result = dict(summary)
        condition_result.update(
            {
                "gt_image_count": len(gt_images),
                "generated_image_count": len(generated_images),
            }
        )
        condition_result.update(inception_metrics)
        condition_result.update(dino_metrics)
        results["results"][condition_dir.name] = condition_result

    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output_json:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
