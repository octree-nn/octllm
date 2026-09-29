import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "VoxelDataset",
    "voxel_collate_fn",
    "VoxelDatasetConfig",
    "PairedVoxelDataset",
    "PairedVoxelDatasetConfig",
    "paired_voxel_collate_fn",
]


@dataclass
class VoxelDatasetConfig:
    root: str
    extensions: Sequence[str] = ("npy", "pt", "binvox")
    enforce_shape: Optional[Tuple[int, int, int]] = (64, 64, 64)
    binary_threshold: Optional[float] = None
    normalize: Optional[str] = None
    clamp: bool = True
    dtype: torch.dtype = torch.float32
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    return_path: bool = False


def _expand_root(root: str) -> str:
    expanded = os.path.expanduser(root)
    return os.path.abspath(expanded)


def _gather_files(root: str, extensions: Sequence[str]) -> List[str]:
    extensions = tuple(ext.lower() for ext in extensions)
    files: List[str] = []
    for parent, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("."):
                continue
            ext = name.split(".")[-1].lower()
            if ext in extensions:
                files.append(os.path.join(parent, name))
    files.sort()
    return files


def _load_tensor(path: str) -> torch.Tensor:
    ext = path.split(".")[-1].lower()
    if ext == "npy":
        array = np.load(path)
        tensor = torch.from_numpy(array)
    elif ext == "pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            tensor = obj
        else:
            raise TypeError(f"Unsupported object stored in {path}: {type(obj)}")
    elif ext == "binvox":
        tensor = _load_binvox(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    if tensor.ndim == 1:
        raise ValueError(f"Invalid tensor shape {tuple(tensor.shape)} from {path}")
    return tensor.to(torch.float32)


def _normalize_tensor(tensor: torch.Tensor, mode: Optional[str]) -> torch.Tensor:
    if mode == "0_1":
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val > min_val:
            tensor = (tensor - min_val) / (max_val - min_val)
        else:
            tensor = torch.zeros_like(tensor)
    elif mode == "neg1_1":
        tensor = 2.0 * tensor - 1.0
    elif mode is None:
        return tensor
    else:
        raise ValueError(f"Unsupported normalize mode: {mode}")
    return tensor


def _postprocess_tensor(
    tensor: torch.Tensor,
    enforce_shape: Optional[Tuple[int, int, int]],
    dtype: torch.dtype,
    binary_threshold: Optional[float],
    normalize: Optional[str],
    clamp: bool,
) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}")
    if enforce_shape is not None:
        expected = (enforce_shape[0], enforce_shape[1], enforce_shape[2])
        spatial = tensor.shape[-3:]
        if spatial != expected:
            raise ValueError(f"Expected spatial shape {expected}, got {spatial}")
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if binary_threshold is not None:
        tensor = (tensor >= binary_threshold).to(dtype)
    tensor = _normalize_tensor(tensor, normalize)
    if clamp:
        tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor


def _load_binvox(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        header = f.readline().strip()
        if not header.startswith(b"#binvox"):
            raise ValueError(f"File {path} is not a valid binvox file.")
        dims = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading header of {path}")
            line = line.strip()
            if line == b"data":
                break
            key, *values = line.split()
            if key == b"dim":
                dims = tuple(int(v) for v in values)
        if dims is None:
            raise ValueError(f"Missing dim header in {path}")
        data = f.read()
        if len(data) % 2 != 0:
            data = data[:-1]
        raw = np.frombuffer(data, dtype=np.uint8)
        values = raw[0::2]
        counts = raw[1::2]
        voxels = np.repeat(values, counts).astype(np.uint8)
        expected = dims[0] * dims[1] * dims[2]
        if voxels.size != expected:
            raise ValueError(f"Invalid RLE payload size in {path}, expected {expected}, got {voxels.size}")
        voxels = voxels.reshape(dims)
        tensor = torch.from_numpy(voxels).to(torch.float32)
        return tensor


class VoxelDataset(Dataset):
    def __init__(
        self,
        root: Union[str, VoxelDatasetConfig],
        extensions: Optional[Sequence[str]] = None,
        enforce_shape: Optional[Tuple[int, int, int]] = None,
        binary_threshold: Optional[float] = None,
        normalize: Optional[str] = None,
        clamp: Optional[bool] = None,
        dtype: torch.dtype = torch.float32,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_path: bool = False,
    ) -> None:
        if isinstance(root, VoxelDatasetConfig):
            config = root
            root = config.root
            extensions = extensions or config.extensions
            enforce_shape = enforce_shape or config.enforce_shape
            if binary_threshold is None:
                binary_threshold = config.binary_threshold
            normalize = normalize or config.normalize
            clamp = clamp if clamp is not None else config.clamp
            dtype = dtype or config.dtype
            transform = transform or config.transform
            return_path = return_path or config.return_path
        if extensions is None:
            extensions = ("npy", "pt", "binvox")
        self.root = _expand_root(root)
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.enforce_shape = enforce_shape
        self.binary_threshold = binary_threshold
        self.normalize = normalize
        self.clamp = True if clamp is None else clamp
        self.dtype = dtype
        self.transform = transform
        self.return_path = return_path

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root {self.root} does not exist.")
        self.files = _gather_files(self.root, self.extensions)
        if not self.files:
            raise RuntimeError(f"No voxel files with extensions {self.extensions} found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        tensor = _load_tensor(path)
        tensor = self._postprocess(tensor)
        if self.transform is not None:
            tensor = self.transform(tensor)
        if self.return_path:
            return tensor, path
        return tensor

    def _postprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        return _postprocess_tensor(
            tensor,
            self.enforce_shape,
            self.dtype,
            self.binary_threshold,
            self.normalize,
            self.clamp,
        )

    @staticmethod
    def _normalize(tensor: torch.Tensor, mode: str) -> torch.Tensor:
        return _normalize_tensor(tensor, mode)


@dataclass
class PairedVoxelDatasetConfig:
    input: VoxelDatasetConfig
    target: VoxelDatasetConfig
    transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
    return_path: bool = False
    strict: bool = True


class PairedVoxelDataset(Dataset):
    """
    Dataset that loads paired voxels (e.g., partial and completed) matched by filename.
    """

    def __init__(self, config: PairedVoxelDatasetConfig) -> None:
        if not isinstance(config, PairedVoxelDatasetConfig):
            raise TypeError("PairedVoxelDataset requires PairedVoxelDatasetConfig.")

        self.input_config = config.input
        self.target_config = config.target
        self.transform = config.transform
        self.return_path = config.return_path
        self.strict = config.strict

        self.input_root = _expand_root(self.input_config.root)
        self.target_root = _expand_root(self.target_config.root)

        if not os.path.isdir(self.input_root):
            raise FileNotFoundError(f"Input dataset root {self.input_root} does not exist.")
        if not os.path.isdir(self.target_root):
            raise FileNotFoundError(f"Target dataset root {self.target_root} does not exist.")

        self.input_extensions = tuple(ext.lower() for ext in self.input_config.extensions)
        self.target_extensions = tuple(ext.lower() for ext in self.target_config.extensions)

        self.input_files = _gather_files(self.input_root, self.input_extensions)
        if not self.input_files:
            raise RuntimeError(f"No voxel files with extensions {self.input_extensions} found in {self.input_root}")

        self.pairs: List[Tuple[str, str]] = []
        for input_path in self.input_files:
            rel_path = os.path.relpath(input_path, self.input_root)
            target_path = os.path.join(self.target_root, rel_path)
            if not os.path.exists(target_path):
                stem, _ = os.path.splitext(rel_path)
                target_path = self._find_target_candidate(stem)
                if target_path is None:
                    if self.strict:
                        raise FileNotFoundError(
                            f"Target voxel matching {input_path} not found under {self.target_root}."
                        )
                    continue
            self.pairs.append((input_path, target_path))

        if not self.pairs:
            raise RuntimeError("No paired voxel samples found. Check directory structure and file extensions.")

    def _find_target_candidate(self, stem: str) -> Optional[str]:
        for ext in self.target_extensions:
            suffix = ext if ext.startswith(".") else f".{ext}"
            candidate = os.path.join(self.target_root, stem + suffix)
            if os.path.exists(candidate):
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        input_path, target_path = self.pairs[index]
        input_tensor = _load_tensor(input_path)
        target_tensor = _load_tensor(target_path)

        input_tensor = _postprocess_tensor(
            input_tensor,
            self.input_config.enforce_shape,
            self.input_config.dtype,
            self.input_config.binary_threshold,
            self.input_config.normalize,
            self.input_config.clamp,
        )
        target_tensor = _postprocess_tensor(
            target_tensor,
            self.target_config.enforce_shape,
            self.target_config.dtype,
            self.target_config.binary_threshold,
            self.target_config.normalize,
            self.target_config.clamp,
        )

        if self.input_config.transform is not None:
            input_tensor = self.input_config.transform(input_tensor)
        if self.target_config.transform is not None:
            target_tensor = self.target_config.transform(target_tensor)
        if self.transform is not None:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        if self.return_path:
            return input_tensor, target_tensor, (input_path, target_path)
        return input_tensor, target_tensor


def voxel_collate_fn(batch):
    if isinstance(batch[0], tuple):
        first = batch[0]
        if len(first) == 2 and isinstance(first[1], torch.Tensor):
            inputs, targets = zip(*batch)
            return torch.stack(inputs, dim=0), torch.stack(targets, dim=0)
        tensors, paths = zip(*batch)
        tensors = torch.stack(tensors, dim=0)
        return tensors, list(paths)
    return torch.stack(batch, dim=0)


def paired_voxel_collate_fn(batch):
    first = batch[0]
    if len(first) == 3:
        inputs, targets, paths = zip(*batch)
        return torch.stack(inputs, dim=0), torch.stack(targets, dim=0), list(paths)
    if len(first) == 2 and all(isinstance(item, torch.Tensor) for item in first):
        inputs, targets = zip(*batch)
        return torch.stack(inputs, dim=0), torch.stack(targets, dim=0)
    raise ValueError("Unexpected batch structure for paired_voxel_collate_fn.")

