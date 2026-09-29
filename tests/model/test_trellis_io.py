"""Regression checks for Gaussian file transforms and sparse batch ordering."""

import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "transform",
    [None, [[1, 0, 0], [0, 0, -1], [0, 1, 0]], [[0, -1, 0], [1, 0, 0], [0, 0, 1]]],
    ids=["no-transform", "default-transform", "rotated-axes"],
)
def test_gaussian_ply_round_trip_preserves_geometry(tmp_path, monkeypatch, transform):
    utils3d = pytest.importorskip("utils3d")
    pytest.importorskip("plyfile")
    from trellis.representations.gaussian.gaussian_model import Gaussian

    # Isolate file I/O from the constructor's GPU-only bias allocation.
    with monkeypatch.context() as patches:
        patches.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
        original = Gaussian([-1, -1, -1, 2, 2, 2], device="cpu")
        restored = Gaussian([-1, -1, -1, 2, 2, 2], device="cpu")

    original.from_xyz(torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.4, 0.1]]))
    original.from_scaling(torch.tensor([[0.02, 0.03, 0.04], [0.05, 0.03, 0.02]]))
    original.from_rotation(torch.nn.functional.normalize(torch.tensor([[1.0, 0.3, 0.2, 0.4], [1.0, 0.1, 0.5, 0.2]])))
    original.from_features(torch.tensor([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]]))
    original.from_opacity(torch.tensor([[0.3], [0.7]]))
    path = tmp_path / "gaussians.ply"

    original.save_ply(path, transform=transform)
    restored.load_ply(path, transform=transform)

    for name in ("get_xyz", "get_scaling", "get_features", "get_opacity"):
        torch.testing.assert_close(getattr(restored, name), getattr(original, name))
    # A quaternion and its negation describe the same rotation.
    np.testing.assert_allclose(
        utils3d.numpy.quaternion_to_matrix(restored.get_rotation.numpy()),
        utils3d.numpy.quaternion_to_matrix(original.get_rotation.numpy()),
        atol=1e-6,
    )


def test_sparse_convolution_restores_contiguous_batch_layout():
    spconv = pytest.importorskip("spconv.pytorch")
    from trellis.modules.sparse import SparseTensor
    from trellis.modules.sparse.conv.conv_spconv import SparseConv3d

    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.int32)
    x = SparseTensor(torch.ones(4, 1), coords)
    permutation = torch.tensor([2, 0, 3, 1])
    unsorted = spconv.SparseConvTensor(
        torch.tensor([[10.0], [20.0], [30.0], [40.0]]), coords[permutation], [2, 2, 2], 2
    )

    class ConvolutionOutput(torch.nn.Module):
        out_channels = 1

        def forward(self, data):
            return unsorted

    layer = SparseConv3d(1, 1, kernel_size=3, stride=2, padding=0)
    # Exercise the batch reorder independently of the GPU convolution kernel.
    layer.conv = ConvolutionOutput()
    result = layer(x)

    assert result.coords[:, 0].tolist() == [0, 0, 1, 1]
    assert result.feats[:, 0].tolist() == [20.0, 40.0, 10.0, 30.0]
    reverse_order = result.get_spatial_cache("conv_(2, 2, 2)_sort_bwd")
    torch.testing.assert_close(result.feats[reverse_order], unsorted.features)
