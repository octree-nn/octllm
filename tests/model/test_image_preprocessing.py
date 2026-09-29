"""Regression checks for image-to-3D framing and alpha handling."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.generate_octree import _infer_alpha_from_border, _preprocess_image_paths, preprocess_image_to_train_format


@pytest.mark.parametrize("input_size", [2, 8])
def test_straight_alpha_is_applied_once_even_for_dark_edges(input_size):
    source = Image.new("RGBA", (input_size, input_size), (40, 80, 120, 128))
    result = preprocess_image_to_train_format(source, size=4, object_fill=1.0)

    assert result.mode == "RGBA"
    assert np.all(np.array(result) == [20, 40, 60, 128])
    assert result.convert("RGB").getpixel((0, 0)) == (20, 40, 60)
    assert source.getpixel((0, 0)) == (40, 80, 120, 128)


def test_known_premultiplied_input_keeps_its_colors_and_alpha():
    source = Image.new("RGBA", (2, 2), (20, 40, 60, 128))
    result = preprocess_image_to_train_format(source, size=4, object_fill=1.0, alpha_mode="premultiplied")

    assert np.all(np.array(result) == [20, 40, 60, 128])


def test_centering_preserves_aspect_ratio_and_blacks_out_hidden_rgb():
    source = Image.new("RGBA", (20, 20), (255, 0, 255, 0))
    source.paste((200, 100, 50, 255), (1, 2, 5, 10))
    result = preprocess_image_to_train_format(source, size=32, object_fill=0.75)

    assert result.size == (32, 32)
    assert result.getbbox() == (10, 4, 22, 28)
    pixels = np.array(result)
    assert np.all(pixels[pixels[:, :, 3] == 0, :3] == 0)
    assert result.getpixel((16, 16)) == (200, 100, 50, 255)


def test_background_removal_preserves_enclosed_regions_of_the_same_color():
    pixels = np.full((32, 32, 3), 255, dtype=np.uint8)
    pixels[8:24, 8:24] = (100, 0, 0)
    pixels[12:20, 12:20] = 255
    alpha = _infer_alpha_from_border(pixels)

    assert np.all(alpha[:8] == 0)
    assert np.all(alpha[8:24, 8:24] == 255)


def test_busy_background_is_kept_instead_of_guessing_a_foreground():
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    assert np.all(_infer_alpha_from_border(pixels) == 255)


def test_fully_transparent_input_does_not_reveal_hidden_rgb():
    with pytest.raises(ValueError, match="no visible foreground"):
        preprocess_image_to_train_format(Image.new("RGBA", (16, 16), (255, 0, 0, 0)))


@pytest.mark.parametrize("from_path", [False, True])
def test_exif_orientation_is_applied_before_framing(tmp_path, from_path):
    source = Image.new("RGB", (8, 4), (50, 100, 150))
    source.getexif()[274] = 6  # Rotate a landscape image to portrait.
    image = source
    if from_path:
        image = tmp_path / "oriented.png"
        source.save(image, exif=source.getexif())

    result = preprocess_image_to_train_format(image, size=8, object_fill=1.0)

    assert result.getbbox() == (2, 0, 6, 8)
    assert source.size == (8, 4)


def test_saved_preprocessing_matches_the_model_input_and_disabled_path_is_untouched(tmp_path):
    path = tmp_path / "input.png"
    Image.new("RGBA", (4, 4), (40, 80, 120, 128)).save(path)
    original_bytes = path.read_bytes()
    paths = [str(path)]
    assert _preprocess_image_paths(paths, enabled=False) is paths

    images = _preprocess_image_paths(paths, size=8, object_fill=0.5, save_dir=str(tmp_path / "processed"))
    with Image.open(tmp_path / "processed/input_preprocessed.png") as saved:
        assert np.array_equal(np.array(saved), np.array(images[0]))
    assert images[0].getpixel((4, 4)) == (20, 40, 60, 128)
    assert path.read_bytes() == original_bytes
