import numpy as np

from heatsplat.colormap import grayscale, ironbow, normalize


def test_ironbow_endpoints_and_monotonic_brightness():
    t = np.linspace(0, 1, 50)
    rgb = ironbow(t)
    assert np.allclose(rgb[0], [0, 0, 0])
    assert np.allclose(rgb[-1], [1, 1, 1])
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    assert np.all(np.diff(luma) >= -1e-6)


def test_normalize_uses_percentiles_and_clips():
    v = np.concatenate([np.linspace(0, 10, 100), [1000.0]])
    out = normalize(v)
    assert out.min() == 0.0 and out.max() == 1.0
    assert out[-1] == 1.0 and out[-2] == 1.0  # outlier does not stretch the range


def test_normalize_constant_input_is_zero():
    assert np.all(normalize(np.full(5, 3.0)) == 0.0)


def test_grayscale_scales_brightness():
    out = grayscale(np.array([[1.0, 1.0, 1.0]], np.float32), brightness=0.5)
    assert np.allclose(out, 0.5)
