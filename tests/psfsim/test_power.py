"""Simple test functions for filter."""

import numpy as np
from psfsim.filter_detector_properties import FilterDetector, n_mercadtel
from psfsim.index_cdte import n_cdte


def test_power():
    """Test for filter power."""

    theta = np.array([0, 21]) * np.pi / 180.0

    for i in range(201):
        wl = 0.35 * (2.4 / 0.35) ** (i / 200.0)
        f = FilterDetector(
            [1.35, 1.82, 2.45, n_cdte(wl), n_mercadtel(wl)],
            [0.163, 0.137, 0.084, 0.010, 0.008],
            1,
        )
        fs, fp = f.transmitted_power(wl, theta)
        assert np.abs(fs[0] - fp[0]) < 1.0e-5
        assert 0 <= fs[0] <= 1
        assert 0 <= fp[0] <= 1
        assert 0 <= fs[1] <= 1
        assert 0 <= fp[1] <= 1

        if np.abs(wl - 0.5) < 0.001:
            assert 0.4 < fs[0] < 0.5
            assert -0.04 < fs[1] - fp[1] < -0.02

        # print(f"{wl:5.3f} {fp[0]:6.4f} {fs[1]:6.4f} {fp[1]:6.4f}")
