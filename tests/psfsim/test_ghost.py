"""Tests for ghost paths."""

import numpy as np
from psfsim.romantrace import RomanRayBundle


def test_ghost():
    """Ghost ray fit."""

    xan = 0.25
    yan = -0.02
    rb = RomanRayBundle(xan, yan, 6, "H", wl=1.14e-3, hasE=True, ghostpath=True, savexy=True)
    coef = rb.fit("xyFPA_from_u")
    print(coef)

    # we should be about 10 mm out of focus
    assert 9.0 < coef["Slope"][0, 0] < 11.0
    assert -0.5 < coef["Slope"][0, 1] < 0.5
    assert -0.5 < coef["Slope"][1, 0] < 0.5
    assert 9.0 < coef["Slope"][1, 1] < 11.0

    coef = rb.fit("u_from_xyi")
    print(coef)

    # test (u, v) from ray at center of the entrance
    assert np.all(np.abs(coef["Intercept"] - np.array([0.11465909, -0.22080707])) < 1.0e-3)
    # now the slope: distortion should be < 2%
    assert np.all(np.abs(1.9e4 * coef["Slope"] + np.identity(2)) < 0.02)
