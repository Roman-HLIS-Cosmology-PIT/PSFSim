"""Test functions for opticspsf.py"""

import numpy as np
from psfsim.opticspsf import GeometricOptics, altgriddata


def test_altgriddata():
    """Test for alternate grid data."""

    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.asarray([3.0, 0.0, 2.0, 4.0, 1.0])
    f = 3.0 * x * y + 0.5 * ((x - 1.0) * (y - 3.0))
    points = np.vstack((x, y)).T
    xi = 1.4
    yi = 3.2
    f_est = altgriddata(points, f, (xi, yi))
    f_good = 3.0 * xi * yi + 0.5 * ((xi - 1.0) * (yi - 3.0))
    assert np.abs(f_est - f_good) < 1e-6


def test_geometricoptics9():
    """Test function for GeometricOptics"""

    g = GeometricOptics(
        4, 20.44, -20.44, wavelength=1.36, ulen=2048, ray_trace=True, pixelsampling=2.0, cycle=9
    )

    print(g.xan, g.yan)
    assert np.hypot(-0.268 - g.xan, -0.126 - g.yan) < 0.003
    print(g.distortionMatrix)
    assert np.all(np.abs(np.identity(2) + g.distortionMatrix / 5.3e-5) < 0.015)

    print("--")

    # Pupil locations
    print(g.samplingwidth)
    print(g.ucen, g.vcen)
    assert np.abs(13056 - g.samplingwidth) < 5.0
    assert np.hypot(-0.12413403996321462 - g.ucen, -0.17128826467513136 - g.vcen) < 0.01

    omega = g.du**2 * np.sum(g.pupil_mask)
    assert np.abs(g.du * 2.0 * 2048 / 1.36 + 1.0) < 0.002
    assert omega > 0.0092
    assert omega < 0.0105

    x = np.where(g.pupil_mask > 0, g.path_difference, np.nan)
    print(np.count_nonzero(g.pupil_mask > 0))
    assert np.shape(x) == (2048, 2048)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    assert iqr > 0.03
    assert iqr < 0.10


def test_geometricoptics10():
    """Test function for GeometricOptics"""

    g = GeometricOptics(
        4, 20.44, -20.44, wavelength=1.36, ulen=2048, ray_trace=True, pixelsampling=2.0, cycle=10
    )

    print(g.xan, g.yan)
    assert np.hypot(-0.268 - g.xan, -0.126 - g.yan) < 0.003
    print(g.distortionMatrix)
    assert np.all(np.abs(np.identity(2) + g.distortionMatrix / 5.3e-5) < 0.015)

    print("--")

    # Pupil locations
    print(g.samplingwidth)
    print(g.ucen, g.vcen)
    assert np.abs(13056 - g.samplingwidth) < 5.0
    assert np.hypot(-0.12413403996321462 - g.ucen, -0.17128826467513136 - g.vcen) < 0.01

    omega = g.du**2 * np.sum(g.pupil_mask)
    assert np.abs(g.du * 2.0 * 2048 / 1.36 + 1.0) < 0.002
    assert omega > 0.0092
    assert omega < 0.0105

    x = np.where(g.pupil_mask > 0, g.path_difference, np.nan)
    print(np.count_nonzero(g.pupil_mask > 0))
    assert np.shape(x) == (2048, 2048)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    assert iqr > 0.03
    assert iqr < 0.20  # includes tip-tilt

    # take out slope
    m = np.zeros((3, 3))
    v = np.zeros((3,))

    c2, c1 = np.mgrid[:2048, :2048]

    v[0] = np.sum(g.pupil_mask * g.path_difference)
    v[1] = np.sum(g.pupil_mask * g.path_difference * c1)
    v[2] = np.sum(g.pupil_mask * g.path_difference * c2)

    m[0, 0] = np.sum(g.pupil_mask)
    m[0, 1] = m[1, 0] = np.sum(g.pupil_mask * c1)
    m[0, 2] = m[2, 0] = np.sum(g.pupil_mask * c2)
    m[1, 1] = np.sum(g.pupil_mask * c1**2)
    m[2, 2] = np.sum(g.pupil_mask * c2**2)
    m[1, 2] = m[2, 1] = np.sum(g.pupil_mask * c1 * c2)
    coefs = np.linalg.solve(m, v)
    resid = x - coefs[1] * c1 - coefs[2] * c2

    iqr = np.nanpercentile(resid, 75) - np.nanpercentile(resid, 25)
    assert iqr > 0.03
    assert iqr < 0.10  # does not include
