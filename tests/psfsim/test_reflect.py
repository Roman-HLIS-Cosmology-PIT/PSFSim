"""Test that the reflections due to Neumann boundary conditions are correctly placed at the SCA edges."""

import galsim
import numpy as np
from psfsim.mtf_diffusion import intensity_to_image
from psfsim.polychrom import PolychromaticPSF


def test_neumann():
    """Test that the Neumann boundary conditions place mirror PSFs in the right place."""

    positions = [(2300.0, 4073.25), (4073.75, 1910.0), (433.7, 13.5), (12.0, 3000.1)]

    # postage stamp size and oversampling
    n = 81
    ov = 4
    sca = 16

    psfs = []
    for pos in positions:
        (x, y) = pos
        psfs.append(
            PolychromaticPSF(sca, x, y, np.array([1.58]), frame="science").compute_poly_psf(
                postage_stamp_size=n, cycle=10, ovsamp=ov, use_filter="H", centerpix=False
            )
        )

    # Now analyze the PSFs

    # Case 0:
    # center of the PSF is at (161.5, 161.5) ...
    moms0 = galsim.hsm.FindAdaptiveMom(galsim.Image(psfs[0][152:172, 152:172]))
    print(moms0)
    assert 0.3 < moms0.moments_amp < 0.4
    assert np.hypot(moms0.moments_centroid.x - 10.5, moms0.moments_centroid.y - 10.5) < 1.0
    #
    # but the 'reflection' above should be at the boundary y = 4087.5 so the position relative to the center
    # is
    # dy = 2 * (4087.5 - 4073.25) = 28.5 native pixels = 114 oversampled pixels -> @ (161.5, 275.5)
    moms0 = galsim.hsm.FindAdaptiveMom(galsim.Image(psfs[0][266:286, 152:172]))
    print(moms0)
    assert 0.3 < moms0.moments_amp < 0.4
    assert np.hypot(moms0.moments_centroid.x - 10.5, moms0.moments_centroid.y - 10.5) < 1.0
    #
    # shouldn't be anything the other way
    assert np.all(np.abs(psfs[0][38:58, 152:172]) < 1.0e-6)

    # Case 1:
    # center of the PSF is at (161.5, 161.5) ...
    moms0 = galsim.hsm.FindAdaptiveMom(galsim.Image(psfs[1][152:172, 152:172]))
    print(moms0)
    assert 0.3 < moms0.moments_amp < 0.4
    assert np.hypot(moms0.moments_centroid.x - 10.5, moms0.moments_centroid.y - 10.5) < 1.0
    #
    # but the 'reflection' to the right should be at the boundary x = 4087.5 so the position relative to the
    # center is
    # dx = 2 * (4087.5 - 4073.75) = 27.5 native pixels = 110 oversampled pixels -> @ (271.5, 161.5)
    moms0 = galsim.hsm.FindAdaptiveMom(galsim.Image(psfs[1][152:172, 262:282]))
    print(moms0)
    assert 0.3 < moms0.moments_amp < 0.4
    assert np.hypot(moms0.moments_centroid.x - 10.5, moms0.moments_centroid.y - 10.5) < 1.0
    #
    # shouldn't be anything the other way
    assert np.all(np.abs(psfs[1][152:172, 42:62]) < 1.0e-6)

    # Case 2:
    # center of the PSF is at (161.5, 161.5) ...
    moms0 = galsim.hsm.FindAdaptiveMom(galsim.Image(psfs[2][152:172, 152:172]))
    print(moms0)
    assert 0.3 < moms0.moments_amp < 0.4
    assert np.hypot(moms0.moments_centroid.x - 10.5, moms0.moments_centroid.y - 10.5) < 1.0
    #
    # but the 'reflection' below should be at the boundary y = -0.5 so the position relative to the center is
    # dy = 2 * (-0.5 - 13.5) = -28.0 native pixels = -112 oversampled pixels -> @ (161.5, 49.5)
    moms0 = galsim.hsm.FindAdaptiveMom(galsim.Image(psfs[2][40:60, 152:172]))
    print(moms0)
    assert 0.3 < moms0.moments_amp < 0.4
    assert np.hypot(moms0.moments_centroid.x - 10.5, moms0.moments_centroid.y - 10.5) < 1.0
    #
    # shouldn't be anything the other way
    assert np.all(np.abs(psfs[2][264:284, 152:172]) < 1.0e-6)

    # Case 3:
    # center of the PSF is at (161.5, 161.5) ...
    moms0 = galsim.hsm.FindAdaptiveMom(galsim.Image(psfs[3][152:172, 152:172]))
    print(moms0)
    assert 0.3 < moms0.moments_amp < 0.4
    assert np.hypot(moms0.moments_centroid.x - 10.5, moms0.moments_centroid.y - 10.5) < 1.0
    #
    # but the 'reflection' to the left should be at the boundary x = -0.5 so the position relative to the
    # center is
    # dx = 2 * (-0.5 - 12.0) = -25.0 native pixels = -100 oversampled pixels -> @ (61.5, 161.5)
    moms0 = galsim.hsm.FindAdaptiveMom(galsim.Image(psfs[3][152:172, 52:72]))
    print(moms0)
    assert 0.3 < moms0.moments_amp < 0.4
    assert np.hypot(moms0.moments_centroid.x - 10.5, moms0.moments_centroid.y - 10.5) < 1.0
    #
    # shouldn't be anything the other way
    assert np.all(np.abs(psfs[3][152:172, 252:272]) < 1.0e-6)


def test_clip():
    """Tests for clipping in intensity_to_image."""

    # make a simple Gaussian image
    sig = 4.0
    sig2 = np.hypot(sig, 3.2)
    _s = np.linspace(-20, 20, 21)
    _x, _y = np.meshgrid(_s, _s)
    intensity = np.exp(-0.5 * (_x**2 + _y**2) / sig**2)
    n = 30
    outmap = np.zeros((9, n, n))
    _so = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * (_s[1] - _s[0])
    _xo, _yo = np.meshgrid(_so, _so)

    for j in range(9):
        x_in = 20430.0 * (j % 3 - 1)
        y_in = 20430.0 * (j // 3 - 1)
        outmap[j, :, :] = intensity_to_image(
            intensity, x_in, y_in, x_in, y_in, n, _s[1] - _s[0], reflect=True, tophat=False
        )

        im_target = np.zeros((n, n))

        factor = sig**2 / sig2**2 / (_s[1] - _s[0]) ** 2
        wx = np.exp(-0.5 * _xo**2 / sig2**2)
        if j % 3 == 0:
            wx += np.exp(-0.5 * (_xo + 20) ** 2 / sig2**2)
        if j % 3 == 2:
            wx += np.exp(-0.5 * (_xo - 20) ** 2 / sig2**2)
        wy = np.exp(-0.5 * _yo**2 / sig2**2)
        if j // 3 == 0:
            wy += np.exp(-0.5 * (_yo + 20) ** 2 / sig2**2)
        if j // 3 == 2:
            wy += np.exp(-0.5 * (_yo - 20) ** 2 / sig2**2)
        im_target = factor * wx * wy
        assert np.sum((outmap[j] - im_target) ** 2) / np.sum(im_target**2) < 0.01

    for j in range(9):
        x_in = 20450.0 * (j % 3 - 1)
        y_in = 20450.0 * (j // 3 - 1)
        outmap[j, :, :] = intensity_to_image(
            intensity, x_in, y_in, x_in, y_in, n, _s[1] - _s[0], reflect=True, tophat=False
        )

    sum = np.sum(outmap, axis=(1, 2))
    ctr_x = np.sum(outmap * _xo[None, ...], axis=(1, 2)) / sum
    ctr_y = np.sum(outmap * _yo[None, ...], axis=(1, 2)) / sum
    print(sum)
    print(ctr_x)
    print(ctr_y)

    sa = 1.27046717e-03
    sb = 8.93446439e-02
    sc = 6.28309457e00
    assert np.all(np.log(sum / np.array([sa, sb, sa, sb, sc, sb, sa, sb, sa])) < 0.05)
    assert np.all(np.abs(ctr_x - np.array([10, 0, -10, 10, 0, -10, 10, 0, -10])) < 0.01)
    assert np.all(np.abs(ctr_y - np.array([10, 10, 10, 0, 0, 0, -10, -10, -10])) < 0.01)
