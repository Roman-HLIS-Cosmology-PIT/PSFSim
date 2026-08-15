"""Test that the reflections due to Neumann boundary conditions are correctly placed at the SCA edges."""

import galsim
import numpy as np
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
