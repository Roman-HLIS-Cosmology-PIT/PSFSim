"""Test functions for psfobject.py."""

import numpy as np
from psfsim.psfobject import PSFObject


def _pt(cycle):
    """
    Test function for PSF object in a single cycle.

    Parameters
    ----------
    cycle : int
        Which model cycle to use.

    Returns
    -------
    None

    """

    n = 8

    obj = PSFObject(
        4,
        20.15,
        5.12,
        wavelength=1.35,
        postage_stamp_size=31,
        ovsamp=n,
        npix_boundary=1,
        use_postage_stamp_size=False,
        add_focus=None,
        cycle=cycle,
    )

    assert np.abs(obj.dx - 10.0 / n) < 1.0e-3

    print(obj.ulen)

    obj.get_optical_psf()
    assert obj.E_FPA_h_polarized.shape == obj.E_FPA_v_polarized.shape
    assert obj.E_FPA_h_polarized.shape == (obj.ulen, obj.ulen, 3)
    assert obj.Optical_PSF.shape == (obj.ulen, obj.ulen)
    assert np.isclose(np.sum(obj.Optical_PSF), 1.0, rtol=1e-12, atol=1e-12)
    assert np.min(obj.Optical_PSF) >= -1e-10

    obj.get_image_from_Intensity()
    assert obj.detector_image.shape == (obj.postage_stamp_size * n, obj.postage_stamp_size * n)
    assert np.all(obj.detector_image >= 0)
    # assert obj.npix_boundary == -1 # <-- used to force failure so we can look at the logs


def test_psfobject():
    """Test function for PSF object for each cycle."""

    for c in [9, 10]:
        _pt(c)
