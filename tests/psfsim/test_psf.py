"""Test functions for psfobject.py."""

import numpy as np
import pytest
from psfsim.basis import basis_set_cy10
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
        use_postage_stamp_size=None,
        extra_aberrations=None,
        cycle=cycle,
    )

    assert np.abs(obj.dx - 10.0 / n) < 1.0e-3

    obj.get_optical_psf()
    assert obj.E_FPA_h_polarized.shape == obj.E_FPA_v_polarized.shape
    assert obj.E_FPA_h_polarized.shape == (obj.ulen, obj.ulen, 3)
    assert obj.Optical_PSF.shape == (obj.ulen, obj.ulen)
    assert np.isclose(np.sum(obj.Optical_PSF), 1.0, rtol=1e-12, atol=1e-12)
    assert np.min(obj.Optical_PSF) >= -1e-10

    obj.get_image_from_Intensity()
    assert obj.detector_image.shape == (obj.postage_stamp_size * n, obj.postage_stamp_size * n)
    assert np.all(obj.detector_image >= 0)


def test_psfobject():
    """Test function for PSF object for each cycle."""

    for c in [9, 10]:
        _pt(c)


def test_psfobject_extra_aberrations():
    """Test function for PSF object with extra aberrations."""

    n = 8

    extra_aberrations = [0.1, 0.2, 0.3, 0.4, 0.5]
    fake_aberrations = [0.5, 0.4, 0.3, 0.2, 0.1, 0.9]

    obj_base = PSFObject(
        4,
        20.15,
        5.12,
        wavelength=1.35,
        postage_stamp_size=31,
        ovsamp=n,
        use_postage_stamp_size=None,
        extra_aberrations=None,
        cycle=10,
    )

    obj = PSFObject(
        4,
        20.15,
        5.12,
        wavelength=1.35,
        postage_stamp_size=31,
        ovsamp=n,
        use_postage_stamp_size=None,
        extra_aberrations=extra_aberrations,
        cycle=10,
    )

    with pytest.raises(ValueError, match="extra_aberrations supports at most 5 coefficients"):
        PSFObject(
            4,
            20.15,
            5.12,
            wavelength=1.35,
            postage_stamp_size=31,
            ovsamp=n,
            use_postage_stamp_size=None,
            extra_aberrations=fake_aberrations,
            cycle=10,
        )
    with pytest.raises(
        ValueError, match="Please increase oversampling factor - can't draw undersampled PSF."
    ):
        PSFObject(
            4,
            20.15,
            5.12,
            wavelength=1.35,
            postage_stamp_size=31,
            ovsamp=1,
            use_postage_stamp_size=96,
            extra_aberrations=None,
            cycle=10,
        )
    assert np.abs(obj.dx - 10.0 / n) < 1.0e-3

    obj.get_optical_psf()
    obj_base.get_optical_psf()
    assert obj.E_FPA_h_polarized.shape == obj.E_FPA_v_polarized.shape
    assert obj.E_FPA_h_polarized.shape == (obj.ulen, obj.ulen, 3)
    assert obj.Optical_PSF.shape == (obj.ulen, obj.ulen)
    assert not np.allclose(obj_base.Optical_PSF, obj.Optical_PSF)


def test_psf_object_override_errs():
    """Test function for PSF object overriding the default errors."""

    obj_base = PSFObject(
        4,
        20.15,
        5.12,
        wavelength=1.35,
        postage_stamp_size=31,
        ovsamp=6,
        use_filter="J",
        cycle=10,
    )

    obj_nopert = PSFObject(
        4,
        20.15,
        5.12,
        wavelength=1.35,
        postage_stamp_size=31,
        ovsamp=6,
        use_filter="J",
        cycle=10,
        perturbations={"arr": np.zeros(basis_set_cy10.N), "basis": basis_set_cy10},
    )

    obj_base.get_optical_psf()
    obj_nopert.get_optical_psf()

    psf1 = obj_base.Optical_PSF
    psf0 = obj_nopert.Optical_PSF
    dpsf = psf1 - psf0
    print(np.amin(dpsf), np.amax(dpsf), dpsf[1025, 1021], dpsf[1020, 1025])

    assert -0.0016 < np.amin(dpsf) < -0.0012
    assert 0.0012 < np.amax(dpsf) < 0.0016
    assert -0.0012 < dpsf[1025, 1021] < -0.0008
    assert 0.0008 < dpsf[1020, 1025] < 0.0012


def test_psf_object_ghost():
    """Test function for PSF object with a ghost."""

    obj = PSFObject(
        4,
        20.15,
        5.12,
        wavelength=1.35,
        postage_stamp_size=31,
        ovsamp=8,
        use_postage_stamp_size=None,
        extra_aberrations=None,
        cycle=10,
        ghost=True,
    )

    assert obj.ghost is True

    obj.get_optical_psf()
    assert obj.E_FPA_h_polarized.shape == obj.E_FPA_v_polarized.shape
    assert obj.E_FPA_h_polarized.shape == (obj.ulen, obj.ulen, 3)
    assert obj.Optical_PSF.shape == (obj.ulen, obj.ulen)
    assert np.isclose(np.sum(obj.Optical_PSF), 1.0, rtol=1e-12, atol=1e-12)
    assert np.min(obj.Optical_PSF) >= -1e-10

    obj.get_image_from_Intensity()
    assert obj.detector_image.shape == (obj.postage_stamp_size * 8, obj.postage_stamp_size * 8)
    assert np.all(obj.detector_image >= 0)


def test_psf_object_ghost_errs():
    """Test function for PSF object with a ghost and ray_trace=False."""

    with pytest.raises(ValueError, match="Ghost path requires ray_trace=True"):
        PSFObject(
            4,
            20.15,
            5.12,
            wavelength=1.35,
            postage_stamp_size=31,
            ovsamp=8,
            use_postage_stamp_size=None,
            extra_aberrations=None,
            cycle=10,
            ghost=True,
            ray_trace=False,
        )
