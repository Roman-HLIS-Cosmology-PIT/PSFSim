"""Ensure that the correct exceptions are raised for invalid values."""

import numpy as np
import pytest
from psfsim.aberration_models import aberration_transfer_matrix
from psfsim.opticspsf import GeometricOptics
from psfsim.polychrom import PolychromaticPSF
from psfsim.wfi_coordinate_transformations import from_sca_to_analysis, from_sca_to_fpa
from psfsim.zernike import zernike, zernike_radial


def test_scaerr():
    """Invalid SCA numbers."""

    for f in [from_sca_to_fpa, from_sca_to_analysis]:
        for sca in [0, 19]:
            with pytest.raises(ValueError):
                f(sca, 0.0, 0.0)


def test_cycleerr():
    """Invalid cycle number."""

    with pytest.raises(ValueError):
        GeometricOptics(4, 1.5, -4.0, cycle=1).path_diff()


def test_zernikeerr():
    """Invalid indices."""

    r = zernike_radial(2, 1, np.linspace(0, 1, 5))
    assert np.size(r) == 5
    assert np.all(r == 0)

    with pytest.raises(ValueError):
        zernike(3, 5, np.linspace(0, 1, 5), np.zeros(5))


def test_aberr_odd():
    """Test that the correct error is raised for odd values."""

    with pytest.raises(ValueError):
        aberration_transfer_matrix(nn=25)


def test_polychrom_noframe():
    """Test that an error is raised for an invalid frame."""

    with pytest.raises(ValueError):
        PolychromaticPSF(7, 0.0, 0.0, np.array([1.4, 1.6]), frame="invalid")
