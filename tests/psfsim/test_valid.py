"""Ensure that the correct exceptions are raised for invalid values."""

import pytest
from psfsim.opticspsf import GeometricOptics
from psfsim.wfi_coordinate_transformations import from_sca_to_analysis, from_sca_to_fpa


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
