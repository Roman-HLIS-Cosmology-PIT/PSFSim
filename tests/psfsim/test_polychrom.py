"""Tests for polychrom.py."""

import numpy as np
import pytest

import psfsim.polychrom as polychrom


class DummyPSFObject:
    """Minimal PSFObject stub for fast PolychromaticPSF testing."""

    def __init__(
        self,
        scanum,
        scax,
        scay,
        wavelength,
        postage_stamp_size=31,
        ovsamp=10,
        **kwargs,
    ):
        self.scanum = scanum
        self.scax = scax
        self.scay = scay
        self.wavelength = wavelength
        self.postage_stamp_size = postage_stamp_size
        self.ovsamp = ovsamp

    def get_optical_psf(self):
        """Populate a deterministic optical PSF image."""
        self.Optical_PSF = np.ones((2048, 2048), dtype=np.float32)

    def get_image_from_Intensity(self, centerpix=True, reflect=True, tophat=True):
        """Populate a deterministic detector image."""
        image_size = self.postage_stamp_size * self.ovsamp
        self.detector_image = np.ones((image_size, image_size), dtype=np.float32)


@pytest.mark.parametrize(
    ("optical_psf_only", "expected_shape"),
    [
        (False, (6, 6)),
        (True, (2048, 2048)),
    ],
)
def test_compute_poly_psf_shape_and_normalization(monkeypatch, optical_psf_only, expected_shape):
    """PolychromaticPSF output has expected shape and unit normalization."""
    monkeypatch.setattr(polychrom, "PSFObject", DummyPSFObject)

    model = polychrom.PolychromaticPSF(
        scanum=1,
        scax=0.0,
        scay=0.0,
        wavelengths=np.array([1.5, 1.6]),
    )
    image = model.compute_poly_psf(
        postage_stamp_size=3,
        ovsamp=2,
        use_filter="H",
        optical_psf_only=optical_psf_only,
    )

    assert image.shape == expected_shape
    assert np.isclose(np.sum(image), 1.0, rtol=1e-12, atol=1e-12)


def test_compute_poly_psf_raises_when_all_wavelengths_out_of_band(monkeypatch):
    """PolychromaticPSF raises ValueError when no wavelength contributes flux."""
    monkeypatch.setattr(polychrom, "PSFObject", DummyPSFObject)

    model = polychrom.PolychromaticPSF(
        scanum=1,
        scax=0.0,
        scay=0.0,
        wavelengths=np.array([0.4, 0.5]),
    )

    with pytest.raises(ValueError, match="No flux accumulated in polychromatic PSF"):
        model.compute_poly_psf(use_filter="H")
