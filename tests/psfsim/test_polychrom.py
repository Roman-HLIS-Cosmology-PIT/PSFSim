"""Tests for polychromatic PSF wavelength integration behavior."""

import importlib
import sys
import types

import numpy as np
import pytest


class _FakeBandpass:
    def __init__(self, blue_limit_nm, red_limit_nm, response=1.0):
        self.blue_limit = blue_limit_nm
        self.red_limit = red_limit_nm
        self._response = response

    def __call__(self, wav_nm):
        if np.isscalar(self._response):
            return float(self._response)
        return float(self._response(wav_nm))


class _FakePSFObject:
    def __init__(self, scanum, scax, scay, wavelength, **kwargs):
        self.wavelength = wavelength
        self.postage_stamp_size = kwargs["postage_stamp_size"]
        self.ovsamp = kwargs["ovsamp"]

    def get_optical_psf(self):
        # Keep shape compatible with optical-only mode, though these tests target detector branch.
        self.Optical_PSF = np.full((2048, 2048), self.wavelength, dtype=float)

    def get_image_from_Intensity(self, centerpix=True, reflect=True, tophat=True):
        side = self.postage_stamp_size * self.ovsamp
        image = np.zeros((side, side), dtype=float)
        image[0, 0] = 1.0
        image[-1, -1] = self.wavelength**2
        self.detector_image = image


@pytest.fixture
def patch_poly_deps(monkeypatch):
    """Load polychrom with stubs and provide helpers for bandpass-aware tests."""

    fake_psfobject_module = types.ModuleType("psfsim.psfobject")
    fake_psfobject_module.PSFObject = _FakePSFObject
    monkeypatch.setitem(sys.modules, "psfsim.psfobject", fake_psfobject_module)

    sys.modules.pop("psfsim.polychrom", None)
    polychrom = importlib.import_module("psfsim.polychrom")
    original_in_bandpass = polychrom.inBandpass

    def _set_bandpass(blue_nm=1000.0, red_nm=2000.0, response=1.0, adapt_compute=True):
        fake_bp = {"H": _FakeBandpass(blue_nm, red_nm, response=response)}
        monkeypatch.setattr(polychrom.galsim.roman, "getBandpasses", lambda: fake_bp)
        if adapt_compute:
            # Compatibility adapter while compute_poly_psf still calls inBandpass(wav, filter).
            monkeypatch.setattr(
                polychrom,
                "inBandpass",
                lambda wav, filter_string: original_in_bandpass(wav, filter_string, fake_bp),
            )
        else:
            monkeypatch.setattr(polychrom, "inBandpass", original_in_bandpass)
        return fake_bp

    return polychrom, _set_bandpass


def test_inbandpass_accepts_bandpasses_argument_exact_key(patch_poly_deps):
    """inBandpass should accept explicit bandpasses and honor exact key matches."""
    polychrom, set_bandpass = patch_poly_deps
    fake_bp = set_bandpass(blue_nm=1000.0, red_nm=1500.0, response=1.0, adapt_compute=False)

    is_in, key = polychrom.inBandpass(1.2, "H", fake_bp)
    assert is_in is True
    assert key == "H"


def test_inbandpass_accepts_bandpasses_argument_substring_key(patch_poly_deps):
    """inBandpass should find a matching substring key from provided bandpasses."""
    polychrom, set_bandpass = patch_poly_deps
    set_bandpass(blue_nm=1000.0, red_nm=1500.0, response=1.0, adapt_compute=False)

    fake_bp = {
        "H158": _FakeBandpass(1400.0, 1700.0, response=1.0),
        "J129": _FakeBandpass(1100.0, 1400.0, response=1.0),
    }
    is_in, key = polychrom.inBandpass(1.58, "H", fake_bp)
    assert is_in is True
    assert key == "H158"


def test_inbandpass_raises_for_missing_filter_in_provided_bandpasses(patch_poly_deps):
    """inBandpass should raise if requested filter is missing from explicit bandpasses."""
    polychrom, set_bandpass = patch_poly_deps
    set_bandpass(blue_nm=1000.0, red_nm=1500.0, response=1.0, adapt_compute=False)

    fake_bp = {"J129": _FakeBandpass(1100.0, 1400.0, response=1.0)}
    with pytest.raises(ValueError, match="not found in bandpasses"):
        polychrom.inBandpass(1.25, "H", fake_bp)


def test_single_in_band_node_returns_monochromatic(patch_poly_deps):
    """A single in-band node should return the corresponding monochromatic PSF."""
    polychrom, set_bandpass = patch_poly_deps
    set_bandpass(blue_nm=1000.0, red_nm=1400.0, response=1.0)

    p = polychrom.PolychromaticPSF(
        scanum=1,
        scax=0.0,
        scay=0.0,
        wavelengths=[2.5, 1.2, 0.9],  # Only 1.2 micron is in-band
        sed=lambda wav: 99.0,
    )

    out = p.compute_poly_psf(postage_stamp_size=1, ovsamp=2, use_filter="H")

    expected = np.array([[1.0, 0.0], [0.0, 1.2**2]], dtype=float)
    expected /= expected.sum()
    assert np.allclose(out, expected)


def test_zero_in_band_nodes_raises(patch_poly_deps):
    """If no nodes are in-band, integration should raise a clear error."""
    polychrom, set_bandpass = patch_poly_deps
    set_bandpass(blue_nm=1000.0, red_nm=1200.0, response=1.0)

    p = polychrom.PolychromaticPSF(scanum=1, scax=0.0, scay=0.0, wavelengths=[0.5, 2.0])

    with pytest.raises(ValueError, match="No in-band wavelength nodes"):
        p.compute_poly_psf(postage_stamp_size=1, ovsamp=2, use_filter="H")


def test_nonuniform_nodes_change_trapezoid_result(patch_poly_deps):
    """Different node spacing should change the trapezoid-weighted integrated PSF."""
    polychrom, set_bandpass = patch_poly_deps
    set_bandpass(blue_nm=1000.0, red_nm=2000.0, response=1.0)

    p_uniform = polychrom.PolychromaticPSF(scanum=1, scax=0.0, scay=0.0, wavelengths=[1.0, 1.5, 2.0])
    p_nonuniform = polychrom.PolychromaticPSF(scanum=1, scax=0.0, scay=0.0, wavelengths=[1.0, 1.25, 2.0])

    out_uniform = p_uniform.compute_poly_psf(postage_stamp_size=1, ovsamp=2, use_filter="H")
    out_nonuniform = p_nonuniform.compute_poly_psf(postage_stamp_size=1, ovsamp=2, use_filter="H")

    assert not np.allclose(out_uniform, out_nonuniform)
    assert np.isclose(out_uniform.sum(), 1.0)
    assert np.isclose(out_nonuniform.sum(), 1.0)


def test_sed_callable_changes_integrated_result(patch_poly_deps):
    """A non-flat SED callable should alter the integrated PSF relative to flat SED."""
    polychrom, set_bandpass = patch_poly_deps
    set_bandpass(blue_nm=1000.0, red_nm=2000.0, response=1.0)

    wavelengths = [1.0, 1.5, 2.0]
    p_flat = polychrom.PolychromaticPSF(scanum=1, scax=0.0, scay=0.0, wavelengths=wavelengths, sed=None)
    p_tilted = polychrom.PolychromaticPSF(
        scanum=1,
        scax=0.0,
        scay=0.0,
        wavelengths=wavelengths,
        sed=lambda wav: wav,
    )

    out_flat = p_flat.compute_poly_psf(postage_stamp_size=1, ovsamp=2, use_filter="H")
    out_tilted = p_tilted.compute_poly_psf(postage_stamp_size=1, ovsamp=2, use_filter="H")

    assert not np.allclose(out_flat, out_tilted)
    assert np.isclose(out_flat.sum(), 1.0)
    assert np.isclose(out_tilted.sum(), 1.0)


def test_wavelengths_are_sorted_internally(patch_poly_deps):
    """Integration should be invariant to input wavelength ordering."""
    polychrom, set_bandpass = patch_poly_deps
    set_bandpass(blue_nm=1000.0, red_nm=2000.0, response=1.0)

    unsorted_nodes = [2.0, 1.0, 1.5]
    sorted_nodes = sorted(unsorted_nodes)

    p_unsorted = polychrom.PolychromaticPSF(scanum=1, scax=0.0, scay=0.0, wavelengths=unsorted_nodes)
    p_sorted = polychrom.PolychromaticPSF(scanum=1, scax=0.0, scay=0.0, wavelengths=sorted_nodes)

    out_unsorted = p_unsorted.compute_poly_psf(postage_stamp_size=1, ovsamp=2, use_filter="H")
    out_sorted = p_sorted.compute_poly_psf(postage_stamp_size=1, ovsamp=2, use_filter="H")

    assert np.allclose(out_unsorted, out_sorted)
