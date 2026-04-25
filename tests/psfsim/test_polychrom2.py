"""Tests for polychromatic PSF wavelength integration behavior."""

import numpy as np
import psfsim.polychrom


def test_poly_h():
    """Simple H-band test."""

    # This will go out of the bandpass, and since req_in_band is True
    # by default the final wavelengths don't get used.
    p = psfsim.polychrom.PolychromaticPSF(6, 12.1, -2.2, np.linspace(1.4, 1.9, 6))
    arr = p.compute_poly_psf(use_filter="H", ovsamp=8)

    # These are to alert us to things that change.
    # If you do a big enough model update, they might fail,
    # and you should assess whether the change is reasonable.
    assert 0.003 <= np.amax(arr) / np.sum(arr) <= 0.004

    # check that the diffraction spikes are there
    # draw a circle around the PSF and measure the Fourier modes in longitude
    N = 64
    sc = np.zeros(N)
    for i in range(N):
        x = 123.5 + 100.0 * np.cos(i * np.pi * 2.0 / N)
        y = 123.5 + 100.0 * np.sin(i * np.pi * 2.0 / N)
        xi = int(np.floor(x))
        yi = int(np.floor(y))
        xf = x - xi
        yf = y - yi
        sc[i] = arr[yi, xi] * (1 - xf) * (1 - yf) + arr[yi + 1, xi] * (1 - xf) * yf
        sc[i] += arr[yi, xi + 1] * xf * (1 - yf) + arr[yi + 1, xi + 1] * xf * yf
    sc /= np.mean(sc)
    scft = np.fft.ifft(sc)
    # for i in range(N//2+1): print(f"{scft[i].real:8.5f} {scft[i].imag:8.5f} {np.abs(scft[i]):8.5f}")

    # expect lots of m=6 and m=12, less of the others
    assert np.abs(scft[4]) < 0.075
    assert np.abs(scft[5]) < 0.075
    assert 0.1 < np.abs(scft[6]) < 0.2
    assert np.abs(scft[7]) < 0.075
    assert np.abs(scft[8]) < 0.075
    assert np.abs(scft[11]) < 0.075
    assert 0.1 < np.abs(scft[12]) < 0.2
    assert np.abs(scft[13]) < 0.075
