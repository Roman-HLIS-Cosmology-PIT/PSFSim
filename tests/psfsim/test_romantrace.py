import numpy as np
from psfsim import romantrace


def test_romantrace():
    """Test function for romantrace."""
    romantrace.demo(writefiles=False)


def test_lanczos_weight():
    """Test the Lanczos interpolation weight calculation function."""
    # Test at origin
    w_origin = romantrace._lanczos_weight(0.0, 0.0, a=3)
    assert w_origin == 1.0

    # Test outside the 'a' boundary
    w_outside = romantrace._lanczos_weight(3.0, 0.0, a=3)
    assert w_outside == 0.0

    # Test intermediate value using numpy available via romantrace
    w_mid = romantrace._lanczos_weight(0.5, 0.0, a=3)
    expected = np.sinc(0.5) * np.sinc(0.5 / 3.0)
    assert np.isclose(w_mid, expected)


def test_RomanRayBundle_lanczos():
    """Test RomanRayBundle creation utilizing the Lanczos interpolation logic."""
    N = 16
    rb = romantrace.RomanRayBundle(
        xan=0.0, yan=0.0, N=N, usefilter="W", wl=1.29e-3, hasE=True, width=2500.0, ovsamp=3, a_lanczos=2
    )

    # Check resulting property shapes and types
    assert rb.open.shape == (N, N)
    assert rb.x_out.shape == (2,)

    # Verify fractional open logic outputs valid weights within [0.0, 1.0] tolerance
    assert np.all(rb.open >= -1e-10)
    assert np.all(rb.open <= 1.0 + 1e-10)


def test_apply_lanczos_reweighting():
    """Test the application of Lanczos reweighting to a RomanRayBundle."""
    # Create a simple low-res aperture (all ones)
    RB_open_lowres = np.ones((8, 8), dtype=np.float64)

    # Create synthetic high-res data for boundary cells
    num_bdy = 4
    ovsamp = 2
    RB_open_hires = np.ones(num_bdy * ovsamp * ovsamp, dtype=np.float64) * 0.8

    # Define boundary cells (corners of the grid)
    bdycells = (np.array([1, 1, 6, 6]), np.array([1, 6, 1, 6]))

    # Apply reweighting
    a_lanczos = 2
    new_values = romantrace._apply_lanczos_reweighting(
        RB_open_lowres,
        RB_open_hires,
        bdycells,
        ovsamp,
        a_lanczos,
    )

    # Assertions
    assert new_values.shape == (num_bdy,), f"Expected shape ({num_bdy},), got {new_values.shape}"
    assert np.allclose(new_values, 0.8, atol=0.1), "Values should be close to 0.8"
