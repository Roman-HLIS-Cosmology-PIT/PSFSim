"""Test for basis function-based surface errors."""

import numpy as np
from astropy.io import fits
from psfsim.aberration_models import (
    aberration_transfer_matrix,
    aberration_transfer_matrix_svd,
    display_aberration_gradients,
)
from psfsim.romantrace import RomanRayBundle


def test_grad(tmp_path):
    """Gradient test."""

    RB = RomanRayBundle(-0.399, 0.208, 128, "W", wl=9.27e-4, hasE=True, errs={"grad": True})
    assert np.shape(RB.grad)[:2] == (128, 128)

    # Right now, checks that these functions run --- still need to insert tests that they were *correct*!

    display_aberration_gradients(str(tmp_path) + "/grad.fits")

    od = str(tmp_path) + "/zernike_corner0.fits"
    t, svec = aberration_transfer_matrix(use_filter="W", nn=128, n_zernike=22, outdiagnostic=od)
    fits.HDUList([fits.PrimaryHDU(t), fits.ImageHDU(svec)]).writeto(
        str(tmp_path) + "/transfer.fits", overwrite=True
    )

    U, S, Vh = aberration_transfer_matrix_svd(use_filter="W", nn=128, n_zernike=22)
    fits.HDUList([fits.PrimaryHDU(U), fits.ImageHDU(S), fits.ImageHDU(Vh)]).writeto(
        str(tmp_path) + "/transfer_svd.fits", overwrite=True
    )
