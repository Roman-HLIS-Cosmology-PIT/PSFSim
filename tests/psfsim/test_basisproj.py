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

    # check the aberration transfer matrices
    od = str(tmp_path) + "/zernike_corner0.fits"
    t, svec = aberration_transfer_matrix(use_filter="W", nn=128, n_zernike=22, outdiagnostic=od)
    with fits.open(od) as f:
        n = np.shape(f[0].data)[-1]
        s = np.linspace(-0.5 * (1 - 1 / n), 0.5 * (1 - 1 / n), n) * 2500.0 / 1184.02
        x, y = np.meshgrid(s, s)
        im = np.sqrt(8) * (3 * (x**2 + y**2) - 2) * x
        im2 = f[0].data[7, :, :]
        err = np.where(np.abs(im2) > 1.0e-24, im, 0.0) - im2
        assert np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(im2**2)) < 0.1

    # save and test these files
    fits.HDUList([fits.PrimaryHDU(t), fits.ImageHDU(svec)]).writeto(
        str(tmp_path) + "/transfer.fits", overwrite=True
    )
    with fits.open(str(tmp_path) + "/transfer.fits") as ft:
        target = np.array([[1.0, np.sqrt(3)], [np.sqrt(3), -1.0]])
        assert np.amax(np.abs(ft[0].data[:, 4:6, 1:3] - target)) < 0.2

    # check the SVD functions
    U, S, Vh = aberration_transfer_matrix_svd(use_filter="W", nn=128, n_zernike=22)
    fits.HDUList([fits.PrimaryHDU(U), fits.ImageHDU(S), fits.ImageHDU(Vh)]).writeto(
        str(tmp_path) + "/transfer_svd.fits", overwrite=True
    )
    assert np.allclose(t.reshape((-1, np.shape(Vh)[-1])), U @ np.diag(S) @ Vh)

    # Now check that the pictures are right
    display_aberration_gradients(str(tmp_path) + "/grad.fits")
    with fits.open(str(tmp_path) + "/grad.fits") as f:
        data = f[0].data
    assert np.shape(data[0]) == (375, 600)
    im_sca5 = data[:, 125:225, 100:200]
    s = np.linspace(-0.495, 0.495, 100) * 2500.0 / 1184.02
    x, y = np.meshgrid(s, s)
    xm = (-np.sqrt(3) * x + y) / 2.0
    ym = (np.sqrt(3) * y + x) / 2.0
    for iz in range(3):
        print(np.count_nonzero(im_sca5[iz]))
        assert 5670 <= np.count_nonzero(im_sca5[iz]) <= 5680
    print(im_sca5[0, ::10, ::10])
    test1 = np.where(im_sca5[0] == 0, 0.0, 4 * np.sqrt(3) * (xm**2 + ym**2 - 0.5) + im_sca5[0])
    assert np.all(np.abs(test1) < 0.75)
    test1 = np.where(im_sca5[1] == 0, 0.0, 2 * np.sqrt(6) * (2 * xm * ym) + im_sca5[1])
    assert np.all(np.abs(test1) < 0.75)
    test1 = np.where(im_sca5[2] == 0, 0.0, 2 * np.sqrt(6) * (xm**2 - ym**2) + im_sca5[2])
    assert np.all(np.abs(test1) < 0.75)
