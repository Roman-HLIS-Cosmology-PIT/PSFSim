"""Detector MTF functions."""

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import erf

# Pixel properties
from .wfi_data import cdpar, nside, pix


def diffusion_green(xd, yd, x2=0.0, y2=0.0):
    """
    Charge diffusion Green's function as a function of Analysis coordinates in microns.

    The MTF is calculated using a three-gaussian approximation of the charge diffusion in the SCA,
    where the parameters are derived from the charge diffusion model
    described in Emily Macbeth's paper and the three-gaussian approximation in https://arxiv.org/pdf/2501.05632.

    Parameters
    ----------
    xd, yd : float or np.ndarray of float
        The coordinates where the Green's function is to be computed.
        Must be the same shape, or broadcastable to the same shape.
    x2, y2 : float, optional
        The location where the charges are generated.

    Returns
    -------
    float or np.ndarray of float
        The Green's function at the indicated positions, same shape as `xd`.

    """

    sigma_s = cdpar["sigma_s"]

    sq = (xd - x2) ** 2 + (yd - y2) ** 2

    term = 0.0
    for j in range(len(cdpar["w"])):
        sigmaj = cdpar["c"][j] * sigma_s
        term += cdpar["w"][j] * np.exp(-(sq / (2.0 * sigmaj**2))) * (1.0 / (2.0 * np.pi * sigmaj**2))

    return term


def diffusion_prob(xd, yd, width=10.0, x2=0.0, y2=0.0):
    """
    Charge diffusion probability as a function of Analysis coordinates in microns.

    This is the integral of the Green's function over a square of width `width` centered on (0, 0).

    The MTF is calculated using a three-gaussian approximation of the charge diffusion in the SCA,
    where the parameters are derived from the charge diffusion model
    described in Emily Macbeth's paper and the three-gaussian approximation in https://arxiv.org/pdf/2501.05632.

    Parameters
    ----------
    xd, yd : float or np.ndarray of float
        The coordinates where the Green's function is to be computed.
        Must be the same shape, or broadcastable to the same shape.
    width : float, int
        The width of the integration zone in microns (on each axis).
    x2, y2 : float, optional
        The location where the charges are generated.

    Returns
    -------
    float or np.ndarray of float
        The integrated probability centered at the indicated positions, same shape as `xd`.

    See Also
    --------
    diffusion_green
        The probability density function.

    """

    sigma_s = cdpar["sigma_s"]

    cut_xm = xd - x2 - width / 2.0
    cut_xp = xd - x2 + width / 2.0
    cut_ym = yd - y2 - width / 2.0
    cut_yp = yd - y2 + width / 2.0

    term = 0.0
    for j in range(len(cdpar["w"])):
        sigmaj2 = cdpar["c"][j] * sigma_s * np.sqrt(2.0)
        term += (
            0.25
            * cdpar["w"][j]
            * (erf(cut_xp / sigmaj2) - erf(cut_xm / sigmaj2))
            * (erf(cut_yp / sigmaj2) - erf(cut_ym / sigmaj2))
        )

    return term


def intensity_to_image(intensity, x_in, y_in, x_out, y_out, n_out, dx, reflect=True, tophat=True):
    """
    Function to convolve an intensity image with the Green's function.

    Parameters
    ----------
    intensity : np.ndarray of float
        The intensity image.
    x_in, y_in : float
        The center of the intensity input image in microns.
    x_out, y_out: float
        The center of the desired output image in microns.
    n_out : int
        The desired output postage stamp size.
    dx : float
        The grid spacing in microns.
    reflect : bool, optional
        Reflect at the detector active region edge.
    tophat : bool, optional
        Integrate over the pixel tophat. Note that if this option is chosen, the unit of
        the output is the unit of the intensity times micron^2.

    Returns
    -------
    np.ndarray of float
        The output image.

    """

    (ny_in, nx_in) = np.shape(intensity)

    # List of positions to reflect.
    pos = [(0, 0)]
    if reflect:
        x_sgn = 1 if x_in > 0 else -1
        y_sgn = 1 if y_in > 0 else -1
        pos = [(0, 0), (x_sgn, 0), (0, y_sgn), (x_sgn, y_sgn)]
    hds = pix * nside / 2.0  # coordinate of side

    # Sum over reflected images
    im_out = np.zeros((n_out, n_out))
    for p in pos:
        ii = np.copy(intensity)
        if p[0] % 2 != 0:
            ii = np.fliplr(ii)
        if p[1] % 2 != 0:
            ii = np.flipud(ii)
        x_refl = p[0] * hds + (-1) ** p[0] * (x_in - p[0] * hds)
        y_refl = p[1] * hds + (-1) ** p[1] * (y_in - p[1] * hds)

        # Now do the convolution
        delta_x_ctr = x_out - x_refl
        delta_y_ctr = y_out - y_refl
        nyvals = ny_in + n_out - 1
        nxvals = nx_in + n_out - 1
        dyvals = np.linspace(
            delta_y_ctr - dx * (nyvals - 1.0) / 2.0, delta_y_ctr + dx * (nyvals - 1.0) / 2.0, nyvals
        )
        dxvals = np.linspace(
            delta_x_ctr - dx * (nxvals - 1.0) / 2.0, delta_x_ctr + dx * (nxvals - 1.0) / 2.0, nxvals
        )
        if tophat:
            g_offset = diffusion_prob(dxvals[None, :], dyvals[:, None], width=pix)
        else:
            g_offset = diffusion_green(dxvals[None, :], dyvals[:, None])
        im_out[:, :] += fftconvolve(g_offset, ii, mode="valid")

    return im_out
