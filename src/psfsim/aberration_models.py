"""Aberration models."""

from importlib.resources import files

import numpy as np
from astropy.io import fits

from .romantrace import RomanRayBundle
from .zernike import noll_to_zernike, zernike

# parameters
fratio = 8.0


def aberration_gradients(use_filter="W", nn=128, subtract_offset=False, mask=False):
    """
    Build a table of the derivatives of the wavefront error with respect to each surface mode.

    Parameters
    ----------
    use_filter : str, optional
        Which filter to use.
    nn : int, optional
        What size pupil array to use.
    subtract_offset : bool, optional
        Whether to subtract the Z1 mode (which has no effect on the PSF).
    mask : bool, optional
        Whether to null out masked regions.

    Returns
    -------
    np.ndarray
        A table of sensitivities of OPD to each surface error mode. The shape is
        (number of field points, `nn`, `nn`, number of basis modes).

        Reminder that this can be huge!

    """

    fpos = np.loadtxt(files("psfsim.data").joinpath("fpos.dat"))

    for j in range(np.shape(fpos)[0]):
        RB = RomanRayBundle(fpos[j, 0], fpos[j, 1], nn, use_filter, hasE=False, errs={"grad": True})
        if j == 0:
            n = np.shape(RB.grad)[-1]  # get number of basis modes
            g = np.zeros((np.shape(fpos)[0], nn, nn, n))
        if subtract_offset:
            u = RB.open != 0
            for k in range(n):
                mean = np.sum(RB.grad[:, :, k] * u) / np.sum(u)
                RB.grad[:, :, k] -= mean
        if mask:
            for k in range(n):
                RB.grad[:, :, k] *= RB.open > 0.5
        g[j, :, :] = -RB.grad
        # the - sign is because we want OPD in Poppy convention, not path length.

    return g


def aberration_transfer_matrix(use_filter="W", nn=128, n_zernike=22, outdiagnostic=None):
    """
    Computes the transfer matrix from surface distortion modes to Zernikes.

    Parameters
    ----------
    use_filter : str, optional
        Which filter to use.
    nn : int, optional
        What size pupil array to use.
    n_zernike : int, optional
        How many Zernikes to use?
    outdiagnostic : str, optional
        Output FITS file for diagnostics (for debugging only).

    Returns
    -------
    transfer : np.ndarray
        The transfer matrix, shape (number of field points, `n_zernike`, number of basis modes).
    s_decomp : np.ndarray
        The decomposition of the ray-traced design wavefront, shape (number of field points, `n_zernike`).
        The return value is in Poppy OPD sign convention (*negative* of the path length difference).

    """

    if nn % 2 != 0:
        raise ValueError("nn must be even")

    # first get the field positions
    fpos = np.loadtxt(files("psfsim.data").joinpath("fpos.dat"))
    npos = np.shape(fpos)[0]

    for j in range(npos):
        RB = RomanRayBundle(fpos[j, 0], fpos[j, 1], nn, use_filter, hasE=False, errs={"grad": True})
        if j == 0:
            n = np.shape(RB.grad)[-1]  # get number of basis modes
            transfer = np.zeros((npos, n_zernike, n))
            s_decomp = np.zeros((npos, n_zernike))
        for k in range(n):
            mean = np.sum(RB.grad[:, :, k] * RB.open) / np.sum(RB.open)
            RB.grad[:, :, k] -= mean
            opd = -RB.s + np.sum(RB.s * RB.open) / np.sum(RB.open)
            # this flips sign so that OPD is consistent with the Poppy convention
            # (we're not using Poppy here but it is in such common use across the Roman
            # community that I want to be consistent!!)

        # now get the center
        ctr = np.mean(RB.u[nn // 2 - 1 : nn // 2 + 1, nn // 2 - 1 : nn // 2 + 1, :], axis=(0, 1))
        rho12 = 2.0 * fratio * (RB.u - ctr[None, None, :])
        rho = np.hypot(rho12[:, :, 0], rho12[:, :, 1])
        theta = np.arctan2(rho12[:, :, 1], rho12[:, :, 0])
        del rho12

        # and the aberrations
        zmode = np.zeros((nn, nn, n_zernike))
        for q in range(n_zernike):
            _n, _m = noll_to_zernike(q + 1)
            zmode[:, :, q] = zernike(_n, _m, rho, theta, normalized=True)

        # diagnostics, if desired
        if outdiagnostic is not None and j == 0:
            z2 = np.transpose(zmode, axes=(2, 0, 1)) * (RB.open >= 0.5)[None, :, :]
            fits.PrimaryHDU(z2).writeto(outdiagnostic, overwrite=True)
            del z2

        # overlap matrix of the Zernikes --- not zero on an obstructed pupil!
        A = np.zeros((n_zernike, n_zernike))
        B = np.zeros((n_zernike, n))
        c = np.zeros((n_zernike,))
        for za in range(n_zernike):
            for zb in range(za + 1):
                A[za, zb] = A[zb, za] = np.sum(zmode[:, :, za] * zmode[:, :, zb] * RB.open)
            for k in range(n):
                B[za, k] = np.sum(zmode[:, :, za] * RB.grad[:, :, k] * RB.open)
            c[za] = np.sum(opd * zmode[:, :, za] * RB.open)
        s_decomp[j, :] = np.linalg.solve(A, c)
        transfer[j, :, :] = np.linalg.solve(A, B)

    return transfer, -s_decomp


def aberration_transfer_matrix_svd(use_filter="W", nn=128, n_zernike=22):
    """
    Computes the SVD of the transfer matrix from surface distortion modes to Zernikes.

    Parameters
    ----------
    use_filter : str, optional
        Which filter to use.
    nn : int, optional
        What size pupil array to use.
    n_zernike : int, optional
        How many Zernikes to use?
    outdiagnostic : str, optional
        Output FITS file for diagnostics (for debugging only).

    Returns
    -------
    U, S, Vh: np.ndarray
        The SVD of the transfer matrix:

        - `U` has dimension (number of field points * `n_zernike`, nbasis)
        - `S` has length nbasis
        - `Vh` has dimension (nbasis, nbasis)

        The transfer matrix is ``U @ diag(S) @ Vh``.

    """

    T, _ = aberration_transfer_matrix(use_filter=use_filter, nn=nn, n_zernike=n_zernike)
    T = T.reshape((-1, np.shape(T)[-1]))
    (m, n) = np.shape(T)
    if m < n:
        assert ValueError("Too many surface modes to constrain with these Zernikes.")
    U, S, Vh = np.linalg.svd(T, full_matrices=False, compute_uv=True)
    return U, S, Vh


def extract_basis_coefs(infile, use_filter, nn=128, smin=0.01, pars_input=None, flip_y=True, verbose=True):
    """
    Computes basis coefficients from a Zernike file.

    Parameters
    ----------
    infile : str
        The input Zernike file to read. This should be a ``.csv`` file.
    use_filter : str
        The filter to use (1-character string).
    nn : int, optional
        Grid size to use for the Zernikes.
    smin : float, optional
        The minimum singular value to invert.
    pars_input : np.ndarray, optional
        If provided, take this set of instrument parameters as an initial condition.
    flip_y : bool, optional
        Whether to flip the Y-axis of the input Zernikes (useful for WFI-local vs FPA coordinates).
    verbose : bool, optional
        Whether to talk a lot to the output.

    Returns
    -------
    np.ndarray
        A 1D array of the inferred basis coefficients.

    """

    fpos = np.loadtxt(files("psfsim.data").joinpath("fpos.dat"))
    npos = np.shape(fpos)[0]

    # Extract input file and columns
    indata = np.loadtxt(infile, delimiter=",", skiprows=1)
    with open(infile, "r") as f:
        header = f.readline().strip().split(",")
    idx_wl = header.index("wavelength")
    w = indata[:, idx_wl]
    indata = indata[np.where(np.abs(w - np.median(w)) < 0.01), :].reshape((npos, -1))
    w = np.median(w)  # in microns

    nz = 1
    while f"Z{nz + 1:d}" in header:
        nz += 1
    input_zernikes = np.zeros((npos, nz))
    for iz in range(nz):
        input_zernikes[:, iz] = indata[:, header.index(f"Z{iz + 1:d}")]
    input_zernikes *= 0.001 * w  # convert input wavefront to millimeters

    if flip_y:
        # Z3, 5, etc. need to be flipped, so in Python indexing, start with 2
        input_zernikes[:, 2::2] *= -1.0

    T, s_decomp = aberration_transfer_matrix(use_filter=use_filter, nn=nn, n_zernike=nz)
    shape_orig = np.shape(T)
    T = T.reshape((-1, np.shape(T)[-1]))

    # Get the input parameters
    pars = np.zeros(np.shape(T)[-1]) if pars_input is None else np.copy(pars_input)
    if verbose:
        print(np.shape(s_decomp), np.shape(T), np.shape(pars))
    s_decomp += (T @ pars).reshape(np.shape(s_decomp))

    # and now the target change in Zernikes
    delta_zernike = input_zernikes - s_decomp

    if verbose:
        # print error table, excluding Z1-3
        err_design = np.sum(s_decomp[:, 3:] ** 2, axis=1) ** 0.5
        err_input = np.sum(input_zernikes[:, 3:] ** 2, axis=1) ** 0.5
        err_residual = np.sum(delta_zernike[:, 3:] ** 2, axis=1) ** 0.5
        # and without focus
        err_design4 = np.sum(s_decomp[:, 4:] ** 2, axis=1) ** 0.5
        err_input4 = np.sum(input_zernikes[:, 4:] ** 2, axis=1) ** 0.5
        err_residual4 = np.sum(delta_zernike[:, 4:] ** 2, axis=1) ** 0.5
        for ipos in range(npos):
            print(
                f"{ipos:3d} {1.0e6 * err_design[ipos]:8.5f} {1.0e6 * err_input[ipos]:8.5f} "
                f"{1.0e6 * err_residual[ipos]:8.5f}"
                f"   {1.0e6 * err_design4[ipos]:8.5f} {1.0e6 * err_input4[ipos]:8.5f} "
                f"{1.0e6 * err_residual4[ipos]:8.5f}"
            )
        print("rms residual =", (np.sum(delta_zernike[:, 3:] ** 2) / npos) ** 0.5 * 1.0e6, "nm")

    # SVD of the free parameters
    c = 3  # number of modes to skip -- we don't fit the tip-tilt here
    Tfit = T.reshape(shape_orig)[:, c:, :].reshape((-1, shape_orig[-1]))
    m_, n_ = np.shape(T)
    nmin = 0
    nmax = n_
    Tfit = Tfit[:, nmin:nmax]  # shape of the parameters to fit
    m_ -= npos * c
    if m_ < nmax - nmin:
        assert ValueError("Too many surface modes to constrain with these Zernikes.")
    U, S, Vh = np.linalg.svd(Tfit, full_matrices=False, compute_uv=True)

    S_pseudoinv = np.where(smin < S, 1.0 / S, 0.0)
    delta_pars = Vh.T @ np.diag(S_pseudoinv) @ U.T @ delta_zernike[:, c:].ravel()
    if verbose:
        print(f"number of Zernikes used in fit = {m_:d}")
        print(f"number of instrument modes constrained = {nmax - nmin:d}")
        print(f"number of singular values used = {np.count_nonzero(smin < S):d}")
        print("S >>", S)

    pars[nmin:nmax] += delta_pars

    # get new OPDs
    s_decomp += (T[:, nmin:nmax] @ delta_pars).reshape(np.shape(s_decomp))
    delta_zernike = input_zernikes - s_decomp

    if verbose:
        # print error table, excluding Z1-3
        err_design = np.sum(s_decomp[:, 3:] ** 2, axis=1) ** 0.5
        err_input = np.sum(input_zernikes[:, 3:] ** 2, axis=1) ** 0.5
        err_residual = np.sum(delta_zernike[:, 3:] ** 2, axis=1) ** 0.5
        # and without focus
        err_design4 = np.sum(s_decomp[:, 4:] ** 2, axis=1) ** 0.5
        err_input4 = np.sum(input_zernikes[:, 4:] ** 2, axis=1) ** 0.5
        err_residual4 = np.sum(delta_zernike[:, 4:] ** 2, axis=1) ** 0.5
        for ipos in range(npos):
            print(
                f"{ipos:3d} {1.0e6 * err_design[ipos]:8.5f} {1.0e6 * err_input[ipos]:8.5f} "
                f"{1.0e6 * err_residual[ipos]:8.5f}"
                f"   {1.0e6 * err_design4[ipos]:8.5f} {1.0e6 * err_input4[ipos]:8.5f} "
                f"{1.0e6 * err_residual4[ipos]:8.5f}"
            )
        print("rms residual =", (np.sum(delta_zernike[:, 3:] ** 2) / npos) ** 0.5 * 1.0e6, "nm")

    return pars


def display_aberration_gradients(outfile):
    """
    Save aberration gradients as a FITS file.

    Parameters
    ----------
    outfile : str
        Where to write.

    """

    nn = 100
    table = aberration_gradients(nn=nn, subtract_offset=True, mask=True)[::5, :, :, :].astype(np.float32)
    n = np.shape(table)[-1]
    table2 = np.zeros((n, 3 * nn, 6 * nn), dtype=np.float32)
    indx = [2, 2, 2, 1, 1, 1, 0, 0, 0, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    indy = [2, 1, 0] * 6
    for sca in range(18):
        table2[
            :, indy[sca] * nn : (indy[sca] + 1) * nn, indx[sca] * nn : (indx[sca] + 1) * nn
        ] = np.transpose(table[sca], axes=(2, 0, 1))

    fits.PrimaryHDU(table2).writeto(outfile, overwrite=True)
