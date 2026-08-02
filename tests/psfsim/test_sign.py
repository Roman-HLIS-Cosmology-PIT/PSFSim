"""Test for whether focus / parity work the right way."""

import numpy as np
import psfsim.offsets
import psfsim.polychrom
import psfsim.romantrace
from astropy.io import fits  # annoying to put this back every time # noqa: F401
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


class FPAOffsetContext:
    """
    Context manager for changing the position of the FPA.

    Parameters
    ----------
    pdelta : dict
        The parameters to change, and by how much. So for example, ``{"DZ": 0.05}``
        changes the FPA ``DZ`` offset by +0.05 mm.

    """

    def __init__(self, pdelta):
        self.pdelta = pdelta

    def __enter__(self):
        self.orig = {}
        for k in self.pdelta:
            if k in psfsim.offsets.fpa_offset:
                try:
                    u = float(psfsim.offsets.fpa_offset[k])
                except ValueError:
                    continue
                self.orig[k] = u
                psfsim.offsets.fpa_offset[k] += self.pdelta[k]

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k in self.orig:
            psfsim.offsets.fpa_offset[k] = self.orig[k]
        return False


class ShadowContext:
    """
    Context manager for changing the position of the FPA.

    Parameters
    ----------
    shadowtype : str
        The type of shadow to patch. Options are "+x" and "+y".

    """

    def __init__(self, shadowtype):
        self.shadowtype = shadowtype

    def __enter__(self):
        self.old = psfsim.romantrace._shadow
        match self.shadowtype.lower():
            case "+x":

                def _q(im):
                    (ny, nx) = np.shape(im)
                    im[:, nx // 2 :] = 0

                psfsim.romantrace._shadow = _q
            case "+y":

                def _q(im):
                    (ny, nx) = np.shape(im)
                    im[ny // 2 :, :] = 0

                psfsim.romantrace._shadow = _q
            case _:
                raise ValueError("Invalid option.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        psfsim.romantrace._shadow = self.old
        return False


def test_centroid():
    """Tests that the centroid moves in the correct way if we pull the FPA back."""

    # This chooses the far corner of SCA07
    xan, yan = -0.40107, -0.19181
    N = 256
    use_filter = "J"

    # Get original position
    rb = psfsim.romantrace.RomanRayBundle(xan, yan, N, use_filter)
    u = np.mean((rb.u[N // 2 - 1 : N // 2 + 1, N // 2 - 1 : N // 2 + 1]), axis=(0, 1))

    # Now move the FPA.
    # This changes zde in the rotated frame by +0.1 mm, i.e., moves the FPA
    # 0.1 mm *away from* the element wheel.
    with FPAOffsetContext({"DZ": -0.1}):
        xnew = psfsim.romantrace.RomanRayBundle(xan, yan, N, use_filter).x_out
    xorig = psfsim.romantrace.RomanRayBundle(xan, yan, N, use_filter).x_out
    assert np.allclose(xorig, rb.x_out)

    delta = xnew - xorig
    delta_expected = 0.1 * u / np.sqrt(1 - np.sum(u**2))
    assert np.all(np.abs(delta - delta_expected) < 1.0e-4)
    assert np.all(np.abs(delta) > 0.01)


def test_extra_aberrations():
    """Tests that adding extra aberrations has the correct effect."""

    # First, do Z2 and Z3.
    # Convention should be:
    # +Z2 --> shorter path length on exit pupil +X (large u) --> ray moves toward FPA -X (left in image)
    # +Z3 --> shorter path length on exit pupil +Y (large v) --> ray moves toward FPA -Y (down in image)

    # 1.0 microns RMS in Noll convention
    # -> Delta wavefront = 4.0 microns left-to-right
    # -> motion on FPA is 4.0 microns * f = 32 microns = 3.2 native pix = 19.4 oversamp pix
    obj = psfsim.polychrom.PolychromaticPSF(1, -10.22, 10.22, np.linspace(1.131, 1.454, 3))
    psf = obj.compute_poly_psf(
        cycle=10,
        postage_stamp_size=81,
        ovsamp=6,
        use_filter="J",
        extra_aberrations=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    )

    # smooth the PSF
    cpsf = gaussian_filter(psf, sigma=10)

    # this should result in a "right half annulus"
    (yop, xop) = np.where(cpsf > 0.1 * np.amax(cpsf))
    xop2 = np.mean(xop)
    yop2 = np.mean(yop)
    print(xop2, yop2)
    # fits.PrimaryHDU(psf).writeto("test_psf_displace_x.fits", overwrite=True)

    # 1.0 microns RMS in Noll convention
    # -> Delta wavefront = 4.0 microns top-to-bottom
    # -> motion on FPA is 4.0 microns * f = 32 microns = 3.2 native pix = 19.4 oversamp pix
    obj = psfsim.polychrom.PolychromaticPSF(1, -10.22, 10.22, np.linspace(1.131, 1.454, 3))
    psf = obj.compute_poly_psf(
        cycle=10,
        postage_stamp_size=81,
        ovsamp=6,
        use_filter="J",
        extra_aberrations=np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
    )

    # smooth the PSF
    cpsf = gaussian_filter(psf, sigma=10)

    # this should result in a "right half annulus"
    (yop, xop) = np.where(cpsf > 0.1 * np.amax(cpsf))
    xop3 = np.mean(xop)
    yop3 = np.mean(yop)
    print(xop3, yop3)
    # fits.PrimaryHDU(psf).writeto("test_psf_displace_y.fits", overwrite=True)

    assert -22 < xop2 - xop3 < -17
    assert -22 < yop3 - yop2 < -17

    # move the FPA 4 mm closer to the exit pupil.
    # (pretty exaggerated! this leads to a spot 50 native pixels wide)
    # with FPAOffsetContext({"DZ": 4.0}):
    #     obj = psfsim.polychrom.PolychromaticPSF(1, -10.22, 10.22, np.linspace(0.48, 0.76, 4))
    #     psf = obj.compute_poly_psf(cycle=10, postage_stamp_size=81, ovsamp=6, use_filter="R")

    # move the FPA 2 mm closer to the exit pupil *and* shorten the path
    # from the edge by 1.125 microns (so positive OPD in Poppy convention).
    # These should cancel out.
    with FPAOffsetContext({"DZ": 2.0}):
        obj = psfsim.polychrom.PolychromaticPSF(1, -10.22, 10.22, np.linspace(1.131, 1.454, 2))
        psf = obj.compute_poly_psf(
            cycle=10,
            postage_stamp_size=81,
            ovsamp=6,
            use_filter="J",
            extra_aberrations=np.array([0.0, 0.0, 1.125, 0.0, 0.0]),
        )

        # smooth the PSF, get the center
        cpsf = gaussian_filter(psf, sigma=10)
        (yop, xop) = np.where(cpsf > 0.1 * np.amax(cpsf))
        xop0 = np.mean(xop)
        yop0 = np.mean(yop)

        assert 0.006 < np.amax(psf) < 0.008
        # fits.PrimaryHDU(psf).writeto("test_psf_small.fits", overwrite=True)
        #
        # opposite way as a test
        # psf2 = obj.compute_poly_psf(
        #     cycle=10,
        #     postage_stamp_size=81,
        #     ovsamp=6,
        #     use_filter="J",
        #     extra_aberrations=np.array([0.0, 0.0, -1.125, 0.0, 0.0]),
        # )
        # fits.PrimaryHDU(psf2).writeto("test_psf_large.fits", overwrite=True)

    # Test offsets of Z4 and Z5 or Z6
    obj = psfsim.polychrom.PolychromaticPSF(1, -10.22, 10.22, np.linspace(1.131, 1.454, 2))
    psf_z45 = obj.compute_poly_psf(
        cycle=10,
        postage_stamp_size=81,
        ovsamp=6,
        use_filter="J",
        extra_aberrations=np.array([0.0, 0.0, 0.5, 0.5, 0.0]),
    )
    psf_z46 = obj.compute_poly_psf(
        cycle=10,
        postage_stamp_size=81,
        ovsamp=6,
        use_filter="J",
        extra_aberrations=np.array([0.0, 0.0, 0.5, 0.0, 0.5]),
    )
    # fits.PrimaryHDU(psf_z45).writeto("test_psf_z45.fits", overwrite=True)
    # fits.PrimaryHDU(psf_z46).writeto("test_psf_z46.fits", overwrite=True)

    # smooth the PSF, get the center
    cpsf = gaussian_filter(psf_z45, sigma=10)
    (yop, xop) = np.where(cpsf > 0.1 * np.amax(cpsf))
    dx, dy = xop - xop0, yop - yop0
    e1 = np.sum(dx**2 - dy**2) / np.sum(dx**2 + dy**2)
    e2 = np.sum(2 * dx * dy) / np.sum(dx**2 + dy**2)
    assert -0.05 < e1 < 0.05
    assert 0.63 < e2 < 0.73
    cpsf = gaussian_filter(psf_z46, sigma=10)
    (yop, xop) = np.where(cpsf > 0.1 * np.amax(cpsf))
    dx, dy = xop - xop0, yop - yop0
    e1 = np.sum(dx**2 - dy**2) / np.sum(dx**2 + dy**2)
    e2 = np.sum(2 * dx * dy) / np.sum(dx**2 + dy**2)
    assert 0.63 < e1 < 0.73
    assert -0.05 < e2 < 0.05


def test_obstruct():
    """Test obstructing each half of an out-of-focus image."""

    # move the FPA 4 mm closer to the exit pupil.
    # (pretty exaggerated! this leads to a spot 50 native pixels wide)
    # AND block the right half of the entrance pupil
    with ShadowContext("+x"), FPAOffsetContext({"DZ": 4.0}):
        obj = psfsim.polychrom.PolychromaticPSF(1, -10.22, 10.22, np.linspace(1.131, 1.454, 3))
        psf = obj.compute_poly_psf(cycle=10, postage_stamp_size=81, ovsamp=6, use_filter="J")

    # smooth the PSF
    cpsf = gaussian_filter(psf, sigma=10)

    # this should result in a "right half annulus"
    (yop, xop) = np.where(cpsf > 0.1 * np.amax(cpsf))
    xop = np.mean(xop)
    yop = np.mean(yop)
    print(xop, yop)
    assert 282 < xop < 322
    assert 222 < yop < 262

    # move the FPA 4 mm closer to the exit pupil.
    # (pretty exaggerated! this leads to a spot 50 native pixels wide)
    # AND block the right half of the entrance pupil
    with ShadowContext("+y"), FPAOffsetContext({"DZ": 4.0}):
        obj = psfsim.polychrom.PolychromaticPSF(1, -10.22, 10.22, np.linspace(1.131, 1.454, 3))
        psf = obj.compute_poly_psf(cycle=10, postage_stamp_size=81, ovsamp=6, use_filter="J")

    # smooth the PSF
    cpsf = gaussian_filter(psf, sigma=10)
    # fits.PrimaryHDU(psf).writeto("test_psf_blocky.fits", overwrite=True)

    # this should result in a "top half annulus"
    (yop, xop) = np.where(cpsf > 0.1 * np.amax(cpsf))
    xop = np.mean(xop)
    yop = np.mean(yop)
    print(xop, yop)
    assert 222 < xop < 262
    assert 282 < yop < 322


def test_obstruct_pattern():
    """Tests that the obstruction pattern has the right parity if move the FPA toward the element wheel."""

    # move the FPA 4 mm closer to the exit pupil.
    # (pretty exaggerated! this leads to a spot 50 native pixels wide)
    with FPAOffsetContext({"DZ": 4.0}):
        obj = psfsim.polychrom.PolychromaticPSF(1, -10.22, 10.22, np.linspace(0.48, 0.76, 4))
        psf = obj.compute_poly_psf(cycle=10, postage_stamp_size=81, ovsamp=6, use_filter="R")

    # smooth the PSF
    cpsf = gaussian_filter(psf, sigma=10)

    # check that the "maxima" of the inner part of the PSF are at position angles
    # 90, 210, and 330 degrees, measured counterclockwise from the X axis.
    i = np.linspace(0, 59, 60)
    x = 242.0 + 55.0 * np.cos(np.pi * i / 30)
    y = 240.0 + 55.0 * np.sin(np.pi * i / 30)
    xx, yy = np.meshgrid(np.linspace(0, 485, 486), np.linspace(0, 485, 486))
    interp_vals = griddata((xx.ravel(), yy.ravel()), cpsf.ravel(), (x, y), method="linear")
    idx = np.where(
        np.logical_and(interp_vals > np.roll(interp_vals, 1), interp_vals > np.roll(interp_vals, -1))
    )
    assert np.all(np.abs(idx[0] - np.array([15, 35, 55])) < 1.5)

    # fits.PrimaryHDU(cpsf).writeto("test_cpsf.fits", overwrite=True)
