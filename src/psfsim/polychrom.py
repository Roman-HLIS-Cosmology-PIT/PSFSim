"""Functions for polychromatic PSFs."""

import galsim.roman
import numpy as np

from .psfobject import PSFObject


def inBandpass(wav, filter_string, bandpasses):
    """
    Compute whether a wavelength is in a filter.

    Parameters
    ----------
    wav : float
        Wavelength in microns
    filter_string: str
        String to identify the filter. Can either be just a letter or letter + wl (e.g. 'H' or 'H158').
    bandpasses : dict
        Dictionary of GalSim Roman bandpasses, typically from ``galsim.roman.getBandpasses()``.

    Returns
    -------
    bool, str
        Whether the wavelength is in the filter, and the galsim key of the filter if it is in the bandpass.

    """

    wav *= 1e3  # convert to nm for galsim
    bp = bandpasses

    # First, check for an exact match of the filter string to a bandpass key.
    if filter_string in bp:
        band = bp[filter_string]
        if band.blue_limit <= wav <= band.red_limit:
            return True, filter_string
        else:
            return False, None

    # Otherwise, look for all keys that contain the filter_string as a substring.
    matching_keys = [key for key in bp if filter_string in key]
    if not matching_keys:
        raise ValueError(
            f"Filter {filter_string} not found in bandpasses. Available filters are: {bp.keys()}"
        )

    # Check each matching key and return True on the first one whose bandpass contains the wavelength.
    for key in matching_keys:
        band = bp[key]
        if band.blue_limit <= wav <= band.red_limit:
            return True, key

    # If none of the candidate bandpasses contain the wavelength, report that it is out of band.
    return False, None


class PolychromaticPSF:
    """
    Compute and draw weighted polychromatic PSFs.

    Parameters
    ----------
    scanum : int
        Roman SCA index passed through to ``PSFObject``.
    scax, scay : float
        Source x and y positions on the SCA. Units depend on the frame: mm (analysis, default) or
        native pixels (science).
    wavelengths : array-like
        Wavelength samples in microns. Values are evaluated in the provided order.
    sed : callable, optional
        Spectral energy distribution weight function evaluated as ``sed(wav_microns)``.
        This should be in units proportional to photons/m^2/s/micron.
        If ``None``, a flat spectral weight (in lambda F_lambda) is assumed.
    ghost : bool, optional
        Whether to include the ghost path from refraction in the optical filter.
        Default: False. Only can be true if ray_trace=True.
    frame : str, optional
        Which frame to use? Options are "analysis", "science", and "fpa".

        - **analysis** : scax and scay are in mm with (0, 0) at the SCA center.
        - **science** : scax and scay are in pixels with (0, 0) at the lower-left corner (in ds9 display).
        - **fpa** : scax and scay are in pixels, in the orientation of the focal plane array
          (see ``docs/coordinates.rst``).

        The PSF is also flipped if necessary to coincide with the desired output.

        Note the parity and orientation relations: SCI +Y is the same direction as ANA -Y,
        and the FPA frame is rotated 180 degrees from the analysis frame (so FPA +X is
        ANA -X and FPA +Y is ANA -Y). The science and FPA frames are related by a flip of
        the X axis only, so SCI x + FPA x = 4087 for the same point on the detector.

    """

    def __init__(self, scanum, scax, scay, wavelengths, sed=None, ghost=False, frame="analysis"):
        self.scanum = scanum
        self.scax = scax
        self.scay = scay
        self.wavelengths = wavelengths  # replace with something better later
        self.sed = sed
        self.bandpass = galsim.roman.getBandpasses()
        self.ghost = ghost
        self.frame = frame.lower()  # make this case-insensitive
        if self.frame not in ("analysis", "science", "fpa"):
            raise ValueError(f"Unknown frame '{frame}'; options are 'analysis', 'science', and 'fpa'.")

        # convert positions to analysis frame
        if self.frame in ("science", "fpa"):
            # both are in native pixels measured from the corner of the SCA
            dx = 0.01 * (self.scax - 2043.5)
            dy = 0.01 * (self.scay - 2043.5)
            # SCI +X is ANA +X, but the FPA frame is rotated 180 degrees from the analysis frame
            self.scax = -dx if self.frame == "fpa" else dx
            self.scay = -dy

    def compute_poly_psf(
        self,
        postage_stamp_size=31,
        ovsamp=10,
        use_filter="H",
        npix_boundary=1,
        use_postage_stamp_size=None,
        ray_trace=True,
        extra_aberrations=None,
        optical_psf_only=False,
        req_in_bandpass=True,
        centerpix=True,
        cycle=9,
        mjd=None,
        perturbations=None,
    ):
        """
        Compute the polychromatic PSF by integrating monochromatic PSFs across wavelength.

        Integration uses a trapezoidal rule over the caller-provided wavelength
        nodes (internally sorted). Out-of-band nodes contribute zero. If exactly
        one node is in-band, this returns the corresponding monochromatic PSF.

        Parameters
        ----------
        postage_stamp_size : int, optional
            Size of the postage stamp to draw, in native pixels.
        ovsamp : int, optional
            The number of samples per native pixel on each axis.
        use_filter : str, optional
            The filter as a string (e.g., "H").
        use_postage_stamp_size : int, optional
            Force pupil postage stamp size instead of internal calculation. In native pixels.
        npix_boundary : int, optional
            ?
        ray_trace : bool, optional
            Whether to use ray tracing. (Only turn off for testing.)
        extra_aberrations: float array, optional
            Parameters corresponding to zernike polynomials for introducing aberrations that
            add to the optical path length and produce different aberrations. Supports up to
            5 parameters (Z2, Z3, Z4, Z5, and Z6 in that order). The effects of each polynomial
            are as follows:
            Z2: horizontal centering
            Z3: vertical centering
            Z4: focus
            Z5: astigmatism
            Z6: also astigmatism
        optical_psf_only : bool, optional
            Whether to draw the optical PSF only.
        req_in_bandpass : bool, optional
            Whether to only accept in-band light (turning this on will make things faster
            for some settings, but will miss detail in the PSF from out-of-band leakage).
            Recommend True for fast computation, False for best accuracy.
        centerpix : bool, optional
            Whether to center the PSF on a pixel.
        cycle : int, optional
            Which cycle to use for the Zernike modes.
        mjd : float, optional
            The MJD to use for the optical model.
        perturbations : dict, optional
            If provided, replaces the default perturbations in the cycle model. Should contain the keys:

            - ``arr``: np.ndarray

            - ``basis``: psfsim.basis.RomanBasisSet

              The number of basis modes, ``basis.N``, must equal the number of entries in ``arr``.

        Returns
        -------
        np.ndarray
            The polychromatic PSF as a 2D numpy array.

        """

        wavelengths = np.asarray(self.wavelengths, dtype=float)
        if wavelengths.ndim != 1 or wavelengths.size == 0:
            raise ValueError("wavelengths must be a non-empty 1D sequence in microns.")

        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        if np.any(np.diff(wavelengths) <= 0.0):
            raise ValueError("wavelengths must be unique values for trapezoidal integration.")

        in_band_info = [inBandpass(wav, use_filter, self.bandpass) for wav in wavelengths]
        in_band_mask = np.array([is_in for is_in, _ in in_band_info], dtype=bool)
        n_in_band = int(np.count_nonzero(in_band_mask))

        if n_in_band == 0:
            raise ValueError(
                f"No in-band wavelength nodes found for filter '{use_filter}'. "
                f"Provided range: [{wavelengths.min():.6g}, {wavelengths.max():.6g}] microns."
            )

        if (not ray_trace) and (self.ghost):
            raise ValueError("ghost must be False")

        def _compute_mono_image(wav):
            this_psf = PSFObject(
                self.scanum,
                self.scax,
                self.scay,
                wav,
                postage_stamp_size=postage_stamp_size,
                ovsamp=ovsamp,
                use_filter=use_filter,
                npix_boundary=npix_boundary,
                use_postage_stamp_size=use_postage_stamp_size,
                ray_trace=ray_trace,
                extra_aberrations=extra_aberrations,
                cycle=cycle,
                mjd=mjd,
                ghost=self.ghost,
                perturbations=perturbations,
            )
            this_psf.get_optical_psf()
            if optical_psf_only:
                return this_psf.Optical_PSF

            this_psf.get_image_from_Intensity(centerpix=centerpix, reflect=True, tophat=True)
            return this_psf.detector_image

        # output flips (the PSF is computed in the analysis frame)
        _px = -1 if self.frame == "fpa" else 1
        _py = -1 if self.frame in ("science", "fpa") else 1

        if n_in_band == 1:
            wav = wavelengths[in_band_mask][0]
            chromatic_psf = _compute_mono_image(wav).astype(float, copy=True)
            total_flux = np.sum(chromatic_psf)
            if total_flux == 0.0:
                raise ValueError("Monochromatic PSF has zero flux for the only in-band wavelength node.")
            chromatic_psf /= total_flux
            chromatic_psf = chromatic_psf[::_py, ::_px]
            self.chromatic_psf = chromatic_psf
            return chromatic_psf

        trap_weights = np.empty_like(wavelengths)
        trap_weights[0] = 0.5 * (wavelengths[1] - wavelengths[0])
        trap_weights[-1] = 0.5 * (wavelengths[-1] - wavelengths[-2])
        trap_weights[1:-1] = 0.5 * (wavelengths[2:] - wavelengths[:-2])

        chromatic_psf = None
        for i in range(wavelengths.size):
            wav = wavelengths[i]
            quad_weight = trap_weights[i]
            is_in_bandpass, filter_key = in_band_info[i]
            if req_in_bandpass and not is_in_bandpass:
                continue

            if self.sed is not None:
                wav_nm = wav * 1e3
                bp = self.bandpass[filter_key]
                integrand_weight = bp(wav_nm) * self.sed(wav)
            else:
                integrand_weight = 1.0

            weight = quad_weight * integrand_weight
            if weight == 0.0:
                continue

            mono_image = _compute_mono_image(wav)
            if chromatic_psf is None:
                chromatic_psf = np.zeros_like(mono_image, dtype=float)
            chromatic_psf += weight * mono_image

        if chromatic_psf is None:
            raise ValueError(
                "No flux accumulated in polychromatic PSF after applying bandpass/SED weights; "
                "check wavelength nodes, filter, and SED values."
            )

        total_flux = np.sum(chromatic_psf)
        if total_flux == 0.0:
            raise ValueError(
                "No flux accumulated in polychromatic PSF after integration; "
                "check wavelength nodes, filter, and SED values."
            )
        chromatic_psf /= total_flux

        # Convert to the desired coordinate system
        chromatic_psf = chromatic_psf[::_py, ::_px]

        self.chromatic_psf = chromatic_psf
        return chromatic_psf
