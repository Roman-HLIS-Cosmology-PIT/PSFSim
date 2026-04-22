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
    scax : float
        Source x-position on the SCA, in mm.
    scay : float
        Source y-position on the SCA, in mm.
    wavelengths : array-like
        Wavelength samples in microns. Values are evaluated in the provided order.
    sed : callable, optional
        Spectral energy distribution weight function evaluated as ``sed(wav_microns)``.
        This should be in units proportional to photons/m^2/s/micron.
        If ``None``, a flat spectral weight (in lambda F_lambda) is assumed.

    """

    def __init__(self, scanum, scax, scay, wavelengths, sed=None):
        self.scanum = scanum
        self.scax = scax
        self.scay = scay
        self.wavelengths = wavelengths  # replace with something better later
        self.sed = sed
        self.bandpass = galsim.roman.getBandpasses()

    def compute_poly_psf(
        self,
        postage_stamp_size=31,
        ovsamp=10,
        use_filter="H",
        npix_boundary=1,
        use_postage_stamp_size=None,
        ray_trace=True,
        add_focus=None,
        optical_psf_only=False,
        req_in_bandpass=True,
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
            Force pupil postage stamp size instead of internal calculation.
        npix_boundary : int, optional
            ?
        ray_trace : bool, optional
            Whether to use ray tracing. (Only turn off for testing.)
        add_focus : variable
            Parameter for adding focus.
        optical_psf_only : bool, optional
            Whether to draw the optical PSF only.
        req_in_bandpass : bool, optional
            Whether to only accept in-band light (turning this on will make things faster
            for some settings, but will miss detail in the PSF from out-of-band leakage).
            Recommend True for fast computation, False for best accuracy.
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
                add_focus=add_focus,
            )
            this_psf.get_optical_psf()
            if optical_psf_only:
                return this_psf.Optical_PSF

            this_psf.get_image_from_Intensity(centerpix=True, reflect=True, tophat=True)
            return this_psf.detector_image

        if n_in_band == 1:
            wav = wavelengths[in_band_mask][0]
            chromatic_psf = _compute_mono_image(wav).astype(float, copy=True)
            total_flux = np.sum(chromatic_psf)
            if total_flux == 0.0:
                raise ValueError("Monochromatic PSF has zero flux for the only in-band wavelength node.")
            chromatic_psf /= total_flux
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
        self.chromatic_psf = chromatic_psf
        return chromatic_psf
