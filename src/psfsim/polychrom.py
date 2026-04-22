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
        Source x-position on the SCA.
    scay : float
        Source y-position on the SCA.
    wavelengths : array-like
        Wavelength samples in microns. Values are evaluated in the provided order.
    sed : callable, optional
        Spectral energy distribution weight function evaluated as ``sed(wav_microns)``.
        If ``None``, a flat spectral weight is assumed.
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
    ):
        """
        Compute a normalized polychromatic PSF from monochromatic PSF samples.

        The accumulator sums per-wavelength contributions of
        ``bandpass(wavelength_nm) * sed(wavelength_microns) * PSF(wavelength)``,
        with unit weight for ``sed`` when ``sed=None``. The returned image is then
        normalized to unit sum.

        Notes
        -----
        This method currently applies one contribution per entry in ``self.wavelengths``
        (no explicit bin-width quadrature factor), so the approximation depends on
        the supplied wavelength sampling.
        """
        # I'm going to accumulate iteratively for now to save on memory, but open to changing later
        if optical_psf_only:
            chromatic_psf = np.zeros((2048, 2048))
        else:
            chromatic_psf = np.zeros((postage_stamp_size * ovsamp, postage_stamp_size * ovsamp))
        for wav in self.wavelengths:
            is_in_bandpass, filter_key = inBandpass(wav, use_filter, self.bandpass)
            if is_in_bandpass:
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
                    if self.sed is not None:
                        # Convert wavelength from microns to nm for GalSim Bandpass evaluation
                        wav_nm = wav * 1e3
                        bp = self.bandpass[filter_key]
                        weight = bp(wav_nm) * self.sed(wav)
                    else:
                        # If no SED is provided, assume flat response
                        weight = 1.0

                    chromatic_psf += weight * this_psf.Optical_PSF
                    continue  # set to the value that should come from Charuhas' branch

                this_psf.get_image_from_Intensity(centerpix=True, reflect=True, tophat=True)
                if self.sed is not None:
                    # Convert wavelength from microns to nm for GalSim Bandpass evaluation
                    wav_nm = wav * 1e3
                    bp = self.bandpass[filter_key]
                    weight = bp(wav_nm) * self.sed(wav)
                else:
                    # If no SED is provided, assume flat response
                    weight = 1.0

                chromatic_psf += weight * this_psf.detector_image
        total_flux = np.sum(chromatic_psf)
        if total_flux == 0.0:
            raise ValueError(
                f"No flux accumulated in polychromatic PSF; "
                f"check that the provided wavelengths fall within the '{use_filter}' bandpass."
            )
        chromatic_psf /= total_flux
        self.chromatic_psf = chromatic_psf
        return chromatic_psf
