import galsim.roman
import numpy as np

from .psfobject import PSFObject


def inBandpass(wav, filter_string):
    """
    Compute whether a wavelength is in a filter.

    Parameters
    ----------
    wav : float
        Wavelength in microns
    filter_string: str
        String to identify the filter. Can either be just a letter or letter + wl (e.g. 'H' or 'H158').

    Returns
    -------
    bool, str
        Whether the wavelength is in the filter, and the galsim key of the filter if it is in the bandpass.
    """
    wav *= 1e3  # convert to nm for galsim
    bp = galsim.roman.getBandpasses()
    for key in bp:
        if filter_string in key:
            if wav >= bp[key].blue_limit and wav <= bp[key].red_limit:
                return True, key
            else:
                return False, None
    raise ValueError(f"Filter {filter_string} not found in bandpasses. Available filters are: {bp.keys()}")


class PolychromaticPSF:
    """
    Class to compute and draw polychromatic PSFs
    """

    def __init__(self, scanum, scax, scay, wavelengths, sed=None):
        self.scanum = scanum
        self.scax = scax
        self.scay = scay
        self.wavelengths = wavelengths  # replace with something better later
        if sed is not None:
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
    ):
        """
        Compute the polychromatic PSF by summing over monochromatic PSFs at different wavelengths.
        """
        # I'm going to accumulate iteratively for now to save on memory, but open to changing later
        chromatic_psf = np.zeros((postage_stamp_size * ovsamp, postage_stamp_size * ovsamp))
        for wav in self.wavelengths:
            is_in_bandpass, filter_key = inBandpass(wav, use_filter)
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
                this_psf.get_image_from_Intensity(centerpix=True, reflect=True, tophat=True)
                if self.sed is not None:
                    # Convert wavelength from microns to nm for GalSim Bandpass evaluation
                    wav_nm = wav * 1e3
                    bp = self.bandpass[filter_key]
                    weight = bp(wav_nm) * self.sed(wav)
                else:
                    # If no SED is provided, assume flat response
                    weight = 1.0

                chromatic_psf += (
                    weight * this_psf.detector_image
                )  # set to the value that should come from Charuhas' branch
        total_flux = np.sum(chromatic_psf)
        if total_flux == 0.0:
            raise ValueError(
                f"No flux accumulated in polychromatic PSF; "
                f"check that the provided wavelengths fall within the '{use_filter}' bandpass."
            )
        chromatic_psf /= total_flux
        self.chromatic_psf = chromatic_psf
        return chromatic_psf
