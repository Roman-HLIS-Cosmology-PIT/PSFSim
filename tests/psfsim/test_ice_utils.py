import numpy as np
from psfsim.psfobject import PSFObject
from psfsim.utils.ice_utils import _parse_filter_name, get_ice_transmission


def test_parse_filtername():
    """Test parsing filter name"""
    fname = _parse_filter_name("H158")
    assert fname == "H158"
    fname = _parse_filter_name("H")
    assert fname == "H158"


def test_main_interface():
    """Test main user interface"""
    T_noice = get_ice_transmission(filter_name="H158", ice_thickness=0.0)
    assert all(x == 1.0 for x in T_noice[1])

    def ice_function(scanum, scax, scay, wave, ice_thickness):
        return 1

    # check dummy function
    T_ice = get_ice_transmission(filter_name="H158", ice_thickness=0.1, ice_response=ice_function)
    assert np.array_equal(T_noice[0], T_ice[0])
    assert np.array_equal(T_noice[1], T_ice[1])


def test_ice_accuracy():
    """Test if helper function returns the same flux response in the presence of ice
    as that in PSF calculation.
    """
    T_ice = get_ice_transmission(filter_name="H158", ice_thickness=0.1)
    wav = T_ice[0][0]
    response_func = T_ice[1][0]

    def _compute_mono_image(wav, use_filter="H", ice_thickness=0.0):
        this_psf = PSFObject(
            1,
            0,
            0,
            wav,
            use_filter=use_filter,
            ice_thickness=ice_thickness,
        )

        this_psf.get_image_from_Intensity(centerpix=True, reflect=False, tophat=True)
        return this_psf.detector_image

    ## calculate response from flux from images
    mono_image_ice = _compute_mono_image(wav, ice_thickness=0.1)
    mono_image_noice = _compute_mono_image(wav, ice_thickness=0.0)
    response = np.sum(mono_image_ice) / np.sum(mono_image_noice)
    diff = response_func / response - 1
    # check that response is accurate to 0.01% or better
    assert np.abs(diff) < 1e-4
