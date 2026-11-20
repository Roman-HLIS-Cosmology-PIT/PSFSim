"""Utility functions for handling SCAs, etc."""

import numpy as np

from ..wfi_data import scapos


def get_closest_sca_from_fpa(fpapos):
    """Gets the SCA that is closest to that point on the focal plane.

    Parameters
    ----------
    fpapos: np.array of float
        A length-2 array (x and y position in mm).

    Returns
    -------
    int
        The closest SCA number in the range 1..18.

    """

    return 1 + np.argmin(np.sum((np.array(fpapos)[None, :] - scapos) ** 2, axis=1))
