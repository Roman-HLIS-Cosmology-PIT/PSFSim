"""
This has utility functions for the optical perturbations.

It will be updated with additional cycles as needed.

"""

from importlib.resources import files

import numpy as np


def _load_cycle10():
    """Makes a dictionary with the Cycle 10 data."""

    # the data itself
    infile = files("psfsim.data").joinpath("cycle10_perturbations.csv")  # reads data directory
    perturbations = np.loadtxt(infile, delimiter=",", comments="#") * 1.0e-6  # convert from nm to mm

    # now read the header to get the order of the filters
    with open(infile, "r") as f:
        header = f.readline()
    col = {}
    j = 0
    for c in header:
        if c.isalpha():
            col[c] = j
            j += 1

    return {
        "columns": col,  # dictionary mapping filter letters to columns
        "data": perturbations,  # in mm
    }


# load the data
cycle10data = _load_cycle10()


def cycle10_perturbations(use_filter):
    """
    Gets the perturbation vector for the designated filter.

    Parameters
    ----------
    use_filter : str
        The filter name (starts with unique letter: Y, J, etc.)

    Returns
    -------
    np.ndarray
        The perturbations (in mm) for that filter.

    """

    return cycle10data["data"][:, cycle10data["columns"][use_filter[0]]]
