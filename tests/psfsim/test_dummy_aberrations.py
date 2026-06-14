"""Test for extract_basis_coefs."""

from urllib.request import urlretrieve

import numpy as np
from psfsim.aberration_models import extract_basis_coefs
from psfsim.basis import basis_set


def test_extraction(tmp_path):
    """
    Test for extract_basis_coefs.

    Parameters
    ----------
    tmp_path : str or str-like
        Location of the directory to download files.

    Returns
    -------
    None

    """

    tmp_path = str(tmp_path)

    # remind us of where things start
    for obj in basis_set.basis:
        b = basis_set.basis[obj]
        print(obj, b.start, b.N, b)

    # filters
    wfi_filters = {
        "H": "158",
        "R": "062",
        "Z": "087",
        "Y": "106",
        "J": "129",
        "F": "184",
        "K": "213",
        "W": "146",
    }
    flts = list(wfi_filters.keys())
    nmodes = basis_set.N

    # we'll only do 2
    fmax = 2
    data = np.zeros((nmodes, fmax))

    nmin = None
    p = None
    c = 3
    for i in range(fmax):  # only do the first 2 right now
        filter = flts[i]

        print(f"\n---\nFILTER {filter}{wfi_filters[filter]}\n")
        lname = tmp_path + f"/dummy_aberrations{wfi_filters[filter]}.csv"
        url = f"https://github.com/Roman-HLIS-Cosmology-PIT/PSFSim/wiki/files/dummy_aberrations{wfi_filters[filter]}.csv"
        print(url, "->", lname)
        urlretrieve(url, lname)
        p = extract_basis_coefs(
            lname,
            filter,
            nn=64,
            pars_input=p,
            nmin=nmin,
            smin=0.05,
            c=c,
        )
        print(p)
        data[:, i] = p

        nmin = basis_set.basis["S1"].start
        c = 1

    assert 4 < np.count_nonzero(np.abs(data) > 2e-4) < 10

    # some checks on the data
    assert np.all(np.abs(data) < 1e-3)
    assert np.allclose(data[:nmin, 0], data[:nmin, 1])

    # these you might have to update if you change the model
    assert 23 < 1.0e6 * data[0, 0] < 25
    assert -2 < 1.0e6 * data[nmin, 0] < 1
    assert -15 < 1.0e6 * data[nmin, 1] < -12
