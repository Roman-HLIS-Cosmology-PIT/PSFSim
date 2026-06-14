"""
This script figures out the offsets from the ideal WFI geometry that should go into offsets.py,
based on the astrometric distortions.

It is separate and intended to be called once per model update --- don't run this as part of
your workflow, since it messes up offsets.py (although it puts it back, this won't be good
for some parallel workflows).

Requires the directories:

- ``indata/`` : should have the Cycle 10 .csv files for WFI Imaging Mode.

This runs through 16 iterations (way more than we need) and returns the parameters and residuals that
should be coded into offsets.py.

"""

import copy

import numpy as np
from psfsim import offsets
from psfsim.aberration_models import extract_basis_coefs

npar = 9
steps = [1e-4, 1e-4, 1e-4, 0.1, 0.1, 0.05, 0.01, 0.01, 0.01]


def _residuals(p):
    """
    Gets residuals for a given parameters. Vector p has length npar.

    The returned residuals are in order of:
    d(delta x)/dxfpa, d(delta x)/dyfpa, (delta x),
    d(delta y)/dxfpa, d(delta y)/dyfpa, (delta y),
    d(delta focus)/dxfpa, d(delta focus)/dyfpa, (delta focus),

    """

    # checks
    assert len(p) == len(steps) == npar

    # set biases
    stash = {
        "fbias_offset": copy.deepcopy(offsets.fbias_offset),
        "sm_offset": copy.deepcopy(offsets.sm_offset),
        "fpa_offset": copy.deepcopy(offsets.fpa_offset),
    }
    offsets.fbias_offset["VERTICAL"] = p[0]
    offsets.fbias_offset["HORIZONTAL"] = p[1]
    offsets.sm_offset["DZ"] = p[2]
    offsets.fpa_offset["DX"] = p[3]
    offsets.fpa_offset["DY"] = p[4]
    offsets.fpa_offset["DZ"] = p[5]
    offsets.fpa_offset["DZ"] = p[5]
    offsets.fpa_offset["TILT"] = p[6]
    offsets.fpa_offset["TIP"] = p[7]
    offsets.fpa_offset["ROLL"] = p[8]

    rc = extract_basis_coefs(
        "indata/WIM_F158_zernikes_cycle10.csv", "H", nn=64, return_coefs=True, verbose=False
    )

    # restore
    for k in offsets.fbias_offset:
        offsets.fbias_offset[k] = stash["fbias_offset"][k]
    for k in offsets.sm_offset:
        offsets.sm_offset[k] = stash["sm_offset"][k]
    for k in offsets.fpa_offset:
        offsets.fpa_offset[k] = stash["fpa_offset"][k]

    return rc.T.flatten()


def _run():
    """Run the optimizer."""

    p = np.zeros((npar,))
    r = _residuals(p)

    # build matrix
    M = np.zeros((npar, npar))
    for i in range(npar):
        dp = np.zeros((npar,))
        dp[i] = steps[i] / 2.0
        M[:, i] = (_residuals(p + dp) - _residuals(p - dp)) / steps[i]
        print(":", i)

    for j in range(16):
        r = _residuals(p)
        s = f"{j:2d}\n"
        for k in range(9):
            s += f" {p[k]:16.9E}"
        s += "\n"
        for k in range(9):
            s += f" {r[k]:16.9E}"
        s += "\n"
        print(s)
        change = -np.linalg.solve(M, r)
        p += change

        print(np.amax(np.abs(change)))


_run()
