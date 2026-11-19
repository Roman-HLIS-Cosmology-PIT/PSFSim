"""
This script figures out the offsets from the ideal WFI geometry that should go into offsets.py,
based on the astrometric distortions.

It is separate and intended to be called once per model update --- don't run this as part of
your workflow, since it messes up offsets.py (although it puts it back, this won't be good
for some parallel workflows).

Call this with the SIAF file and cycle number as the command-line argument.

There needs to be a tmpdir/ directory where intermediate products can be stashed.

This runs through 16 iterations (way more than we need) and returns the parameters and residuals that
should be coded into offsets.py.

"""

import copy
import sys
from pathlib import Path

import numpy as np
import pysiaf
from psfsim import offsets
from psfsim.aberration_models import extract_basis_coefs
from psfsim.wfi_data import scapos

npar = 9
steps = [1e-4, 1e-4, 1e-4, 0.1, 0.1, 0.05, 0.01, 0.01, 0.01]


FPOSLIST = [0.0]


def _residuals(p, cycle):
    """
    Gets residuals for a given parameters. Vector p has length npar.

    The returned residuals are in order of:
    d(delta x)/dxfpa, d(delta x)/dyfpa, (delta x),
    d(delta y)/dxfpa, d(delta y)/dyfpa, (delta y),
    d(delta focus)/dxfpa, d(delta focus)/dyfpa, (delta focus),

    Here ``cycle`` is the integer cycle number.

    """

    # checks
    assert len(p) == len(steps) == npar

    ost = offsets.offsets[cycle]

    # set biases
    stash = {
        "fbias_offset": copy.deepcopy(ost["fbias_offset"]),
        "sm_offset": copy.deepcopy(ost["sm_offset"]),
        "fpa_offset": copy.deepcopy(ost["fpa_offset"]),
    }
    ost["fbias_offset"]["VERTICAL"] = p[0]
    ost["fbias_offset"]["HORIZONTAL"] = p[1]
    ost["sm_offset"]["DZ"] = p[2]
    ost["fpa_offset"]["DX"] = p[3]
    ost["fpa_offset"]["DY"] = p[4]
    ost["fpa_offset"]["DZ"] = p[5]
    ost["fpa_offset"]["TILT"] = p[6]
    ost["fpa_offset"]["TIP"] = p[7]
    ost["fpa_offset"]["ROLL"] = p[8]

    rc, dz, dpos = extract_basis_coefs(
        "tmpdir/zernikes.csv",
        "H",
        nn=64,
        return_coefs=True,
        verbose=False,
        fpos=FPOSLIST[0],
        return_delta_zernike=True,
        cycle=cycle,
    )

    # restore
    for tg in ["fbias_offset", "sm_offset", "fpa_offset"]:
        ost[tg] = stash[tg]

    return rc.T.flatten(), dz, dpos


def _setup(infile):
    """Convert SIAF to Zernike model."""

    nchip = 18
    ns = 3
    npt = nchip * ns**2
    fpos = np.zeros((npt, 2))

    rsiaf = pysiaf.Siaf("Roman", basepath=Path(infile).parent, filename=Path(infile).name)
    v2c, v3c = rsiaf["WFI_CEN"].idl_to_tel(0, 0)
    v2u, v3u = rsiaf["WFI_CEN"].idl_to_tel(0, 120)
    orient = np.arctan2(v3u - v3c, v2u - v2c)  # orientation of WFI +Y relative to V2/V3
    print("# v2c, v3c, orient =", v2c, v3c, orient)
    with open("tmpdir/zernikes.csv", "w") as f:
        f.write("global_x,global_y,wavelength,Z1,Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9,Z10,Z11\n")
        for i in range(npt):
            sca = i // ns**2 + 1
            posy = (i % ns**2) // ns
            posx = ns - 1 - i % ns
            sci_x = posx * 4088 / (ns - 1) - 0.5
            sci_y = posy * 4088 / (ns - 1) - 0.5
            v2, v3 = rsiaf[f"WFI{sca:02d}_FULL"].sci_to_tel(sci_x, sci_y)  # arcsec
            dv2, dv3 = v2 - v2c, v3 - v3c
            fpos[i, 0] = (dv2 * np.sin(orient) - dv3 * np.cos(orient)) / 3600.0
            fpos[i, 1] = -(dv2 * np.cos(orient) + dv3 * np.sin(orient)) / 3600.0
            print(f"{sca:2d} {sci_x:6.1f} {sci_y:6.1f} {fpos[i, 0]:7.4f} {fpos[i, 1]:7.4f}")
            xfpa = scapos[sca - 1, 0] - 0.01 * (sci_x - 2043.5)
            yfpa = scapos[sca - 1, 1] + 0.01 * (sci_y - 2043.5)
            f.write(f"{xfpa},{yfpa},1.58,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n")

    FPOSLIST[0] = fpos


def _run(infile, cycle):
    """Run the optimizer."""

    _setup(infile)

    p = np.zeros((npar,))
    r = _residuals(p, cycle)[0]

    # build matrix
    M = np.zeros((npar, npar))
    for i in range(npar):
        dp = np.zeros((npar,))
        dp[i] = steps[i] / 2.0
        M[:, i] = (_residuals(p + dp, cycle)[0] - _residuals(p - dp, cycle)[0]) / steps[i]
        print(":", i)

    for j in range(16):
        r, dz, dpos = _residuals(p, cycle)
        s = f"{j:2d}\n"
        for k in range(9):
            s += f" {p[k]:16.9E}"
        s += "\n"
        for k in range(9):
            s += f" {r[k]:16.9E}"
        s += "\n"
        for k in range(np.shape(dz)[-1]):
            s += f" {np.mean(dz**2, axis=0)[k]**0.5:16.9E}"
        s += "\n"
        print(s)
        change = -np.linalg.solve(M, r)
        p += change

        print(np.amax(np.abs(change)))

    np.savetxt("tmpdir/astrom_err.dat", np.hstack((FPOSLIST[0], dpos)))


_run(sys.argv[1], int(sys.argv[2]))
