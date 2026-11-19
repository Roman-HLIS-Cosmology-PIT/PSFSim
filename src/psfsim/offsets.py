"""
Perturbation parameters for the optical model.

Note these aren't necessarily the "real" parameters, since there are some degeneracies,
and we used SVD to select out the linear combinations of parameters that "matter".

All the parameters are implemented as dictionaries, so the fitting script (which is only
run once as a stand-alone script) can take derivatives with respect to them.

The parameter values were determined by ``scripts/cycle10/fit_offsets.py``.

This is a nested dictionary with the cycle number at the top level. Below that is:

- fbias_offset  # offset in degrees
  - VERTICAL  # change in field bias (relative to fiducial 0.496)
  - HORIZONTAL  # transverse offset

- sm_offset  # offset of SM (+ = away from PM)

- fpa_offset
  - DX  # x offset in mm
  - DY  # y offset in mm
  - DZ  # z offset in mm
  - TILT  # rotation around x axis in degrees
  - TIP  # rotation around y axis in degrees
  - ROLL  # rotation around z axis in degrees

The sense of this is that to get from *ideal* FPA (x, y) to *true* FPA (x, y), you need to apply::

  xy_fpa_true = M @ xy_fpa_ideal - offset

  offset = (DX, DY)
  M = [[cos ROLL, -sin ROLL], [sin ROLL, cos ROLL]]

"""

null_offsets = {
    "fbias_offset": {
        "VERTICAL": 0.0,
        "HORIZONTAL": 0.0,
    },
    "sm_offset": {
        "DZ": 0.0,
    },
    "fpa_offset": {
        "DX": 0.0,
        "DY": 0.0,
        "DZ": 0.0,
        "TILT": 0.0,
        "TIP": 0.0,
        "ROLL": 0.0,
    },
}

# index by cycle
offsets = {
    0: null_offsets,
    9: null_offsets,
    10:  # cycle 10
    {
        "fbias_offset": {
            "VERTICAL": -1.802006222e-03,
            "HORIZONTAL": 6.975702366e-05,
        },
        "sm_offset": {
            "DZ": 6.254670791e-03,
        },
        "fpa_offset": {
            "DX": -2.737383133e-01,
            "DY": -2.286019099e-01,
            "DZ": 2.416716816e-01,
            "TILT": 1.339988060e-02,
            "TIP": -4.526191624e-04,
            "ROLL": 1.961296976e-02,
        },
    },
    261021:  # SIAF update
    {
        "fbias_offset": {
            "VERTICAL": -1.802006222e-03,
            "HORIZONTAL": 6.975702366e-05,
        },
        "sm_offset": {
            "DZ": 6.254670791e-03,
        },
        "fpa_offset": {
            "DX": -2.737383133e-01,
            "DY": -2.286019099e-01,
            "DZ": 2.416716816e-01,
            "TILT": 1.339988060e-02,
            "TIP": -4.526191624e-04,
            "ROLL": 1.961296976e-02,
        },
    },
}
