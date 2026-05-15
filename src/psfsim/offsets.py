"""
Perturbation parameters for the optical model.

Note these aren't necessarily the "real" parameters, since there are some degeneracies,
and we used SVD to select out the linear combinations of parameters that "matter".

All the parameters are implemented as dictionaries, so the fitting script (which is only
run once as a stand-alone script) can take derivatives with respect to them.

The parameter values were determined by ``scripts/cycle10/fit_offsets.py``.

"""


# Field bias offsets (in degrees)
fbias_offset = {
    "VERTICAL": -1.802006222e-03,  # change in field bias (relative to fiducial 0.496)
    "HORIZONTAL": 6.975702366e-05,  # transverse offset
}

# SM offset
sm_offset = {
    "DZ": 6.254670791e-03  # offset of SM (+ = away from PM)
}

# FPA offsets
#
# The sense of this is that to get from *ideal* FPA (x, y) to *true* FPA (x, y),
# you need to apply:
#
# xy_fpa_true = M @ xy_fpa_ideal - offset
#
# offset = (DX, DY)
# M = [[cos ROLL, -sin ROLL], [sin ROLL, cos ROLL]]

fpa_offset = {
    "DX": -2.737383133e-01,  # x offset in mm
    "DY": -2.286019099e-01,  # y offset in mm
    "DZ": 2.416716816e-01,  # z offset in mm
    "TILT": 1.339988060e-02,  # rotation around x axis in degrees
    "TIP": -4.526191624e-04,  # rotation around y axis in degrees
    "ROLL": 1.961296976e-02,  # rotation around z axis in degrees
}
