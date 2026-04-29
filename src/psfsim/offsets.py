"""
Perturbation parameters for the optical model.

Note these aren't necessarily the "real" parameters, since there are some degeneracies,
and we used SVD to select out the linear combinations of parameters that "matter".

"""


import numpy as np

# Field bias offset
fbias_offset = np.array([-0.00187, 0.0001])  # degrees

# SM offset
sm_offset = {
    "DZ": 0.0063  # offset of SM (+ = away from PM)
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
    "DX": -0.2741,  # x offset in mm
    "DY": -0.2282,  # y offset in mm
    "DZ": 0.244,  # z offset in mm
    "TILT": 0.012,  # rotation around x axis in degrees
    "ROLL": 0.0195,  # rotation around z axis in degrees
}
