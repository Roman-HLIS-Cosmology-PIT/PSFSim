"""Test function for indices of refraction."""

import numpy as np
from psfsim.mirror_properties import ag_epsilon, reflect_RB_off_mirror


def test_ag():
    """Test function for bare silver mirror."""

    # Angle of incidence grid for test
    thetas = np.array([0.0, 20.0, 45.0], dtype=np.float64) * np.pi / 180.0

    for i in range(81):
        wl = 0.3 + 0.025 * i  # wavelength in microns

        # index of refraction
        epsilon = ag_epsilon(wl)
        n = (epsilon + 0j) ** 0.5  # complex index of refraction
        assert n.imag > 0
        if i == 2:
            assert np.abs(0.08443 + 1.341j - n) < 0.001
        if i == 10:
            assert np.abs(0.053265 + 3.4848j - n) < 0.001
        if i == 80:
            assert np.abs(0.56027 + 16.275j - n) < 0.001
        s = f"{wl:5.3f} {n.real:7.4f} {n.imag:7.4f}"

        # get reflections off bare silver
        # wl in mm
        RS, RP = reflect_RB_off_mirror(thetas, wl / 1.0e3, thickness=0.0)
        power_RS = np.abs(RS) ** 2
        power_RP = np.abs(RP) ** 2
        dphase = np.angle(RP / RS)
        # check S=P at normal incidence
        assert np.abs(power_RS - power_RP)[0] < 1.0e-5
        assert np.abs(dphase[0]) < 1.0e-5
        # compare to computation from refractiveindex.info
        if i == 2:
            assert np.abs(power_RS[2] - 0.92733) < 1.0e-4
            assert np.abs(power_RP[2] - 0.85995) < 1.0e-4
            assert np.abs(dphase[2] * 180.0 / np.pi - 49.939) < 0.01
        if i == 8:
            assert np.abs(power_RS[2] - 0.98634) < 1.0e-4
            assert np.abs(power_RP[2] - 0.97288) < 1.0e-4
            assert np.abs(dphase[2] * 180.0 / np.pi - 25.349) < 0.01
        for j in range(3):
            s += f" {power_RS[j]:6.4f} {power_RP[j]:6.4f} {dphase[j] * 180.0 / np.pi:6.2f}"

        print(s)
