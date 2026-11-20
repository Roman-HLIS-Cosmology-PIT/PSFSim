import numpy as np
from psfsim.utils.fpa_utils import get_closest_sca_from_fpa


def test_sca():
    """Test function for finding the nearest SCA."""

    im = np.zeros((151, 151), dtype=np.int8)
    for i in range(151):
        for j in range(151):
            im[j, i] = get_closest_sca_from_fpa([-150 + 2 * i, -150 + 2 * j])

    assert np.all(
        im[40:101:20, 15::15]
        == np.array(
            [
                [9, 6, 6, 3, 3, 12, 15, 15, 18, 18],
                [9, 5, 5, 2, 2, 11, 14, 14, 18, 18],
                [8, 4, 4, 1, 1, 10, 13, 13, 17, 17],
                [7, 7, 4, 1, 1, 10, 13, 16, 16, 16],
            ]
        )
    )
